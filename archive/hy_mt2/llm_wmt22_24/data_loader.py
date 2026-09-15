"""
Data Loader and Preprocessing for Rexhaif/wmt22-24 Multi-Lingual Translation.

Extracts, deduplicates, and splits WMT22-24 reference translations for Hy-MT2-1.8B.
Supports en-zh, zh-en, en-de, and en-ja language pairs.
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple, Any, Optional
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase


LANG_MAP = {
    "zh": "Chinese",
    "en": "English",
    "de": "German",
    "ja": "Japanese",
    "ru": "Russian",
    "es": "Spanish",
    "cs": "Czech",
}


def format_translation_prompt(
    tokenizer: PreTrainedTokenizerBase,
    src_text: str,
    lp: str,
    ref_text: Optional[str] = None,
) -> Dict[str, Any]:
    """Formats a translation example into Hy-MT2 chat template."""
    src_lang, tgt_lang = lp.split("-")
    src_name = LANG_MAP.get(src_lang, src_lang)
    tgt_name = LANG_MAP.get(tgt_lang, tgt_lang)

    user_prompt = f"Translate the following text from {src_name} to {tgt_name}:\n{src_text}"
    messages = [{"role": "user", "content": user_prompt}]
    
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    if ref_text is None:
        return {"prompt_text": formatted_prompt, "src_text": src_text, "lp": lp}

    full_text = formatted_prompt + ref_text + (tokenizer.eos_token or "")
    return {
        "full_text": full_text,
        "prompt_text": formatted_prompt,
        "src_text": src_text,
        "ref_text": ref_text,
        "lp": lp,
    }


def load_wmt22_24_splits(
    tokenizer: PreTrainedTokenizerBase,
    target_lps: Optional[List[str]] = None,
    val_per_lp: int = 15,          # Lean validation set (fast periodic validation)
    test_per_lp: int = 250,        # Large test set (statistically rigorous SacreBLEU)
    max_train: int = 100000,       # Near 100K rows for full training adaptation
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Loads WMT22-24 reference translations and splits into Train (near 100K rows),
    Val (lean ~100 rows), and Test (large ~1,500-2,000 rows) with zero data leakage.
    """
    if target_lps is None:
        target_lps = ["zh-en", "en-zh", "en-de", "de-en", "en-ja", "ja-en", "en-ru", "en-es"]

    print(f"Loading Rexhaif/wmt22-24 dataset for {len(target_lps)} language pairs...")
    ds = load_dataset("Rexhaif/wmt22-24", split="train")

    # 1. Hold out test set (strictly deduplicated source sentences)
    test_src_by_lp: Dict[str, set] = {lp: set() for lp in target_lps}
    val_src_by_lp: Dict[str, set] = {lp: set() for lp in target_lps}

    raw_test: List[Dict[str, Any]] = []
    raw_val: List[Dict[str, Any]] = []
    raw_train: List[Dict[str, Any]] = []

    # First pass: collect test rows
    for ex in ds:
        lp = ex["lp"]
        if lp in test_src_by_lp and len(test_src_by_lp[lp]) < test_per_lp:
            if ex["src"] not in test_src_by_lp[lp]:
                test_src_by_lp[lp].add(ex["src"])
                raw_test.append(ex)

    # Second pass: collect lean validation rows
    for ex in ds:
        lp = ex["lp"]
        if lp in val_src_by_lp and len(val_src_by_lp[lp]) < val_per_lp:
            if ex["src"] not in test_src_by_lp[lp] and ex["src"] not in val_src_by_lp[lp]:
                val_src_by_lp[lp].add(ex["src"])
                raw_val.append(ex)

    # Third pass: collect up to max_train rows (strictly excluding test and val sources)
    for ex in ds:
        lp = ex["lp"]
        if lp in test_src_by_lp:
            if ex["src"] not in test_src_by_lp[lp] and ex["src"] not in val_src_by_lp[lp]:
                raw_train.append(ex)
                if len(raw_train) >= max_train:
                    break

    rng = random.Random(seed)
    rng.shuffle(raw_train)
    rng.shuffle(raw_val)
    rng.shuffle(raw_test)

    print(f"Splits generated (Zero Leakage):")
    print(f"  Train: {len(raw_train):,} rows (Target ~100K)")
    print(f"  Val:   {len(raw_val):,} rows (Lean)")
    print(f"  Test:  {len(raw_test):,} rows (Comprehensive)")

    train_data = [format_translation_prompt(tokenizer, ex["src"], ex["lp"], ex["ref"]) for ex in raw_train]
    val_data   = [format_translation_prompt(tokenizer, ex["src"], ex["lp"], ex["ref"]) for ex in raw_val]
    test_data  = [format_translation_prompt(tokenizer, ex["src"], ex["lp"], ex["ref"]) for ex in raw_test]

    return train_data, val_data, test_data


if __name__ == "__main__":
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("tencent/Hy-MT2-1.8B")
    train, val, test = load_wmt22_24_splits(tok, val_per_lp=50, test_per_lp=50, max_train_per_lp=200)
    print("Sample prompt text:\n", train[0]["prompt_text"][:120], "...")
    print("Sample ref text:\n", train[0]["ref_text"][:60], "...")
