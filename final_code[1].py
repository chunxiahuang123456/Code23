import os
import re
import json
import math
import random
import argparse
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from transformers import AutoTokenizer, TFAutoModel
from datasets import load_dataset
from tqdm import tqdm

DATA_ROOT = r"Path to the Dataset"


DATASET_NAME = "sst5"

LABEL_NAMES: Dict[str, List[str]] = {
    "imdb": ["negative", "positive"],
    "sentiment140": ["negative", "positive"],
    "covidsenti": ["negative", "neutral", "positive"],
    "sst5": ["very_neg", "neg", "neu", "pos", "very_pos"],
    "folder": ["negative", "positive"],
}

CLASS_NAMES: List[str] = LABEL_NAMES[DATASET_NAME]
NUM_CLASSES: int = len(CLASS_NAMES)


def get_class_names(dataset_name: str) -> List[str]:
    if dataset_name in LABEL_NAMES:
        return LABEL_NAMES[dataset_name]
    return [str(i) for i in range(NUM_CLASSES)]


def get_num_classes(dataset_name: str) -> int:
    return len(get_class_names(dataset_name))


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def setup_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"Warning: could not set GPU memory growth: {e}")

        gpu_names = [g.name for g in gpus]
        print(f"GPU(s) detected — training will run on: {gpu_names}")
    else:
        print("No GPU detected — training will run on CPU.")
    return gpus

@dataclass
class Config:
    dataset: str = DATASET_NAME
    model_name: str = "bert-base-uncased"
    max_length: int = 128
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    dropout: float = 0.20
    patience: int = 7
    bert_trainable: bool = True
    warmup_ratio: float = 0.10      
    min_lr_ratio: float = 0.05       
    grad_clip_norm: float = 1.0     
    adam_epsilon: float = 1e-6
    use_class_weights: bool = True   
    gs_blocks: int = 4
    gs_hidden: int = 768
    gs_kernel: int = 3
    gs_dilations: Tuple[int, ...] = (1, 2, 4, 8)
    opinion_dim: int = 300
    opinion_heads: int = 8
    fusion_dim: int = 768
    td_dilations: Tuple[int, ...] = (1, 2, 4, 8)
    td_kernel: int = 3
    pruning_threshold: float = 0.50
    parent_child_coeff: float = 0.50
    local_window: int = 5
    memory_slots: int = 16
    dlta_heads: int = 8
    agte_heads: int = 8
    ffn_dim: int = 3072
    confidence_hidden: int = 256
    confidence_loss_weight: float = 0.30
    seed: int = 42
    output_dir: str = "./runs/hfein_ptdtemformer1"
    subsample_fraction: float = 0.25
    train_split: float = 0.80
    val_split: float = 0.10
    test_split: float = 0.10

URL_RE = re.compile(r"https?://\S+|www\.\S+")
HTML_RE = re.compile(r"<[^>]+>")
MENTION_RE = re.compile(r"@\w+")
SPACE_RE = re.compile(r"\s+")

def basic_clean(text: str) -> str:
    text = str(text).lower()
    text = URL_RE.sub(" ", text)
    text = HTML_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = text.replace("#", "")   
    text = SPACE_RE.sub(" ", text).strip()
    return text

POSITIVE_WORDS = {
    "good","great","excellent","amazing","awesome","love","loved","like","liked",
    "best","wonderful","fantastic","positive","happy","enjoy","enjoyed","perfect",
    "brilliant","beautiful","nice","recommend","recommended","success","successful"
}
NEGATIVE_WORDS = {
    "bad","worst","terrible","awful","hate","hated","poor","boring","negative",
    "sad","disappointing","disappointed","horrible","ugly","fail","failed",
    "failure","waste","annoying","problem","problems","wrong"
}


def opinion_scores_from_tokens(tokens: List[str]) -> np.ndarray:
    scores = np.zeros(len(tokens), dtype=np.float32)
    for i, t in enumerate(tokens):
        w = t.replace("##", "").lower()
        if w in POSITIVE_WORDS:
            scores[i] = 1.0
        elif w in NEGATIVE_WORDS:
            scores[i] = -1.0
    return scores



def _split_train_val(ds, val_fraction=0.1, seed=42):
    split = ds.train_test_split(test_size=val_fraction, seed=seed, stratify_by_column="label")
    return split["train"], split["test"]


def load_imdb(seed=42):
    ds = load_dataset("imdb")
    train, val = _split_train_val(ds["train"], 5000/25000, seed)
    test = ds["test"]
    return (
        [(x["text"], int(x["label"])) for x in train],
        [(x["text"], int(x["label"])) for x in val],
        [(x["text"], int(x["label"])) for x in test],
    )


def load_sst5(seed=42):
    ds = load_dataset("SetFit/sst5")
    return (
        [(x["text"], int(x["label"])) for x in ds["train"]],
        [(x["text"], int(x["label"])) for x in ds["validation"]],
        [(x["text"], int(x["label"])) for x in ds["test"]],
    )


def load_sentiment140(csv_path: str, seed=42):
   
    if not csv_path:
        raise ValueError("--sentiment140_csv is required for dataset=sentiment140")
    cols = ["target","id","date","flag","user","text"]
    df = pd.read_csv(csv_path, encoding="latin-1", header=None, names=cols)
    df = df[["text","target"]].dropna().drop_duplicates("text")
    df["label"] = df["target"].map({0: 0, 4: 1})
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    from sklearn.model_selection import train_test_split
    train_df, tmp_df = train_test_split(
        df, test_size=0.20, random_state=seed, stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        tmp_df, test_size=0.50, random_state=seed, stratify=tmp_df["label"]
    )
    conv = lambda d: list(zip(d["text"].astype(str).tolist(), d["label"].astype(int).tolist()))
    return conv(train_df), conv(val_df), conv(test_df)


def load_covidsenti(csv_path: str, seed=42):
   
    if not csv_path:
        raise ValueError("--covidsenti_csv is required for dataset=covidsenti")
    df = pd.read_csv(csv_path)

    text_col = next((c for c in ["text","tweet","sentence","content"] if c in df.columns), None)
    label_col = next((c for c in ["label","sentiment","polarity"] if c in df.columns), None)
    if text_col is None or label_col is None:
        raise ValueError(
            f"COVIDSenti CSV must contain a text column and label column. Columns found: {list(df.columns)}"
        )

    df = df[[text_col, label_col]].dropna().drop_duplicates(text_col)

    if df[label_col].dtype == object:
        class_names = get_class_names("covidsenti")
        mapping = {}
        for i, name in enumerate(class_names):
            mapping[name.lower()] = i
            mapping[name.lower()[:3]] = i
        labels = df[label_col].astype(str).str.lower().str.strip().map(mapping)
    else:
        labels = df[label_col].astype(int)
        unique = sorted(labels.unique().tolist())
        expected = list(range(get_num_classes("covidsenti")))
        if unique != expected:
            remap = {v: i for i, v in enumerate(unique)}
            labels = labels.map(remap)

    df = pd.DataFrame({"text": df[text_col].astype(str), "label": labels}).dropna()
    df["label"] = df["label"].astype(int)

    from sklearn.model_selection import train_test_split
    train_df, tmp_df = train_test_split(
        df, test_size=0.20, random_state=seed, stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        tmp_df, test_size=0.50, random_state=seed, stratify=tmp_df["label"]
    )
    conv = lambda d: list(zip(d["text"].tolist(), d["label"].tolist()))
    return conv(train_df), conv(val_df), conv(test_df)


def load_folder_dataset(root_path: str, dataset_name: str, seed=42):
    
    if not root_path or root_path == "/path/to/dataset":
        raise ValueError(
            "Set the DATA_ROOT variable near the top of this file to your "
            f"'{dataset_name}' dataset's root folder before running."
        )

    root = Path(root_path) 
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root_path}")

    
    class_names = get_class_names(dataset_name)
    label_name_variants: Dict[int, List[str]] = {}
    for idx, name in enumerate(class_names):
        variants = [name.lower(), name.lower().replace("_", ""), name.lower()[:3]]
        seen = set()
        label_name_variants[idx] = [v for v in variants if not (v in seen or seen.add(v))]

    samples: List[Tuple[str, int]] = []

    for split_name in ["train", "test"]:
        split_dir = root / split_name
        if not split_dir.exists():
            print(f"Warning: expected split folder not found, skipping: {split_dir}")
            continue

        existing = {p.name.lower(): p for p in split_dir.iterdir() if p.is_dir()}

        for label, name_variants in label_name_variants.items():
            class_dir = None
            for name in name_variants:
                if name in existing:
                    class_dir = existing[name]
                    break

            if class_dir is None:
                variants_str = "/".join(name_variants)
                print(f"Warning: expected class folder not found, skipping: {split_dir / variants_str}")
                continue

            files = sorted(class_dir.glob("*.txt"))
            for fp in tqdm(files, desc=f"Reading {split_name}/{class_dir.name}", unit="file"):
                try:
                    text = fp.read_text(encoding="utf-8", errors="ignore").strip()
                except Exception as e:
                    print(f"Skipping unreadable file {fp}: {e}")
                    continue
                if not text:
                    continue
                samples.append((text, label))

    if not samples:
        expected = ", ".join(
            f"{name}(label {idx})" for idx, name in enumerate(class_names)
        )
        raise ValueError(
            f"No .txt samples found under {root_path}. Expected structure: "
            f"train/<class_folder>/*.txt and test/<class_folder>/*.txt, "
            f"with one class folder per label: {expected}"
        )

    df = pd.DataFrame(samples, columns=["text", "label"])
    df["label"] = df["label"].astype(int)

    from sklearn.model_selection import train_test_split
    train_df, tmp_df = train_test_split(
        df, test_size=0.20, random_state=seed, stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        tmp_df, test_size=0.50, random_state=seed, stratify=tmp_df["label"]
    )

    conv = lambda d: list(zip(d["text"].astype(str).tolist(), d["label"].astype(int).tolist()))
    return conv(train_df), conv(val_df), conv(test_df)


def _print_split_stats(name: str, examples: List[Tuple[str, int]], class_names: List[str]):
    labels = [y for _, y in examples]
    counter = Counter(labels)
    dist_str = ", ".join(
        f"{class_names[c] if 0 <= c < len(class_names) else c}={counter.get(c, 0)}"
        for c in sorted(counter.keys())
    )
    print(f"{name}: n={len(examples)} | class distribution -> {dist_str}")


def stratified_subsample(all_examples: List[Tuple[str, int]], cfg: Config):
  
    seed = cfg.seed
    labels = np.array([y for _, y in all_examples])
    total_n = len(all_examples)
    class_names = get_class_names(cfg.dataset)

    print("\n===== Stratified Subsampling =====")
    print(f"Seed: {seed}")
    _print_split_stats("Total (all original splits combined)", all_examples, class_names)

    from sklearn.model_selection import train_test_split

    idx_all = np.arange(total_n)
    selected_idx, _discarded_idx = train_test_split(
        idx_all,
        train_size=cfg.subsample_fraction,
        random_state=seed,
        stratify=labels,
    )
    selected_idx = np.asarray(selected_idx)
    selected_examples = [all_examples[i] for i in selected_idx]

    print(f"\nSelected subset size ({cfg.subsample_fraction * 100:.0f}% stratified sample of total): "
          f"{len(selected_examples)}")
    _print_split_stats("Selected subset", selected_examples, class_names)
    print("===================================\n")

    return selected_examples


def resplit_train_val_test(selected_examples: List[Tuple[str, int]], cfg: Config):
   
    ratios_sum = cfg.train_split + cfg.val_split + cfg.test_split
    if abs(ratios_sum - 1.0) > 1e-6:
        raise ValueError(
            f"train_split + val_split + test_split must sum to 1.0, got {ratios_sum}"
        )

    seed = cfg.seed
    selected_labels = np.array([y for _, y in selected_examples])
    class_names = get_class_names(cfg.dataset)

    print("\n===== Re-splitting Selected Subset into Train/Val/Test =====")

    from sklearn.model_selection import train_test_split

    train_idx, temp_idx = train_test_split(
        np.arange(len(selected_examples)),
        train_size=cfg.train_split,
        random_state=seed,
        stratify=selected_labels,
    )
    temp_labels = selected_labels[temp_idx]
    val_fraction_of_temp = cfg.val_split / (cfg.val_split + cfg.test_split)
    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=val_fraction_of_temp,
        random_state=seed,
        stratify=temp_labels,
    )

    train_set, val_set, test_set = set(train_idx.tolist()), set(val_idx.tolist()), set(test_idx.tolist())
    assert train_set.isdisjoint(val_set), "Data leakage detected: train/val overlap!"
    assert train_set.isdisjoint(test_set), "Data leakage detected: train/test overlap!"
    assert val_set.isdisjoint(test_set), "Data leakage detected: val/test overlap!"
    assert len(train_set) + len(val_set) + len(test_set) == len(selected_examples), \
        "Data leakage detected: split sizes do not add up to the selected subset size!"

    train_ex = [selected_examples[i] for i in train_idx]
    val_ex = [selected_examples[i] for i in val_idx]
    test_ex = [selected_examples[i] for i in test_idx]

    _print_split_stats(f"Train ({cfg.train_split * 100:.0f}% of selected subset)", train_ex, class_names)
    _print_split_stats(f"Validation ({cfg.val_split * 100:.0f}% of selected subset)", val_ex, class_names)
    _print_split_stats(f"Test ({cfg.test_split * 100:.0f}% of selected subset)", test_ex, class_names)
    print("==============================================================\n")

    return train_ex, val_ex, test_ex


def _subset_cache_path(cfg: Config) -> Path:
    pct = int(round(cfg.subsample_fraction * 100))
    cache_dir = Path(cfg.output_dir) / cfg.dataset
    return cache_dir / f"selected_subset_seed{cfg.seed}_{pct}pct.csv"


def load_or_build_stratified_subset(cfg: Config, args):
   
    cache_path = _subset_cache_path(cfg)

    if cache_path.exists():
        print(f"\nFound cached {cfg.subsample_fraction * 100:.0f}% stratified subset at: {cache_path}")
        print("Reading ONLY this cached subset — the full original dataset will NOT be read.")
        df = pd.read_csv(cache_path)
        selected_examples = list(zip(df["text"].astype(str).tolist(), df["label"].astype(int).tolist()))
        class_names = get_class_names(cfg.dataset)
        _print_split_stats("Loaded cached selected subset", selected_examples, class_names)
        return selected_examples

    print(f"\nNo cached subset found for dataset='{cfg.dataset}', seed={cfg.seed}, "
          f"fraction={cfg.subsample_fraction}. Reading the full original dataset once "
          f"to build it (this only happens the first time).")

    
    if cfg.dataset in LABEL_NAMES:
        train_ex, val_ex, test_ex = load_folder_dataset(DATA_ROOT, cfg.dataset, cfg.seed)
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")

    all_examples = train_ex + val_ex + test_ex
    selected_examples = stratified_subsample(all_examples, cfg)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    sel_texts = [t for t, _ in selected_examples]
    sel_labels = [y for _, y in selected_examples]
    pd.DataFrame({"text": sel_texts, "label": sel_labels}).to_csv(cache_path, index=False)
    print(f"Saved the selected {cfg.subsample_fraction * 100:.0f}% stratified subset to: {cache_path}")
    print("Future runs with the same dataset/seed/fraction will read ONLY this file.\n")

    return selected_examples


def load_benchmark_data(cfg: Config, args):
 
    selected_examples = load_or_build_stratified_subset(cfg, args)
    return resplit_train_val_test(selected_examples, cfg)

def encode_examples(examples, tokenizer, cfg: Config):
    texts = [basic_clean(t) for t, _ in examples]
    labels = np.asarray([y for _, y in examples], dtype=np.int32)

    enc = tokenizer(
        texts,
        max_length=cfg.max_length,
        truncation=True,
        padding="max_length",
        return_attention_mask=True,
        return_token_type_ids=True,
    )

    input_ids = np.asarray(enc["input_ids"], dtype=np.int32)
    attn = np.asarray(enc["attention_mask"], dtype=np.int32)
    token_types = np.asarray(enc.get("token_type_ids", np.zeros_like(input_ids)), dtype=np.int32)

    opinion = np.zeros((len(texts), cfg.max_length, 1), dtype=np.float32)
    aspect_mask = np.zeros((len(texts), cfg.max_length), dtype=np.float32)

    for i, ids in enumerate(input_ids):
        toks = tokenizer.convert_ids_to_tokens(ids.tolist())
        scores = opinion_scores_from_tokens(toks)
        scores *= attn[i]
        opinion[i, :, 0] = scores

        idx = np.where(np.abs(scores) > 0)[0]
        if len(idx) == 0:
            aspect_mask[i, 0] = 1.0
        else:
            aspect_mask[i, idx] = 1.0

    x = {
        "input_ids": input_ids,
        "attention_mask": attn,
        "token_type_ids": token_types,
        "opinion_scores": opinion,
        "aspect_mask": aspect_mask,
    }
    return x, labels


def make_tf_dataset(x, y, batch_size, training=False, seed=42):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        ds = ds.shuffle(min(len(y), 20000), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

class GSBlock(layers.Layer):
    def __init__(self, hidden, kernel_size, dilations, dropout, **kwargs):
        super().__init__(**kwargs)
        self.branches = [
            layers.Conv1D(hidden // len(dilations), kernel_size, padding="same",
                          dilation_rate=d, activation=tf.nn.gelu)
            for d in dilations
        ]
        self.proj = layers.Conv1D(hidden, 1, padding="same")
        self.gate = layers.Dense(hidden, activation="sigmoid")
        self.dropout = layers.Dropout(dropout)
        self.norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, training=False):
        fs = [b(x) for b in self.branches]
        f = tf.concat(fs, axis=-1)
        f = self.proj(f)
        g = self.gate(f)
        f = g * f
        f = self.dropout(f, training=training)
        return self.norm(x + f)


class GSNet(layers.Layer):
    def __init__(self, hidden, n_blocks, kernel_size, dilations, dropout, **kwargs):
        super().__init__(**kwargs)
        self.in_proj = layers.Dense(hidden)
        self.blocks = [
            GSBlock(hidden, kernel_size, dilations, dropout, name=f"gs_block_{i}")
            for i in range(n_blocks)
        ]
        self.level_logits = self.add_weight(
            name="level_logits", shape=(n_blocks,), initializer="zeros", trainable=True
        )
        self.norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, training=False):
        h0 = self.in_proj(x)
        h = h0
        levels = []
        for block in self.blocks:
            h = block(h, training=training)
            levels.append(h)
        alpha = tf.nn.softmax(self.level_logits)
        h_ms = tf.add_n([alpha[i] * levels[i] for i in range(len(levels))])
        return self.norm(h0 + h_ms)


class OpinionEmbedding(layers.Layer):
    def __init__(self, hidden, opinion_dim, heads, dropout, **kwargs):
        super().__init__(**kwargs)
        self.score_proj = layers.Dense(opinion_dim, activation="tanh")
        self.to_hidden = layers.Dense(hidden)
        key_dim = max(1, hidden // heads)
        self.attn = layers.MultiHeadAttention(num_heads=heads, key_dim=key_dim, dropout=dropout)
        self.norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, bert_h, opinion_scores, attention_mask=None, training=False):
        o = self.score_proj(opinion_scores)
        o = self.to_hidden(o)
        mask = None
        if attention_mask is not None:
            mask = attention_mask[:, tf.newaxis, :]
        h = self.attn(query=o, key=bert_h, value=bert_h,
                      attention_mask=mask, training=training)
        return self.norm(o + h)


class DFIEN(layers.Layer):
    def __init__(self, fusion_dim, **kwargs):
        super().__init__(**kwargs)
        self.gs_proj = layers.Dense(fusion_dim)
        self.ctx_proj = layers.Dense(fusion_dim)
        self.gate = layers.Dense(fusion_dim, activation="sigmoid")
        self.norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, gs, combined):
        gs = self.gs_proj(gs)
        ctx = self.ctx_proj(combined)
        gs_conf = tf.nn.sigmoid(gs)
        g = self.gate(tf.concat([gs_conf, ctx], axis=-1))
        fused = g * gs_conf + (1.0 - g) * ctx
        return self.norm(fused)


class MultiScaleTDTCN(layers.Layer):
    def __init__(self, hidden, kernel_size, dilations, dropout, **kwargs):
        super().__init__(**kwargs)
        per = hidden // len(dilations)
        self.branches = [
            keras.Sequential([
                layers.Conv1D(per, kernel_size, padding="causal", dilation_rate=d,
                              activation=tf.nn.gelu),
                layers.Dropout(dropout),
                layers.Conv1D(per, kernel_size, padding="causal", dilation_rate=d,
                              activation=tf.nn.gelu),
            ], name=f"td_branch_d{d}")
            for d in dilations
        ]
        self.proj = layers.Dense(hidden)
        self.norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, training=False):
        fs = [b(x, training=training) for b in self.branches]
        h = self.proj(tf.concat(fs, axis=-1))
        return self.norm(x + h)


class ASTP(layers.Layer):
    def __init__(self, threshold=0.5, parent_coeff=0.5, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.parent_coeff = parent_coeff
        self.score = layers.Dense(1, activation="sigmoid")

    def call(self, h, attention_mask=None, training=False):
        s = self.score(h)  
        parent = tf.concat([s[:, :1, :], s[:, :-1, :]], axis=1)
        refined = self.parent_coeff * s + (1.0 - self.parent_coeff) * parent

        if training:
            mask = tf.nn.sigmoid((refined - self.threshold) * 12.0)
        else:
            mask = tf.cast(refined >= self.threshold, h.dtype)

        if attention_mask is not None:
            mask *= tf.cast(attention_mask[:, :, tf.newaxis], h.dtype)

        return h * mask, tf.squeeze(refined, axis=-1), tf.squeeze(mask, axis=-1)


def build_local_attention_mask(seq_len, window):
    idx = tf.range(seq_len)
    dist = tf.abs(idx[:, None] - idx[None, :])
    return tf.cast(dist <= (window // 2), tf.float32)


class DLTA(layers.Layer):
    def __init__(self, hidden, heads, local_window, memory_slots, dropout, **kwargs):
        super().__init__(**kwargs)
        self.hidden = hidden
        self.local_window = local_window
        self.local_attn = layers.MultiHeadAttention(
            num_heads=heads, key_dim=max(1, hidden // heads), dropout=dropout
        )
        self.memory = self.add_weight(
            name="semantic_memory",
            shape=(memory_slots, hidden),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.global_q = layers.Dense(hidden)
        self.gate = layers.Dense(hidden, activation="sigmoid")
        self.norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, h, attention_mask=None, training=False):
        n = tf.shape(h)[1]
        local = build_local_attention_mask(n, self.local_window)
        local = local[tf.newaxis, :, :]
        if attention_mask is not None:
            pad = tf.cast(attention_mask[:, tf.newaxis, :], tf.float32)
            local = local * pad

        f_local = self.local_attn(h, h, attention_mask=local, training=training)

        q = self.global_q(h)
        logits = tf.matmul(q, self.memory, transpose_b=True) / tf.sqrt(tf.cast(self.hidden, tf.float32))
        a = tf.nn.softmax(logits, axis=-1)
        f_global = tf.matmul(a, self.memory)

        alpha = self.gate(tf.concat([f_local, f_global], axis=-1))
        out = alpha * f_local + (1.0 - alpha) * f_global
        return self.norm(h + out)


class AGTE(layers.Layer):
    def __init__(self, hidden, heads, ffn_dim, dropout, **kwargs):
        super().__init__(**kwargs)
        key_dim = max(1, hidden // heads)
        self.cross = layers.MultiHeadAttention(num_heads=heads, key_dim=key_dim, dropout=dropout)
        self.self_attn = layers.MultiHeadAttention(num_heads=heads, key_dim=key_dim, dropout=dropout)
        self.ffn = keras.Sequential([
            layers.Dense(ffn_dim, activation=tf.nn.gelu),
            layers.Dropout(dropout),
            layers.Dense(hidden),
        ])
        self.n1 = layers.LayerNormalization(epsilon=1e-6)
        self.n2 = layers.LayerNormalization(epsilon=1e-6)
        self.n3 = layers.LayerNormalization(epsilon=1e-6)
        self.drop = layers.Dropout(dropout)

    def call(self, h, aspect_mask, attention_mask=None, training=False):
        am = tf.cast(aspect_mask[:, :, tf.newaxis], h.dtype)
        denom = tf.maximum(tf.reduce_sum(am, axis=1, keepdims=True), 1.0)
        aspect = tf.reduce_sum(h * am, axis=1, keepdims=True) / denom

        aspect_q = tf.repeat(aspect, repeats=tf.shape(h)[1], axis=1)

        key_mask = None
        if attention_mask is not None:
            key_mask = attention_mask[:, tf.newaxis, :]

        cross = self.cross(
            query=aspect_q, key=h, value=h,
            attention_mask=key_mask, training=training
        )
        x = self.n1(h + self.drop(cross, training=training))

        sa = self.self_attn(
            query=x, key=x, value=x,
            attention_mask=key_mask, training=training
        )
        x = self.n2(x + self.drop(sa, training=training))
        ff = self.ffn(x, training=training)
        return self.n3(x + ff)


def load_tf_backbone(model_name: str):
    try:
        return TFAutoModel.from_pretrained(model_name)
    except TypeError as e:
        if "safe_open" not in str(e):
            raise
        print(
            f"Warning: TFAutoModel.from_pretrained('{model_name}') hit a "
            "safetensors loading bug ('safe_open' object is not iterable). "
            "Retrying with use_safetensors=False (falls back to/converts from "
            "PyTorch weights). For a permanent fix, also consider upgrading "
            "the 'transformers' and 'safetensors' packages to matching, "
            "up-to-date versions."
        )
        try:
            return TFAutoModel.from_pretrained(model_name, use_safetensors=False)
        except (OSError, EnvironmentError):
            return TFAutoModel.from_pretrained(model_name, from_pt=True)


class HFEINPTDTemFormer(keras.Model):
    def __init__(self, cfg: Config, num_classes: int, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.num_classes = num_classes

        self.bert = load_tf_backbone(cfg.model_name)
        self.bert.trainable = cfg.bert_trainable

        self.gs = GSNet(
            cfg.gs_hidden, cfg.gs_blocks, cfg.gs_kernel,
            cfg.gs_dilations, cfg.dropout, name="gs_net"
        )
        self.opinion = OpinionEmbedding(
            cfg.fusion_dim, cfg.opinion_dim, cfg.opinion_heads,
            cfg.dropout, name="opinion_embedding"
        )
        self.bert_proj = layers.Dense(cfg.fusion_dim)
        self.dfien = DFIEN(cfg.fusion_dim, name="dfien")

        self.ms_td = MultiScaleTDTCN(
            cfg.fusion_dim, cfg.td_kernel, cfg.td_dilations,
            cfg.dropout, name="ms_td_tcn"
        )
        self.astp = ASTP(cfg.pruning_threshold, cfg.parent_child_coeff, name="astp")
        self.dlta = DLTA(
            cfg.fusion_dim, cfg.dlta_heads, cfg.local_window,
            cfg.memory_slots, cfg.dropout, name="dlta"
        )
        self.agte = AGTE(
            cfg.fusion_dim, cfg.agte_heads, cfg.ffn_dim,
            cfg.dropout, name="agte"
        )

        self.pool = layers.GlobalAveragePooling1D()
        self.classifier = layers.Dense(num_classes, name="sentiment_logits")
        self.confidence_mlp = keras.Sequential([
            layers.Dense(cfg.confidence_hidden, activation=tf.nn.gelu),
            layers.Dropout(cfg.dropout),
            layers.Dense(1, activation="sigmoid"),
        ], name="confidence_head")

        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.cls_loss_tracker = keras.metrics.Mean(name="classification_loss")
        self.conf_loss_tracker = keras.metrics.Mean(name="confidence_loss")
        self.acc = keras.metrics.SparseCategoricalAccuracy(name="accuracy")

      
        self.class_weights = None

    def set_class_weights(self, class_weights):
        if class_weights is None:
            self.class_weights = None
        else:
            self.class_weights = tf.constant(class_weights, dtype=tf.float32)

    @property
    def metrics(self):
        return [self.loss_tracker, self.cls_loss_tracker, self.conf_loss_tracker, self.acc]

    def call(self, inputs, training=False, return_aux=False):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]
        opinion_scores = inputs["opinion_scores"]
        aspect_mask = inputs["aspect_mask"]

        bert_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            training=training,
        ).last_hidden_state

        bert_h = self.bert_proj(bert_out)

        gs_h = self.gs(bert_h, training=training)

        opinion_h = self.opinion(
            bert_h, opinion_scores, attention_mask=attention_mask, training=training
        )
        combined = bert_h + opinion_h

        fused = self.dfien(gs_h, combined)
        temporal = self.ms_td(fused, training=training)
        pruned, importance, prune_mask = self.astp(
            temporal, attention_mask=attention_mask, training=training
        )
        temporal2 = self.dlta(pruned, attention_mask=attention_mask, training=training)
        refined = self.agte(
            temporal2, aspect_mask=aspect_mask,
            attention_mask=attention_mask, training=training
        )

        m = tf.cast(attention_mask[:, :, None], refined.dtype)
        pooled = tf.reduce_sum(refined * m, axis=1) / tf.maximum(tf.reduce_sum(m, axis=1), 1.0)

        logits = self.classifier(pooled)
        confidence = self.confidence_mlp(pooled, training=training)

        if return_aux:
            return {
                "logits": logits,
                "confidence": confidence,
                "importance": importance,
                "prune_mask": prune_mask,
            }
        return logits, confidence

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            logits, confidence = self(x, training=True)
            cls_loss_per_example = tf.keras.losses.sparse_categorical_crossentropy(
                y, logits, from_logits=True
            )
            if self.class_weights is not None:
                sample_weights = tf.gather(self.class_weights, tf.cast(y, tf.int32))
                cls_loss = tf.reduce_sum(cls_loss_per_example * sample_weights) / (
                    tf.reduce_sum(sample_weights) + 1e-8
                )
            else:
                cls_loss = tf.reduce_mean(cls_loss_per_example)

            pred = tf.argmax(logits, axis=-1, output_type=tf.int32)
            conf_target = tf.cast(tf.equal(pred, tf.cast(y, tf.int32)), tf.float32)
            conf_target = tf.stop_gradient(conf_target[:, None])

            conf_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(conf_target, confidence)
            )
            total = cls_loss + self.cfg.confidence_loss_weight * conf_loss

            if self.losses:
                total += tf.add_n(self.losses)

        grads = tape.gradient(total, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.loss_tracker.update_state(total)
        self.cls_loss_tracker.update_state(cls_loss)
        self.conf_loss_tracker.update_state(conf_loss)
        self.acc.update_state(y, logits)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        logits, confidence = self(x, training=False)
        cls_loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True)
        )
        pred = tf.argmax(logits, axis=-1, output_type=tf.int32)
        conf_target = tf.cast(tf.equal(pred, tf.cast(y, tf.int32)), tf.float32)[:, None]
        conf_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(conf_target, confidence)
        )
        total = cls_loss + self.cfg.confidence_loss_weight * conf_loss

        self.loss_tracker.update_state(total)
        self.cls_loss_tracker.update_state(cls_loss)
        self.conf_loss_tracker.update_state(conf_loss)
        self.acc.update_state(y, logits)
        return {m.name: m.result() for m in self.metrics}


def compute_macro_specificity(cm) -> float:
   
    cm = np.asarray(cm, dtype=np.float64)
    total = cm.sum()
    specificities = []
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = total - tp - fn - fp
        denom = tn + fp
        specificities.append(tn / denom if denom > 0 else 0.0)
    return float(np.mean(specificities)) if specificities else 0.0


def classification_report_metrics(y_true, y_pred, num_classes):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef, confusion_matrix

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    mcc = matthews_corrcoef(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    specificity = compute_macro_specificity(cm)
    return {
        "accuracy": float(acc),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "specificity_macro": float(specificity),
        "mcc": float(mcc),
        "confusion_matrix": cm.tolist(),
    }


def predict_dataset(model, ds):
    all_y, all_pred, all_conf, all_prob = [], [], [], []
    for x, y in ds:
        out = model(x, training=False, return_aux=True)
        probs = tf.nn.softmax(out["logits"], axis=-1).numpy()
        pred = probs.argmax(axis=-1)
        conf = out["confidence"].numpy().reshape(-1)
        all_y.extend(y.numpy().tolist())
        all_pred.extend(pred.tolist())
        all_conf.extend(conf.tolist())
        all_prob.extend(probs.tolist())
    return (
        np.asarray(all_y),
        np.asarray(all_pred),
        np.asarray(all_conf),
        np.asarray(all_prob),
    )


def save_detailed_test_predictions(outdir, cfg, num_classes, y_true, y_pred, probs,
                                    filename="test_predictions_detailed.csv"):
   
    class_names = get_class_names(cfg.dataset)

    def _name_for(label_idx):
        return class_names[label_idx] if 0 <= label_idx < len(class_names) else str(label_idx)

    n = len(y_true)
    data = {
        "Sample_ID": np.arange(n),
        "True_Label": y_true,
        "True_Class_Name": [_name_for(int(i)) for i in y_true],
    }
    for c in range(num_classes):
        data[f"Predicted_Probability_Class_{c}"] = probs[:, c]
    data["Predicted_Class"] = y_pred
    data["Class_Name"] = [_name_for(int(i)) for i in y_pred]

    detailed_df = pd.DataFrame(data)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / filename
    detailed_df.to_csv(out_path, index=False)
    print(f"Saved detailed per-sample test predictions to: {out_path.resolve()}")
    return out_path


class EpochMetricsCallback(keras.callbacks.Callback):
  
    def __init__(self, eval_ds, num_classes, split_name="validation"):
        super().__init__()
        self.eval_ds = eval_ds
        self.num_classes = num_classes
        self.split_name = split_name

    def on_epoch_end(self, epoch, logs=None):
        y_true, y_pred, _, _ = predict_dataset(self.model, self.eval_ds)
        metrics = classification_report_metrics(y_true, y_pred, self.num_classes)
        print(
            f"\nEpoch {epoch + 1} {self.split_name} metrics -> "
            f"accuracy: {metrics['accuracy']:.4f} | "
            f"precision (macro): {metrics['precision_macro']:.4f} | "
            f"recall (macro): {metrics['recall_macro']:.4f} | "
            f"f1 (macro): {metrics['f1_macro']:.4f} | "
            f"mcc: {metrics['mcc']:.4f}"
        )


def save_epoch_metrics_csv(epoch_metrics_dir, epoch, metrics, val_loss, filename=None):
   
    epoch_metrics_dir = Path(epoch_metrics_dir)
    epoch_metrics_dir.mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = f"epoch_{epoch:03d}_metrics.csv"

    row = {
        "epoch": epoch,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision_macro"],
        "recall": metrics["recall_macro"],
        "f1_score": metrics["f1_macro"],
        "specificity": metrics["specificity_macro"],
        "MCC": metrics["mcc"],
        "val_loss": val_loss,
    }
    out_path = epoch_metrics_dir / filename
    pd.DataFrame([row]).to_csv(out_path, index=False)
    print(f"Saved epoch {epoch} test-set metrics to: {out_path.resolve()}")
    return out_path


class EpochTestPredictionsCallback(keras.callbacks.Callback):
 

    def __init__(self, test_ds, cfg: Config, num_classes: int, outdir):
        super().__init__()
        self.test_ds = test_ds
        self.cfg = cfg
        self.num_classes = num_classes
        self.epoch_dir = Path(outdir) / "epoch_test_predictions"
        self.epoch_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir = Path(outdir) / "epoch_metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        y_true, y_pred, _conf, probs = predict_dataset(self.model, self.test_ds)

        pred_filename = f"epoch_{epoch + 1:03d}_test_predictions_detailed.csv"
        save_detailed_test_predictions(
            self.epoch_dir, self.cfg, self.num_classes, y_true, y_pred, probs, filename=pred_filename
        )

        metrics = classification_report_metrics(y_true, y_pred, self.num_classes)
        logs = logs or {}
        val_loss = logs.get("val_loss")
        metrics_filename = f"epoch_{epoch + 1:03d}_metrics.csv"
        save_epoch_metrics_csv(self.metrics_dir, epoch + 1, metrics, val_loss, filename=metrics_filename)


class WarmupLinearDecaySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
   

    def __init__(self, peak_lr, total_steps, warmup_steps, min_lr_ratio=0.05):
        super().__init__()
        self.peak_lr = float(peak_lr)
        self.total_steps = max(int(total_steps), 1)
        self.warmup_steps = max(int(warmup_steps), 1)
        self.min_lr_ratio = float(min_lr_ratio)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        total_steps = tf.cast(self.total_steps, tf.float32)

        warmup_lr = self.peak_lr * (step / warmup_steps)

        decay_progress = tf.clip_by_value(
            (step - warmup_steps) / tf.maximum(total_steps - warmup_steps, 1.0), 0.0, 1.0
        )
        floor_lr = self.peak_lr * self.min_lr_ratio
        decayed_lr = self.peak_lr - (self.peak_lr - floor_lr) * decay_progress

        return tf.where(step < warmup_steps, warmup_lr, decayed_lr)

    def get_config(self):
        return {
            "peak_lr": self.peak_lr,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr_ratio": self.min_lr_ratio,
        }


def make_optimizer(cfg: Config, total_steps: int):
    warmup_steps = max(1, int(round(total_steps * cfg.warmup_ratio)))
    lr_schedule = WarmupLinearDecaySchedule(
        peak_lr=cfg.learning_rate,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=cfg.min_lr_ratio,
    )
    try:
        return tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=cfg.weight_decay,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=cfg.adam_epsilon,
            clipnorm=cfg.grad_clip_norm,
        )
    except Exception:
        return tf.keras.optimizers.Adam(
            learning_rate=lr_schedule,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=cfg.adam_epsilon,
            clipnorm=cfg.grad_clip_norm,
        )


def train_and_test(cfg: Config, args):
    setup_gpu()
    set_seed(cfg.seed)

    outdir = Path(cfg.output_dir) / cfg.dataset
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    print(f"\nLoading dataset: {cfg.dataset}")
    train_ex, val_ex, test_ex = load_benchmark_data(cfg, args)
    print(f"Train={len(train_ex):,}  Val={len(val_ex):,}  Test={len(test_ex):,}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    print("Tokenizing...")
    x_train, y_train = encode_examples(train_ex, tokenizer, cfg)
    x_val, y_val = encode_examples(val_ex, tokenizer, cfg)
    x_test, y_test = encode_examples(test_ex, tokenizer, cfg)

    train_ds = make_tf_dataset(x_train, y_train, cfg.batch_size, True, cfg.seed)
    val_ds = make_tf_dataset(x_val, y_val, cfg.batch_size, False, cfg.seed)
    test_ds = make_tf_dataset(x_test, y_test, cfg.batch_size, False, cfg.seed)

    num_classes = get_num_classes(cfg.dataset)
    model = HFEINPTDTemFormer(cfg, num_classes)

    steps_per_epoch = max(1, math.ceil(len(train_ex) / cfg.batch_size))
    total_steps = steps_per_epoch * cfg.epochs
    model.compile(optimizer=make_optimizer(cfg, total_steps))

   
    if cfg.use_class_weights:
        from sklearn.utils.class_weight import compute_class_weight

        class_weight_values = compute_class_weight(
            class_weight="balanced",
            classes=np.arange(num_classes),
            y=np.asarray(y_train),
        )
        model.set_class_weights(class_weight_values)
        print(f"\nUsing balanced class weights for the training loss: "
              f"{dict(zip(range(num_classes), np.round(class_weight_values, 4).tolist()))}")
    else:
        model.set_class_weights(None)

    for xb, _ in train_ds.take(1):
        _ = model(xb, training=False)

    ckpt = outdir / "best.weights.h5"
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(ckpt), monitor="val_loss", save_best_only=True,
            save_weights_only=True, mode="min", verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=cfg.patience,
            restore_best_weights=True, verbose=1
        ),
        keras.callbacks.CSVLogger(str(outdir / "training_log.csv")),
        EpochMetricsCallback(val_ds, num_classes, split_name="validation"),
        EpochTestPredictionsCallback(test_ds, cfg, num_classes, outdir),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    if ckpt.exists():
        model.load_weights(str(ckpt))

    y_true, y_pred, conf, probs = predict_dataset(model, test_ds)
    metrics = classification_report_metrics(y_true, y_pred, num_classes)

    print("\n===== FINAL TEST METRICS =====")
    print(f"accuracy:            {metrics['accuracy']:.6f}")
    print(f"precision (macro):   {metrics['precision_macro']:.6f}")
    print(f"recall (macro):      {metrics['recall_macro']:.6f}")
    print(f"f1 (macro):          {metrics['f1_macro']:.6f}")
    print(f"mcc:                 {metrics['mcc']:.6f}")
    print("confusion_matrix:")
    print(np.asarray(metrics["confusion_matrix"]))

    with open(outdir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    pred_df = pd.DataFrame({
        "gold": y_true,
        "prediction": y_pred,
        "confidence_head": conf,
        "max_softmax_probability": probs.max(axis=1),
    })
    for c in range(num_classes):
        pred_df[f"prob_class_{c}"] = probs[:, c]
    pred_df.to_csv(outdir / "test_predictions.csv", index=False)

  
    save_detailed_test_predictions(outdir, cfg, num_classes, y_true, y_pred, probs)

    print(f"\nSaved outputs to: {outdir.resolve()}")


def test_only(cfg: Config, args):
    setup_gpu()
    set_seed(cfg.seed)
    outdir = Path(cfg.output_dir) / cfg.dataset
    ckpt = Path(args.checkpoint) if args.checkpoint else outdir / "best.weights.h5"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    _, _, test_ex = load_benchmark_data(cfg, args)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    x_test, y_test = encode_examples(test_ex, tokenizer, cfg)
    test_ds = make_tf_dataset(x_test, y_test, cfg.batch_size, False, cfg.seed)

    num_classes = get_num_classes(cfg.dataset)
    model = HFEINPTDTemFormer(cfg, num_classes)
    model.compile(optimizer=make_optimizer(cfg, total_steps=1))
    for xb, _ in test_ds.take(1):
        _ = model(xb, training=False)
    model.load_weights(str(ckpt))

    y_true, y_pred, conf, probs = predict_dataset(model, test_ds)
    metrics = classification_report_metrics(y_true, y_pred, num_classes)
    print(json.dumps(metrics, indent=2))

   
    save_detailed_test_predictions(outdir, cfg, num_classes, y_true, y_pred, probs)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "test"], default="train")
    p.add_argument("--dataset", choices=list(LABEL_NAMES.keys()), default=DATASET_NAME)
    p.add_argument("--sentiment140_csv", type=str, default="")
    p.add_argument("--covidsenti_csv", type=str, default="")
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--output_dir", type=str, default="./runs/hfein_ptdtemformer")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.10)
    p.add_argument("--min_lr_ratio", type=float, default=0.05)
    p.add_argument("--grad_clip_norm", type=float, default=1.0)
    p.add_argument("--no_class_weights", action="store_true",
                    help="Disable balanced class weighting of the training loss.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--freeze_bert", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config(
        dataset=args.dataset,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        min_lr_ratio=args.min_lr_ratio,
        grad_clip_norm=args.grad_clip_norm,
        use_class_weights=not args.no_class_weights,
        seed=args.seed,
        bert_trainable=not args.freeze_bert,
    )
    if args.mode == "train":
        train_and_test(cfg, args)
    else:
        test_only(cfg, args)


if __name__ == "__main__":
    main()
