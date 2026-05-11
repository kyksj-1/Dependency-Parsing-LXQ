#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
文件说明：把 word2vec 文本格式的预训练词向量做 PCA 降维。
应用场景：Chinese-Word-Vectors 官方只发布 300d，"100d vs 300d 维度对比"必须从 300d
        post-hoc 降维同源得到 100d 才公平。这是文献里的标准做法
        （"post-hoc dimensionality reduction"）。

流程：
    1) 读首行 "vocab_size dim" + 词表 + 矩阵 X (V × 300)
    2) 中心化 X' = X - mean
    3) 协方差 C = X'^T X' / V （300×300，小矩阵）
    4) eigh(C) 取前 K 个最大特征值对应的特征向量 V_K (300 × K)
    5) 投影 X_K = X' @ V_K  (V × K)
    6) 输出 word2vec 文本格式："V K\\n" + "word v1 v2 ... vK\\n" * V
注意：
    - 若输入是 .bz2，自动解压
    - 跳过非法行（维度不一致），与 DataUtils/Embed.py 的 _read_file 行为一致
    - 仅依赖 numpy，避免引入 sklearn
用法：
    python tools/pca_embed.py \
        --input  Data/Embed/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5.bz2 \
        --output Data/Embed/sgns.mixed-large.100d.txt \
        --dim    100
"""

import argparse
import bz2
import os
import sys
import time
from typing import IO, List, Tuple

import numpy as np


def _open_text(path: str) -> IO:
    """根据扩展名自动选择 bz2 / 普通文本打开。"""
    if path.endswith(".bz2"):
        return bz2.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "r", encoding="utf-8", errors="ignore")


def load_word2vec_text(path: str) -> Tuple[List[str], np.ndarray, int, int]:
    """
    读取 word2vec 文本格式词向量。
    :return: (words, X, V_actual, dim)
        words: 长度为 V_actual 的词列表
        X    : float32 (V_actual, dim) numpy 矩阵
        V_actual: 实际有效行数（可能 < header 中 V，因为跳过非法行）
        dim  : 维度
    """
    print(f"[load] {path}")
    t0 = time.time()
    with _open_text(path) as f:
        header = f.readline().strip().split()
        if len(header) == 2 and header[0].isdigit() and header[1].isdigit():
            V_header, dim = int(header[0]), int(header[1])
            print(f"[load] header: V={V_header}, dim={dim}")
        else:
            # 没有 header，把首行当数据，从词数据推测 dim
            # 罕见情况，回退到逐行读完后估计
            raise ValueError(
                "首行不是 'V dim' 头，请检查文件格式。"
            )

        words: List[str] = []
        # 预分配以减少内存碎片；如果跳过太多行后会有未填满的尾巴，最后再裁
        X = np.empty((V_header, dim), dtype=np.float32)
        idx = 0
        skipped = 0
        for line_no, line in enumerate(f, start=2):
            # 用 split() 而非 split(" ")：自动合并连续空白并忽略行尾空字符串
            # （CWV 发布的部分文件行尾有 trailing space，会导致 split(" ") 多出一项）
            parts = line.split()
            # 与 Embed.py _read_file 一致：跳过 1/2/3 列的奇怪行
            if len(parts) <= 3:
                skipped += 1
                continue
            word = parts[0]
            vec_strs = parts[1:]
            if len(vec_strs) != dim:
                skipped += 1
                continue
            try:
                X[idx] = np.fromiter((float(x) for x in vec_strs), dtype=np.float32, count=dim)
            except ValueError:
                skipped += 1
                continue
            words.append(word)
            idx += 1
            if idx % 50000 == 0:
                sys.stdout.write(f"\r[load] {idx} / {V_header}")
                sys.stdout.flush()
        print()
    X = X[:idx]
    V_actual = idx
    print(f"[load] V_actual={V_actual}, dim={dim}, skipped={skipped}, time={time.time()-t0:.1f}s")
    return words, X, V_actual, dim


def pca_reduce(X: np.ndarray, target_dim: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    对 V × D 矩阵做 PCA 降到 V × K。
    :return: (X_K, mean, components)  其中 components 是 (D, K) 投影矩阵
    """
    V, D = X.shape
    assert target_dim <= D, f"target_dim {target_dim} > input dim {D}"
    print(f"[pca] V={V}, D={D}, target={target_dim}")
    t0 = time.time()

    # 中心化
    mean = X.mean(axis=0)
    Xc = X - mean
    print(f"[pca] centered, ||mean||={np.linalg.norm(mean):.4f}")

    # 协方差矩阵 D × D（很小，300×300）
    C = (Xc.T @ Xc) / max(V - 1, 1)
    print(f"[pca] cov shape={C.shape}, time so far={time.time()-t0:.1f}s")

    # 特征分解，eigh 返回升序，最后 K 个就是最大的
    eigvals, eigvecs = np.linalg.eigh(C)
    print(f"[pca] eigvals min={eigvals[0]:.4e}, max={eigvals[-1]:.4e}")

    # 取最大 K 个并降序排列
    top_k_idx = np.argsort(eigvals)[::-1][:target_dim]
    components = eigvecs[:, top_k_idx]  # (D, K)
    explained = eigvals[top_k_idx]
    total = eigvals.sum()
    pct = 100.0 * explained.sum() / total
    print(f"[pca] retained explained variance: {pct:.2f}% ({target_dim}/{D} components)")

    # 投影
    X_K = Xc @ components  # (V, K)
    print(f"[pca] projected, time={time.time()-t0:.1f}s")
    return X_K.astype(np.float32), mean, components.astype(np.float32)


def save_word2vec_text(words: List[str], X: np.ndarray, path: str) -> None:
    """word2vec 文本格式落盘"""
    V, K = X.shape
    print(f"[save] {path} ({V} × {K})")
    t0 = time.time()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # 不预格式化整行，分块输出避免内存爆
    fmt = " ".join(["%.6f"] * K)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{V} {K}\n")
        for i in range(V):
            f.write(words[i] + " " + (fmt % tuple(X[i])) + "\n")
            if (i + 1) % 50000 == 0:
                sys.stdout.write(f"\r[save] {i+1} / {V}")
                sys.stdout.flush()
    print()
    sz = os.path.getsize(path) / 1024 / 1024
    print(f"[save] done. file size {sz:.1f} MB, time={time.time()-t0:.1f}s")


def main():
    ap = argparse.ArgumentParser(description="PCA 降维 word2vec 文本格式词向量")
    ap.add_argument("--input", required=True, help="输入文件 (.txt 或 .bz2)")
    ap.add_argument("--output", required=True, help="输出 .txt 路径")
    ap.add_argument("--dim", type=int, required=True, help="目标维度，例如 100")
    args = ap.parse_args()

    words, X, V, D = load_word2vec_text(args.input)
    if args.dim == D:
        print(f"[main] target_dim == input_dim ({D})，直接拷贝")
        save_word2vec_text(words, X, args.output)
        return
    X_K, mean, comps = pca_reduce(X, args.dim)
    save_word2vec_text(words, X_K, args.output)


if __name__ == "__main__":
    main()
