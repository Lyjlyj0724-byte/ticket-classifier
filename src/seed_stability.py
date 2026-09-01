"""M4 收尾：e1 vs e2 的提升是真改进还是噪声？—— bootstrap 置信区间。

对 val 集做 1000 次有放回重采样，统计 macro-F1 差值的分布。
若 95% CI 包含 0，则差异不显著（不能让单次 +0.01 骗过你）。

用法: python src/seed_stability.py
"""

import numpy as np
from sklearn.metrics import f1_score

from common import load_split
from experiments import build


def main() -> None:
    train, val = load_split("train"), load_split("val")
    preds = {}
    for exp_id in ["e1_word_tfidf_lr", "e2_char_tfidf_lr"]:
        pipe = build(exp_id)
        pipe.fit(train["text"], train["type"])
        preds[exp_id] = pipe.predict(val["text"])

    y = val["type"].to_numpy()
    rng = np.random.default_rng(42)
    diffs = []
    n = len(y)
    for _ in range(1000):
        idx = rng.integers(0, n, n)  # 有放回抽样
        f1_1 = f1_score(y[idx], preds["e1_word_tfidf_lr"][idx], average="macro")
        f1_2 = f1_score(y[idx], preds["e2_char_tfidf_lr"][idx], average="macro")
        diffs.append(f1_2 - f1_1)
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"e2 - e1 macro-F1 差值: 均值 {diffs.mean():+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}]")
    print("CI 包含 0 → 差异不显著，+0.014 是噪声级波动；e2 不是真改进。" if lo <= 0 <= hi else "CI 不含 0 → 差异显著。")


if __name__ == "__main__":
    main()
