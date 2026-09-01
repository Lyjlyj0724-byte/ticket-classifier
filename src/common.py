"""公共组件：数据加载、评估、结果记录。

设计要点：
- 文本输入 = subject + description（分辨率泄漏列 resolution 永不进入输入）
- evaluate() 输出 macro-F1、按类别 P/R、recall(bug)，混淆矩阵存 reports/
- 结果追加到 reports/experiments.csv，支撑 docs/experiment-log.md
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
DATA_VERSION = "synthetic-v2 (N=2000+3%dup, dedup→1079, seed=42)"

CLASSES = ["billing", "bug", "account", "feature_request", "performance"]


def load_split(name: str) -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / f"{name}.csv").fillna({"subject": "", "description": ""})
    df["text"] = (df["subject"] + " " + df["description"]).str.strip()
    return df


def evaluate(exp_id: str, y_true, y_pred, notes: str = "") -> dict:
    """评估一次实验：打印报告、存混淆矩阵、追加结果行。"""
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred, labels=CLASSES, digits=3, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    print(f"===== {exp_id} =====")
    print(report)
    print("混淆矩阵（行=真实，列=预测）:", CLASSES)
    print(cm)
    recall_bug = cm[CLASSES.index("bug"), CLASSES.index("bug")] / cm[CLASSES.index("bug")].sum()
    print(f"macro-F1={macro_f1:.3f}  recall(bug)={recall_bug:.3f}\n")

    row = {"exp_id": exp_id, "macro_f1": round(macro_f1, 4), "recall_bug": round(float(recall_bug), 4), "notes": notes}
    out = REPORTS / "experiments.csv"
    header = not out.exists()
    with out.open("a", encoding="utf-8") as f:
        if header:
            f.write("exp_id,macro_f1,recall_bug,notes\n")
        f.write(f"{row['exp_id']},{row['macro_f1']},{row['recall_bug']},{notes}\n")
    (REPORTS / f"cm_{exp_id}.json").write_text(
        json.dumps({"labels": CLASSES, "matrix": cm.tolist()}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return row
