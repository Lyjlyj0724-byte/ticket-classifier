"""M5：最终评估 —— champion(e1) 在 train+val 上重训，test 集只评一次，含切片分析。

用法: python src/final_eval.py
产出: reports/final_test_metrics.json
"""

import json

import pandas as pd
from sklearn.metrics import f1_score, recall_score

from common import CLASSES, ROOT, evaluate, load_split
from experiments import build


def slice_metrics(df, y_pred, by) -> dict:
    out = {}
    for key in df[by].unique():
        mask = (df[by] == key).to_numpy()
        out[str(key)] = {
            "n": int(mask.sum()),
            "macro_f1": round(float(f1_score(df["type"].to_numpy()[mask], y_pred[mask], average="macro", zero_division=0)), 3),
        }
    return out


def main() -> None:
    train, val, test = load_split("train"), load_split("val"), load_split("test")
    # 最终模型：train+val 合并重训（常规做法，信息利用最大化）
    full = pd.concat([train, val])
    pipe = build("e1_word_tfidf_lr")
    pipe.fit(full["text"], full["type"])
    y_pred = pipe.predict(test["text"])

    row = evaluate("final_test", test["type"], y_pred, "champion e1, train+val 重训, test 唯一一次")

    # 切片：渠道 / 优先级 / 文本长度
    median_len = full["text"].str.len().median()
    test = test.assign(len_bucket=(test["text"].str.len() > median_len).map({True: "long", False: "short"}))
    metrics = {
        "overall": row,
        "by_channel": slice_metrics(test, y_pred, "channel"),
        "by_priority": slice_metrics(test, y_pred, "priority"),
        "by_length": slice_metrics(test, y_pred, "len_bucket"),
        "recall_bug": recall_score(test["type"], y_pred, labels=CLASSES, average=None, zero_division=0)[CLASSES.index("bug")].round(3),
    }
    (ROOT / "reports" / "final_test_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
