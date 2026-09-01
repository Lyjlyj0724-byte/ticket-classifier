"""M4：误差分析 —— 导出 val 集误分样本，按"真实类→预测类"聚桶。

用法: python src/error_analysis.py [exp_id]   （默认 e1_word_tfidf_lr）

产出：reports/val_errors_<exp>.csv（误分样本，供人工逐条阅读）
"""

import sys

import pandas as pd

from common import ROOT, load_split
from experiments import build


def main(exp_id: str = "e1_word_tfidf_lr") -> None:
    train, val = load_split("train"), load_split("val")
    pipe = build(exp_id)
    pipe.fit(train["text"], train["type"])

    # 误分样本 + 低置信的"侥幸分对"样本（同样值得人工读，是明天的错误）
    proba = pipe.predict_proba(val["text"])
    pred = pipe.classes_[proba.argmax(axis=1)]
    conf = proba.max(axis=1)
    val = val.assign(pred=pred, conf=conf.round(3))
    errors = val[val["pred"] != val["type"]].sort_values("conf")
    lucky = val[(val["pred"] == val["type"]) & (val["conf"] < 0.6)].sort_values("conf")

    out = pd.concat([errors, lucky])[["ticket_id", "subject", "description", "type", "pred", "conf", "channel", "priority"]]
    path = ROOT / "reports" / f"val_errors_{exp_id}.csv"
    out.to_csv(path, index=False, encoding="utf-8")
    print(f"误分 {len(errors)} 条 + 低置信侥幸分对 {len(lucky)} 条 → {path}\n")
    print("真实→预测 聚桶:")
    print(errors.groupby(["type", "pred"]).size().sort_values(ascending=False).to_string())
    print(f"\n误分样本中通用标题占比: {(errors.subject.str.len() <= 5).mean():.0%}")


if __name__ == "__main__":
    main(*sys.argv[1:])
