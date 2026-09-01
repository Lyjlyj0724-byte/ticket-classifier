"""CI 冒烟测试：用 100 条小样本能跑通完整训练/预测链路即可，不跑全量训练。"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from experiments import build  # noqa: E402


def tiny_dataset(n: int = 100) -> pd.DataFrame:
    samples = [
        ("退款没到账请处理", "billing"),
        ("页面报错白屏了", "bug"),
        ("无法登录账号被锁", "account"),
        ("希望增加新功能", "feature_request"),
        ("系统很慢加载超时", "performance"),
    ]
    rows = [samples[i % len(samples)] for i in range(n)]
    return pd.DataFrame(rows, columns=["text", "type"])


def test_pipeline_smoke():
    df = tiny_dataset()
    pipe = build("e1_word_tfidf_lr")
    pipe.fit(df["text"], df["type"])
    pred = pipe.predict(df["text"])
    assert set(pred) <= {"billing", "bug", "account", "feature_request", "performance"}
    # 冒烟只要求链路通、明显高于随机（5 类随机 macro-F1 ≈ 0.2）
    assert f1_score(df["type"], pred, average="macro") > 0.3
