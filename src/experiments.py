"""M3/M4：基线阶梯与迭代实验。

用法: python src/experiments.py e0 e1 ...   （按 id 跑指定实验，评估在 val 集）

实验阶梯：
- e0 L0 关键词规则            —— 理解问题难度下限；打不过它的模型没价值
- e1 L1 word 级 TF-IDF + LR   —— 经典强基线（但对中文有个坑，见实验日志）
- e2 L2 char n-gram TF-IDF + LR（由 M4 误差分析驱动）
- e3 L2 + class_weight='balanced'（由 M4 误差分析驱动）
"""

import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from common import CLASSES, evaluate, load_split

# L0：关键词 → 类别。人工拍脑袋的规则，正是生产里最常见的"先顶上"方案。
RULES = {
    "billing": ["发票", "退款", "账单", "扣款", "支付", "续费", "费用", "多收"],
    "bug": ["报错", "崩溃", "白屏", "异常", "无法", "失败", "丢失"],
    "account": ["登录", "密码", "账号", "绑定", "权限", "注销", "验证码"],
    "feature_request": ["希望", "建议", "能否", "支持", "增加"],
    "performance": ["慢", "超时", "卡顿", "响应", "加载"],
}


def rule_predict(text: str) -> str:
    scores = {c: sum(text.count(kw) for kw in kws) for c, kws in RULES.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "billing"  # 无命中 → 多数类兜底


def build(exp_id: str):
    if exp_id == "e1_word_tfidf_lr":
        # L1：默认按空格分词——对英文是经典做法，对中文是坑（整句成一个词）
        return Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ])
    if exp_id == "e2_char_tfidf_lr":
        # L2-iter1：char_wb 2~4 字 n-gram，绕开中文分词问题
        return Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2)),
            ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ])
    if exp_id == "e3_char_balanced":
        # L2-iter2：e2 + 类别权重均衡，抬小类 performance 的召回
        return Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2)),
            ("lr", LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")),
        ])
    raise ValueError(f"unknown experiment: {exp_id}")


def main(ids: list[str]) -> None:
    train, val = load_split("train"), load_split("val")
    for exp_id in ids:
        if exp_id == "e0_rules":
            y_pred = val["text"].map(rule_predict)
            evaluate(exp_id, val["type"], y_pred, "L0 关键词规则")
            continue
        pipe = build(exp_id)
        pipe.fit(train["text"], train["type"])
        evaluate(exp_id, val["type"], pipe.predict(val["text"]), "见实验日志")


if __name__ == "__main__":
    main(sys.argv[1:] or ["e0_rules", "e1_word_tfidf_lr"])
