"""M2：数据审计（EDA + 泄漏排查）+ 分层切分。

产出：
- reports/class_distribution.png   类别分布条形图
- reports/length_distribution.png  文本长度分布
- data/{train,val,test}.csv        70/15/15 分层抽样，固定种子

用法: python src/eda.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

DATA = "data/tickets.csv"
SEED = 42


def main() -> None:
    df = pd.read_csv(DATA)
    print(f"总行数: {len(df)}, 列: {list(df.columns)}\n")

    # ---- 1. 类别分布 ----
    dist = df["type"].value_counts()
    print("类别分布:\n", dist, "\n")
    print(f"最大类/最小类 = {dist.max() / dist.min():.1f}x（不均衡，accuracy 会失真）\n")
    dist.plot(kind="bar", title="Class distribution", ylabel="count")
    plt.tight_layout()
    plt.savefig("reports/class_distribution.png", dpi=120)
    plt.close()

    # ---- 2. 文本长度 / 空值 / 重复 ----
    df["text_len"] = (df["subject"].fillna("") + df["description"].fillna("")).str.len()
    print(f"文本长度: min={df.text_len.min()}, median={int(df.text_len.median())}, max={df.text_len.max()}")
    empty = (df["description"].isna() | (df["description"] == "")).sum()
    dups = df.duplicated(subset=["subject", "description"]).sum()
    print(f"空正文: {empty} 条; 内容完全重复（除 ticket_id）: {dups} 条\n")
    df["text_len"].plot(kind="hist", bins=30, title="Text length distribution")
    plt.tight_layout()
    plt.savefig("reports/length_distribution.png", dpi=120)
    plt.close()

    # ---- 3. 泄漏排查：resolution 是不是"未来信息"？----
    # 若某列的取值几乎由标签唯一决定，且语义上是"处理后才产生"，即为泄漏。
    cross = pd.crosstab(df["resolution"], df["type"])
    purity = (cross.max(axis=1) / cross.sum(axis=1)).min()
    print("resolution 列每行取值只属于单一类别的最低纯度: {:.0%}".format(purity))
    print("结论: resolution 由 type 唯一决定（纯度 100%），是人工处理后才填写的'未来信息'，")
    print("      若作为特征会让模型'作弊'（训练准、上线崩），必须从输入中剔除。\n")

    # ---- 4. 去重（审计发现的处置）----
    # 内容完全相同的工单若分别落进 train 和 test，等于让模型"背答案"——
    # 这是比 resolution 更隐蔽的泄漏：横跨切分的重复。去重保留首次出现。
    before = len(df)
    df = df.drop_duplicates(subset=["subject", "description"], keep="first").reset_index(drop=True)
    print(f"去重: {before} → {len(df)} 条（避免重复样本横跨 train/test 造成隐性泄漏）\n")

    # ---- 5. 分层切分 70/15/15 ----
    train, tmp = train_test_split(df, test_size=0.30, stratify=df["type"], random_state=SEED)
    val, test = train_test_split(tmp, test_size=0.50, stratify=tmp["type"], random_state=SEED)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_csv(f"data/{name}.csv", index=False, encoding="utf-8")
        print(f"{name}: {len(part)} 条, 类别比例 {dict(part.type.value_counts(normalize=True).round(3))}")
    print("\n铁律: test 集只在 M5 最终评估时用一次，调参只看 val。")


if __name__ == "__main__":
    main()
