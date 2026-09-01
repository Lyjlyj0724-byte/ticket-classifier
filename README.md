# 工单分类：传统 ML 基线（Case 02）

[![CI](https://github.com/Lyjlyj0724-byte/ticket-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/Lyjlyj0724-byte/ticket-classifier/actions/workflows/ci.yml)

给客服工单打类别标签（bug / billing / feature_request / account / performance）的传统 ML 基线。
重点不是模型代码，而是完整的方法论闭环：问题定义 → 数据审计 → 基线阶梯 → 误差分析 → 切片评估。

## 文档（先看这些）

- [docs/problem-definition.md](docs/problem-definition.md) — 输入/输出/指标/代价/边界（M1）
- [docs/experiment-log.md](docs/experiment-log.md) — 5 条实验记录，含统计显著性检验（M3/M4）
- [docs/error-analysis.md](docs/error-analysis.md) — 60 条人工阅读的错误桶分析（M4）
- [docs/model-report.md](docs/model-report.md) — test 终评 + 切片评估 + 下一步建议（M5）

## 数据

`data/` 不进 git。复现方式（合成数据，固定种子）：

```bash
python src/generate_data.py   # 生成 data/tickets.csv（2060 条，含刻意埋入的泄漏列与歧义样本）
python src/eda.py             # EDA + 泄漏排查 + 去重 + 分层切分
```

若要换真实数据：Kaggle 搜 "customer support tickets"，要求含 标题/正文/类别 三列，放入 `data/tickets.csv` 后跑 `src/eda.py`。

## 运行

```bash
uv venv && uv pip install -r requirements.txt
python src/experiments.py e0_rules e1_word_tfidf_lr   # 基线阶梯
python src/error_analysis.py                          # 误差分析
python src/seed_stability.py                          # 显著性检验
python src/final_eval.py                              # test 终评 + 切片
```

## 结果速览

champion = TF-IDF(word) + LogisticRegression：test macro-F1 0.931，recall(bug) 0.918，全部达标。
最大发现：约一半误分是标注不一致/零信号工单，属非模型问题，解法是标注规范 + 低置信转人工。
