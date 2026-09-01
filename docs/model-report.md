# 模型报告：工单自动分类基线

## 问题

输入工单标题+正文，输出 5 类之一（bug/billing/feature_request/account/performance）。详见 [problem-definition.md](problem-definition.md)。

## 数据

- 合成数据 synthetic-v2（Kaggle 需登录，且合成数据可在 CI 复现；生成器：`src/generate_data.py`，seed=42）
- 生成 2060 条 → EDA 去重（防跨切分泄漏）后 1864 条；分层切分 train/val/test = 1346/280/289
- **泄漏排查结论**：`resolution` 列由标签唯一决定（纯度 100%），是处理后才产生的"未来信息"，已从输入剔除；另发现并处置了跨切分重复样本

## 方法与基线阶梯

| 级别 | 方法 | val macro-F1 | recall(bug) | 结论 |
|---|---|---|---|---|
| L0 | 关键词规则 | 0.837 | 0.671 ❌ | 打不过约束，仅作参照 |
| L1（champion） | TF-IDF(word) + LR，Pipeline 串联 | 0.946 | 0.959 ✅ | 上线基线 |
| L2 | char n-gram / class_weight 两轮迭代 | 0.960 / 0.949 | — | 均不显著或破坏约束，未采纳（见[实验日志](experiment-log.md)） |

迭代由误差分析驱动（[error-analysis.md](error-analysis.md)，60 条人工阅读，4 个错误桶）；最大桶是**标注不一致**，属非模型问题。

## 最终指标（test 集，仅用一次；模型在 train+val 重训）

- **macro-F1 = 0.931**（val 0.946，泛化落差正常）
- **recall(bug) = 0.918** ≥ 0.85 ✅
- 全部满足 M1 目标（macro-F1 ≥ 0.75、recall(bug) ≥ 0.85）

### 按切片（test）

| 切片 | 最差 | 最好 | 结论 |
|---|---|---|---|
| 渠道 | phone 0.899 | web 0.983 | phone 工单更口语化、更短，偏弱 |
| 优先级 | low 0.922 | medium 0.949 | 差异不大，high 0.929 无崩盘 ✅ |
| 文本长度 | **short 0.899** | long 0.986 | 短文本是最弱切片，与 M4 桶③（零信号工单）一致 |

**整体达标但短文本切片明显偏弱** —— 已由"低置信/空文本转人工"的上线策略覆盖。

## 主要失败模式

1. 标注不一致（同一措辞两种标签）—— 需标注规范，模型无解
2. bug/performance 词汇重叠（"卡住/超时"）—— char n-gram 有微弱帮助但不显著
3. 空正文/极短工单 —— 转人工规则

## 下一步建议

1. 产品侧：标注规范 + 置信度 <0.6 转人工（预计把桶①③的影响清零）
2. 数据侧：换真实工单数据重训（合成数据分布偏干净）；重点补 performance 小类样本
3. 模型侧：真实数据上重验 char n-gram；若真实文本更长更杂，考虑 fastText/小型中文预训练模型
4. 监控：上线后按周看 macro-F1 与 recall(bug)，数据分布漂移（新渠道上线）最先会在**切片指标**上暴露

## 复现

```bash
uv venv && uv pip install scikit-learn pandas matplotlib pytest
python src/generate_data.py && python src/eda.py      # 数据 + 切分
python src/experiments.py e0_rules e1_word_tfidf_lr   # 基线
python src/error_analysis.py                          # 误差分析
python src/final_eval.py                              # test 终评
```
