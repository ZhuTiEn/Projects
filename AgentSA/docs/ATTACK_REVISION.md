# 正式规模与目标相关 Agent 增强版

## 本次修改的目的和范围

原 pilot 仅注入 3 个假用户，目标的原始排名中位数在 7,000 名以后。排名计算没有发现导致 HR 全零的错误。
因此增加正式 5% 注入配置和独立的目标相关增强版，以检验更大预算、更密集评分历史能否增加曝光。
不能通过修改评估函数保证大多数目标成功。全部目标、训练种子与失败项都必须如实报告。

HR@K 是“目标进入真实用户前 K 名的比例”，物品本身没有一个叫 HR@K 的排名。
“多数目标成功”在新报告中有两种明确口径：

- `fraction_targets_any_hit_at_K`：HR@K 大于 0 的目标占比。
- `fraction_targets_minimum_hr_at_K`：HR@K 至少达到预先设定值的目标占比，默认阈值 0.01，即 1% 评估用户。

两种比例都先对同一目标的训练种子取平均，再按所有被评估目标计算；零命中目标不剔除。
超过一半才表示多数目标达到对应条件。第二个阈值用于描述结果，不用于模型选择或挑目标。

## 配置

| 配置 | Agent 设置 | 作用 |
|---|---|---|
| `configs/formal_original.yaml` | 原流行度曝光、自然选品、992 人/目标 | 原评分型协议的正式规模对照 |
| `configs/formal_target_aware.yaml` | 目标相关曝光、密集选品、992 人/目标 | 新的增强攻击变体 |
| `configs/scale_check.yaml` | 不加入 Agent，只比较 target_only/Average | 无 API 的单目标注入规模检查 |

两个正式配置均使用验证集检查选出的学习率 0.0003、最多 100 轮、早停耐心值 10；保留原冻结的 10 个目标和 3 个训练种子。
原版和增强版使用独立的实验/Agent 输出目录，不复用 3 人 pilot 数据。

## 增强版如何生成

1. 根据公开商品元数据的语义向量，取目标最相近的 200 个训练中出现过的商品。
2. 候选曝光权重为 50% 流行度分布和 50% 语义邻域分布。语义部分使用温度 0.1 的指数权重。
3. 各轮仍排除目标、已交互商品，且无放回抽样曝光。
4. 让 LLM 根据画像从每轮候选中选 4 个商品并评分；5 轮后通常每人有 20 个 filler。
5. 评分仍必须是 1～5 的整数，不强制 filler 全为 5 分；目标保持 5 分。

该变体没有读取真实用户历史、验证/测试标签、受害模型预测或梯度来生成攻击。
它改变了原 AgentSA 的曝光/选品协议，应称为 `target_aware_dense_v1`，不能混称为原论文方法。
当剩余预算或候选不足时，最低选品数会按可用容量降低，实际预算会记录在 manifest。
不能把模拟接口产生的通过结果当成真实增强版攻击的有效性证据。

## 运行

可用一个入口自动执行“接口检查 → 第一个目标的真实评分生成 → 两模型、种子11的全部攻击预算匹配比较”：

```powershell
Set-Location C:\Projects\AgentSA
.venv\Scripts\python.exe -m agentsa.run_enhanced
```

如果当前终端没有 Key，入口会隐藏输入地询问一次，只放在本进程和子进程环境中，不保存到文件。
Key 已设置则直接使用；评分已完整生成则校验后直接进入训练，无需 Key 或新增 API 调用。
任一步失败会停止后续阶段，状态保存在 `runs/formal_target_aware_5pct/workflow_status.json`。
再次执行可继续已有生成进度，并复用配置一致的已完成模型。此入口只运行第一个冻结目标，不自动生成全部10个目标。
可加 `--check-only` 做完全离线的就绪检查。

先启动一个目标的增强版生成。该步骤调用真实 API，需要在本终端设置 Key；不要在未估算预算时直接生成全部目标。

```powershell
Set-Location C:\Projects\AgentSA
$env:PYTHONIOENCODING = 'utf-8'
$deepseekSecret = Read-Host '输入 DeepSeek API Key' -AsSecureString
$env:DEEPSEEK_API_KEY = [System.Net.NetworkCredential]::new('', $deepseekSecret).Password

.venv\Scripts\python.exe -m agentsa --config configs/formal_target_aware.yaml estimate
.venv\Scripts\python.exe -m agentsa --config configs/formal_target_aware.yaml generate-agent --target 1920
```

生成成功后，用同一目标和一个训练种子比较全部方法。此步骤只在本地训练，不再调用 API：

```powershell
.venv\Scripts\python.exe -m agentsa --config configs/formal_target_aware.yaml run --target-count 1 --seeds 11 --include-agent --match-agent-budget
```

对照原协议可将配置换为 `configs/formal_original.yaml`，需另行生成原协议的真实 Agent 数据。
两个协议分别匹配各自实际 filler 数量；比较协议本身时也要报告评分数量，避免把预算差异全部归因于选品机制。

## 新结果字段

`results.csv` 同时包含：

- `clean_hr_at_10/50/100`、`poison_hr_at_10/50/100`、`delta_hr_at_10/50/100`。
- `mean_rank_improvement`、假用户比例、实际评分条数和 `agent_variant`。
- 原有 `clean_hr/poison_hr/delta_hr` 保留，仍对应配置中的 `top_k`。

`summary.json` 同时汇总三个 K 的均值/标准差。
`target_coverage.json` 保存逐目标结果、任意命中目标占比和达到预设曝光率的目标占比。

增强版输出目录：`runs/formal_target_aware_5pct/experiments/`。
单目标检查只能说明这个目标，不能据此声称多数目标成功。

评估继续使用全商品、固定真实用户人群、屏蔽训练已见商品和原有同分排序规则。
没有加分后处理、没有缩小候选集、没有替换目标列表或删掉零命中结果。

## 兼容与验证

模型结构和训练实现未改变，历史 checkpoint 的训练指纹仍可验证。
旧评估文件不会自动补写新字段，新配置使用独立目录保存新结果。
新生成清单保存变体与生成设置；不匹配的变体或设置不能直接混入后续比较。
变更前代码备份：`runs/attack_revision_20261102/source_before/`。

本次自动检查覆盖语义候选排除、确定性抽样、最低选品数、预算不足边界、多 K 指标、零结果保留以及不访问网络的模拟生成/预算匹配/训练端到端流程。
