# AgentSA 第一阶段复现

已增加离线的干净模型选参与 pilot 排名诊断，入口和复现步骤见 [docs/PREFLIGHT.md](docs/PREFLIGHT.md)。该步骤不调用 LLM API，使用独立输出目录保存检查结果。

正式 5% 注入与目标相关 Agent 增强版见 [docs/ATTACK_REVISION.md](docs/ATTACK_REVISION.md)。增强版独立标记为 `target_aware_dense_v1`，评估同时输出 HR@10/50/100 和目标覆盖率，不保证预设攻击成功率。

评论攻击、目标特征传播及评论感知 Dual-Tower 工程适配见 [docs/REVIEW_ATTACK.md](docs/REVIEW_ATTACK.md)。该阶段使用独立缓存和输出目录，不改变既有评分 Agent 身份与已付费请求缓存。

这是根据论文方法编写的独立实现，**不是作者官方代码**。当前范围：Amazon 2014 5-core Pet Supplies 全时间范围，NMF、NeuNMF，六种传统评分攻击，以及可更换 LLM 的评分型 AgentSA。

已添加 DeepSeek 专用配置 `configs/deepseek.yaml`，用法见 `docs/DEEPSEEK.md`。尚未发起真实 LLM API 调用。`--mock` 是确定性接口替身，仅验证程序连接，不能用来论证 AgentSA 效果。

## 1. 在 VS Code 中打开

在 PowerShell 输入：

```powershell
code C:\Projects\AgentSA
```

项目已提供 `.vscode/settings.json` 和 `launch.json`。解释器选择 `.venv\Scripts\python.exe`，运行和调试面板中有环境检查、准备数据、小样本实验、模拟 Agent 等入口。

当前 `.venv` 使用本机已有 Python/PyTorch 包创建，避免重复下载大体积 PyTorch。它继承系统包，并非完全隔离的锁定环境；当前依赖快照见 `docs/environment_snapshot.txt`。换机器可重新建立独立环境：

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e .
```

后续真实 Agent 需要语义检索依赖：

```powershell
.venv\Scripts\python.exe -m pip install -e ".[semantic]"
```

语义依赖固定为 sentence-transformers 3.4.1 / transformers 4.48.3，以兼容当前 PyTorch 2.5.1。遇到 Transformers 要求 PyTorch >= 2.6 的提示时，重新执行上面的安装命令恢复项目版本。

生成用的 LLM 未固定；语义检索编码器暂配置为 `sentence-transformers/all-MiniLM-L6-v2`，可以更换。首次使用语义编码器可能下载模型权重，也可配置本地路径和 `encoder_local_only: true`。正式复现应固定权重版本或保留本地模型副本。

## 2. 推荐阅读顺序

| 文件 | 主要内容 |
|---|---|
| `configs/phase1.yaml` | 全量数据、模型、攻击、评测、API 限额 |
| `agentsa/cli.py` | 命令入口，理解实验如何串起来 |
| `agentsa/data.py` | 原始文件读取、去重、5-core、划分、目标列表 |
| `agentsa/models.py` | NMF 与 NeuNMF 的模型定义 |
| `agentsa/training.py` | 训练、验证集早停、模型保存 |
| `agentsa/attacks.py` | 六种传统攻击及仅目标评分对照 |
| `agentsa/agents.py` | 画像生成、验证/筛选、曝光、记忆、选择与评分 |
| `agentsa/llm.py` | 模型调用、缓存、重试和限额 |
| `agentsa/evaluation.py` | 全物品排序、HR@10、评分误差 |
| `agentsa/experiment.py` | 干净/投毒训练、预算匹配、结果汇总 |
| `docs/PROTOCOL.md` | 原文设置、实现约定与复现边界 |

## 3. 数据已放在项目根目录

```text
reviews_Pet_Supplies_5.json.gz
meta_Pet_Supplies.json.gz
```

原始压缩包只读，不移动、不覆盖。代码支持 JSON 和历史 Python 字面量格式；使用 `ast.literal_eval`，不执行数据中的代码。

已经核对原始交互：19,856 个用户、8,510 个物品、157,836 条记录。该规模是**完整类别数据**，不能直接套用论文按一年筛选后的 Pets 规模。处理后的准确数量以 `data/processed/full/manifest.json` 为准。

## 4. 先验证环境和数据

以下命令均在项目根目录的 VS Code 终端执行，无需激活环境：

```powershell
.venv\Scripts\python.exe -m agentsa doctor
.venv\Scripts\python.exe -m agentsa prepare
.venv\Scripts\python.exe -m agentsa estimate
```

`prepare` 输出固定 train/validation/test、ID 映射、商品信息、目标物品和哈希清单。重复执行会验证后复用。修改数据处理参数时请换一个 `data.output` 目录，避免悄悄覆盖既有划分。

`estimate` 不连接模型服务。它报告假用户数和预估请求数；全量数据按 5% 向下取整约为 992 个假用户/目标，正式生成前务必看实际成本估算。

## 5. 小样本检查（不调用真实 LLM）

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe -m agentsa --config configs/smoke.yaml prepare
.venv\Scripts\python.exe -m agentsa --config configs/smoke.yaml generate-agent --mock
.venv\Scripts\python.exe -m agentsa --config configs/smoke.yaml run --include-agent --match-agent-budget --mock-agent
```

此处烟雾样本经过 160 用户抽样后重新做 2-core，仅用于验证代码；不属于正式全量 5-core 实验。当前样本为 94 用户、128 物品、327 交互。两种模型各训练 8 个攻击配置：六基线、仅目标评分、模拟 Agent。输出在 `runs/smoke_checked/mock_validation/`，所有结果带 `mock_validation=true`。

`--config` 必须放在子命令之前。模拟 Agent 使用词袋哈希检索作为测试替身；真实生成禁止静默回退到该检索器。

## 6. 先运行真实数据上的传统攻击

先训练干净模型：

```powershell
.venv\Scripts\python.exe -m agentsa train-clean --seeds 11
```

再选一个目标、一个训练种子检查六种基线及仅目标评分对照：

```powershell
.venv\Scripts\python.exe -m agentsa run --target-count 1 --seeds 11
```

正式默认配置（运行量更大，当前没有自动执行）：

```powershell
.venv\Scripts\python.exe -m agentsa run
```

默认 10 个独立目标、3 个训练种子、2 种模型。每个目标的攻击数据独立生成，并跨模型/训练种子复用。基线默认每人 10 个 filler；Bandwagon、Segmented 的热门 selected 计入这 10 个。`target_only` 仅给目标最高分，刻意作为低交互量参照。

## 7. 选好模型后，启用真实 Agent

目前适配的是 **chat-completions 兼容协议**，不是对所有厂商 API 的通用承诺。若供应商使用其他请求格式，需要在 `llm.py` 添加独立适配器。下面是配置示意，请填写服务方提供的地址和模型名称：

```powershell
$env:AGENTSA_API_KEY = "你的调用凭据"
$env:AGENTSA_BASE_URL = "https://服务商地址/v1"
$env:AGENTSA_MODEL = "服务商模型名称"
.venv\Scripts\python.exe -m agentsa doctor
.venv\Scripts\python.exe -m agentsa generate-agent
```

最后一条命令会真实调用服务，**默认只生成冻结列表中的第一个目标**。`.env.example` 只是说明，不会自动加载。Key 不写入配置、缓存或日志。可信本地服务可配置 `allow_no_api_key: true`。

每个 Agent 先构建画像，验证偏好并筛选多样性；先给目标写入 5 分，再进行曝光、选品、记忆检索和评分。代码将共享同一 LLM 客户端，每个假用户单独保存状态；不需要每用户一个 Key。

如果默认请求上限不足，生成会在上限处停止并保留状态，而不是无上限继续。可查看 `runs/agent_data/real/target_<id>/llm/usage.json` 后调整限额，再执行同一命令续跑。供应商未返回 token usage 时会记录计数，token 预算不能精确核算；请求数上限仍生效。最后一次响应可能使 token 累计略超限。正式生成前建议复制配置，降低假用户比例/目标数，并采用新的 `agent.output` 目录做 pilot。

为指定目标生成（ID 从 `data/processed/full/targets.json` 取）：

```powershell
.venv\Scripts\python.exe -m agentsa generate-agent --target 目标ID
```

生成全部 10 个目标需要显式指定：

```powershell
.venv\Scripts\python.exe -m agentsa generate-agent --all-targets
```

之后比较真实 Agent 与传统攻击，并匹配每个假用户的 filler 数量：

```powershell
.venv\Scripts\python.exe -m agentsa run --target-count 1 --include-agent --match-agent-budget
```

去掉 `--match-agent-budget` 则使用传统基线固定 10 条 filler、Agent 自然生成数量的协议。两者必须分开报告。若 Agent 实际每人选择数超过 AoP 的候选池容量，程序明确报错，不能偷偷缩减预算。

## 8. 结果在哪里、怎么看

```text
data/processed/full/manifest.json              数据规模、过滤记录和哈希
data/processed/full/targets.json               固定目标 ID
runs/phase1/experiments/clean/                 干净模型和训练记录
runs/phase1/experiments/attacks/               各目标、各攻击的评分 CSV
runs/phase1/experiments/poison/                投毒模型和逐次指标
runs/phase1/experiments/results.csv            本次命令的逐目标逐种子结果
runs/phase1/experiments/summary.json           本次命令的描述性均值/标准差
runs/agent_data/real/target_<id>/              画像、状态、缓存、使用量、评分文件
```

关注 `clean_hr / poison_hr / delta_hr`、有符号和绝对 Prediction Shift、`delta_rmse / delta_mae`，同时核对实际评分条数。HR 是目标物品进入真实用户 Top-K 的比例，不是一般推荐准确率。默认 K=10，使用全物品候选集，屏蔽用户已交互训练商品。

模型按训练身份、配置、模型代码哈希验证后复用；冲突时会报错，不会覆盖。改变模型/训练配置、数据或 Agent 模型时，最清晰的方式是复制配置并为新实验设置新的输出目录。训练中断后当前未完成模型从头重训；Agent 则按用户/轮次 checkpoint 续跑。严禁两个进程同时写同一实验目录；Agent 生成有独占锁，异常关闭留下锁时先确认旧进程已结束，再移除该锁。

## 9. 当前完成与未完成

- 已完成：数据管线、NMF、NeuNMF、六种传统攻击、仅目标评分对照、Agent 评分生成器、可切换 API 接口、缓存/续跑/限额、全物品评测与预算匹配、VS Code 调试入口和测试。
- 已验证：真实压缩数据读取与预处理，小样本上两种模型的训练及 16 个模型-攻击组合；模拟 Agent 数据生成/恢复。详见 `docs/VALIDATION.md`。
- 尚未进行：使用真实 Key 验证 DeepSeek 服务、全量正式攻击比较、生成随机性重复、超参数验证搜索。
- 新增独立扩展：固定评分轨迹上的评论攻击、目标特征传播、评论感知 Dual-Tower Early/Late。
- 仍未实现：评论反向影响选品/评分的联合 Agent、FAP/BayesDetector、工业在线系统。

论文依据与实现差别见 `docs/PROTOCOL.md`。正式实验中不要把这套代码的默认超参数称为作者提供的原始配置。
