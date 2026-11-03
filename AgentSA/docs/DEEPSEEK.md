# 使用 DeepSeek 生成评分型 Agent 攻击

核对日期：2026-10-30。模型请求使用 DeepSeek 官方 Chat Completions 协议，不需要安装额外的厂商 SDK。

## 起始配置

`configs/deepseek.yaml` 继承第一阶段的数据与推荐模型设置：

- 服务地址：`https://api.deepseek.com`。
- 起始模型：`deepseek-flash`，可以通过 `DEEPSEEK_MODEL` 环境变量或 YAML 中的 `llm.model` 切换。
- `thinking: disabled`，显式使用非思考模式作为起点；这是实验选择，不代表已比较它与思考模式的效果。
- `json_mode: true`，使用 JSON Output。
- 每次最多 4,096 输出 tokens，120 秒读取超时；原有限额与断点续跑机制保留。
- Agent 数据写入 `runs/agent_data_deepseek`，推荐实验写入 `runs/phase1_deepseek`，与通用配置结果分开。

当前官方入门文档列出的模型名包括 `deepseek-flash`、`deepseek-v4-pro`。不要照搬旧教程中的历史别名，具体可用模型以账户和当时官方文档为准。

## VS Code 终端操作

先进入项目，再在同一个 PowerShell 终端设置 Key：

```powershell
Set-Location C:\Projects\AgentSA
$secret = Read-Host "输入 DeepSeek API Key" -AsSecureString
$env:DEEPSEEK_API_KEY = [System.Net.NetworkCredential]::new('', $secret).Password
.venv\Scripts\python.exe -m agentsa --config configs/deepseek.yaml doctor
```

输入隐藏显示，避免把 Key 明文留在命令历史。环境变量只对本终端及其子进程有效；从此终端执行后续命令。`doctor` 不连接模型服务，也不显示 Key。

`.env.example` 只是配置示例，不会自动加载。VS Code 的 F5 调试进程也不会自动继承另一个终端新设置的环境变量；当前建议直接使用已设置 Key 的终端。

单独检查一次接口：

```powershell
.venv\Scripts\python.exe -m agentsa --config configs/deepseek.yaml api-check
```

这条命令会向真实服务发送很短的 JSON 请求，可能产生费用。它不读取数据集、不加载语义编码器、不生成假用户。每次检查都发送新请求，以便验证当前连接和凭据；网络故障时仍受配置的重试和预算约束。检查记录在 `runs/agent_data_deepseek/api_checks/`，不会记录 Key。

生成攻击数据还需要安装语义编码器依赖：

```powershell
.venv\Scripts\python.exe -m pip install -e ".[semantic]"
.venv\Scripts\python.exe -m agentsa --config configs/deepseek.yaml estimate
```

先检查估算成本；全量默认 992 个假用户/目标，不建议把它当成接口试运行。可复制 YAML，降低 `attack.fake_ratio`、`agent.rounds` 并设置新的 `agent.output` 与 `experiment.output`，做小规模 pilot。

项目的 semantic 依赖固定为 sentence-transformers 3.4.1、transformers 4.48.3，兼容当前 PyTorch 2.5.1。若旧安装提示 PyTorch >= 2.6 或出现 `NameError: name 'nn' is not defined`，在项目目录重新执行上面的安装命令，然后重试。无需因此升级现有 CUDA/PyTorch。

确认配置后，生成第一个固定目标并进行匹配预算比较：

```powershell
.venv\Scripts\python.exe -m agentsa --config configs/deepseek.yaml generate-agent
.venv\Scripts\python.exe -m agentsa --config configs/deepseek.yaml run --target-count 1 --include-agent --match-agent-budget
```

默认只生成第一个目标，不会自动生成全部 10 个目标。按原来参数粗估每目标约 1.4 万次请求，但默认 HTTP 尝试上限为 1 万，会保存状态并停止；根据实际预算决定是否调高再续跑。

## 模型与模式切换

```powershell
$env:DEEPSEEK_MODEL = "deepseek-v4-pro"
```

环境变量优先于 YAML；不需要同时设置通用的 `AGENTSA_*` 变量。如果修改模型、思考模式、温度或输出上限，需要设置新的 Agent 输出目录，避免把两种设置下的画像/交互混合。

思考模式用 YAML 配置 `thinking: enabled`，可再设置 `reasoning_effort: low/high/max`；代码会省略在该模式下不起作用的 temperature 参数，并将模式纳入缓存身份。思考模式需要另外规划输出 token 上限，当前 4,096 是非思考模式的起始配置。

代码只解析最终 `content` 为实验 JSON，不把 `reasoning_content` 当作评分。空白 content 会被拒绝并交给已有的有界结构校验重试；`finish_reason: length` 会被明确识别为截断，不作为成功评分使用。原始最终文本、结束原因、供应商返回模型标识及 usage 字段会保存，便于核对 token 和缓存命中统计。

## 已验证与未验证

已用模拟 HTTP 响应检查：Bearer 认证、默认地址/模型、环境变量覆盖、JSON 参数、思考模式参数、缓存身份、空白/截断处理、缺失 Key 时不发送请求，以及原有单元测试。

尚未使用真实 Key 请求 DeepSeek，也未启动付费攻击生成。模型服务的实际可用性、额度与当前账户权限，需要由 `api-check` 和小规模 pilot 验证。

官方资料：

- [首次调用](https://api-docs.deepseek.com/en/)
- [JSON Output](https://api-docs.deepseek.com/guides/json_mode/)
- [思考模式](https://api-docs.deepseek.com/guides/thinking_mode/)
