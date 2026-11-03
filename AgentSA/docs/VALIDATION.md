# 验证记录

## DeepSeek 适配追加验证（2026-10-30）

新增 `configs/deepseek.yaml`、单次 `api-check` 命令及 DeepSeek 操作说明。完整测试现为 19 项，全部通过；新增 5 项验证默认请求地址/模型/JSON 参数、思考模式与缓存身份、空白输出、截断输出、缺失 Key 时不发送请求。所有 HTTP 调用均使用测试替身。

`doctor` 已确认该配置解析为 DeepSeek、`deepseek-flash`，当前无 API Key。未发起真实模型调用。下方 14 项测试和 16 组合记录为前一次代码交付的原始验证记录。

日期：2026-10-30。范围：第一阶段代码交付检查，不是论文效果复现报告。

## 数据

- 读取用户原始 reviews 和 metadata 压缩包成功，未改写原始文件。
- 全量处理：19,856 个用户，8,510 个物品，157,836 条交互。
- train=126,269，validation=15,783，test=15,784。
- 初始随机划分后，把 validation 的 1 条记录移回 train，使用户/物品均有训练覆盖；test 无需移动。
- 15 个物品缺标题，它们不参与随机目标选择。冻结目标 ID：1920、3127、3933、4070、4202、4569、7249、7519、7977、8147。
- 数据 ID：`58153da5a44f2f886b47641815346e6ceae32b880b51a0e95c9c3291a19bede7`。
- 完整文件 SHA256 与处理参数见 `data/processed/full/manifest.json`。

## 自动检查

`python -m unittest discover -s tests -v`：14 项全部通过。

覆盖：数据集 train/validation/test 不重叠、warm-start 覆盖、哈希防篡改、历史元数据安全读取、六类攻击的数量和评分边界、Segmented 极端评分、目标排序的已见屏蔽与平局处理、固定真实评估人群、曝光排除规则、LLM 非法 ID/重复评分/布尔伪整数拒绝、增加假用户不改变原有初始化、已知低秩数据训练改善、模型保存重载一致、混合记忆检索、HTTP 缓存与限额、Key 不落盘、中断后恢复结果与完整生成一致。

测试中的 HTTP 请求被本地测试替身替换，没有请求外部服务。

## 端到端检查

在用户真实数据中固定抽样 160 个用户，再执行 2-core 得到 94 用户、128 物品、327 交互，仅用于检查程序。

- 生成了 4 个模拟 Agent，每个对目标评分一次、对 filler 评分 4 次，总计 20 条假评分。
- NMF 和 NeuNMF 均完成训练；各评测 target_only、Random、Average、Bandwagon、Segmented、Mixed、AoP、模拟 Agent，共 16 个组合。
- 结果文件 `runs/smoke_checked/mock_validation/results.csv` 恰有 16 行，指标均有限值。
- 除刻意单独设置的 target_only 外，各方法均为 20 条假评分，预算匹配校验通过。
- 所有结果带 `mock_validation=true`，不能作为真实 LLM 攻击效果报告。

## 当前环境

Python 3.12.10，PyTorch 2.5.1+cu118；CUDA 可用，交付检查使用 CPU。NumPy/PyYAML/Requests 等精确版本见 `environment_snapshot.txt`。项目已在 `.venv` 内以 editable 方式安装。

## 尚未验证

- 未选择真实生成模型，未填写任何真实 API Key，未发起真实 LLM 调用。
- 初次交付时 `sentence-transformers` 尚未安装，真实语义编码模型尚未下载；后续依赖修复和本地编码验证见下文。
- 未进行全量正式模型训练/攻击比较、超参数搜索、跨生成种子重复或检测器实验。
- 没有验证特定厂商的请求参数兼容性；接入模型后先用少量 Agent 做真实接口 pilot。
- 当前 API 调用串行；每目标真实生成的粗略估计为 13,951 次请求（不含失败重试），全 10 目标约 139,510 次。默认每目标最多 10,000 次 HTTP 尝试，会先保存状态并停止，需要依据预算决定是否调高。

## 2026-10-30 语义依赖兼容性修复

- 原先 semantic 依赖没有上限，安装得到 sentence-transformers 6.2.0 / transformers 5.22.1。后者停用 PyTorch 2.5.1，并在导入时触发 nn 未定义。
- 已固定并在项目虚拟环境安装 sentence-transformers 3.4.1 / transformers 4.48.3；解析得到 tokenizers 0.21.4 / huggingface-hub 0.36.2。保留 torch 2.5.1+cu118，CUDA 检查为 True。
- 修复后 19 项单元测试通过。
- 使用临时目录中的本地微型 BERT + Pooling 模型，实测项目 ItemEncoder 的非 mock 路径：模型加载、编码、归一化及缓存回读均通过。该检查验证接口和执行路径，不验证预训练模型的语义质量。
- 本次未下载 all-MiniLM-L6-v2 权重，未调用 DeepSeek，未启动付费生成。
