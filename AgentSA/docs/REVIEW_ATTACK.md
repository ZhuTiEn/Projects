# 评论攻击与目标特征传播

该模块是论文思想下的独立工程实现，不是作者官方 Dual-Tower 代码。它使用固定的评分 Agent 轨迹，分别生成普通评论与目标特征传播评论，从而隔离评论文本的净作用。

## 因果比较

- `rating_only`：注入完全相同的假评分，不注入假评论。
- `plain`：注入画像、评分和商品一致的普通评论。
- `tfp`：保留普通评论作为 `initial_review`，只把 filler 评论替换为传播版本。

三组共享同一 `fake_ratings.csv` 哈希。TFP 不允许直接写出目标商品名称或 ASIN；共享特征必须同时给出目标和 filler 的元数据证据。无合理重叠时应保留初始评论。

## 运行顺序

当前正式评分生成必须先完成，评论阶段不会改变或覆盖其缓存：

```powershell
.venv\Scripts\python.exe -m agentsa --config configs/review_target_aware.yaml prepare-reviews
.venv\Scripts\python.exe -m agentsa --config configs/review_target_aware.yaml estimate-reviews
.venv\Scripts\python.exe -m agentsa --config configs/review_target_aware.yaml generate-reviews --target 1920
.venv\Scripts\python.exe -m agentsa --config configs/review_target_aware.yaml run-reviews --target-count 1 --seeds 11
```

离线 Mock 检查必须使用 `configs/review_smoke.yaml` 和已经用相同配置生成的 Mock 评分 Agent。Mock 结果只证明数据、缓存、模型和评测接线，不代表攻击有效。

## 数据与模型边界

评论 sidecar 严格复用既有用户/商品 ID、去重结果、训练/验证/测试划分。评论模型只用训练集评论构造用户和商品文本表示，验证/测试评论不会进入表示。

`dual_tower_early` 在塔内融合 ID 与评论聚合向量；`dual_tower_late` 在输出端融合 ID 得分和评论语义得分。真实模式默认使用冻结的 `all-MiniLM-L6-v2`，Mock 使用确定性哈希向量。

第一版属于固定评分轨迹上的 post-hoc 评论实验：评论不会反过来改变已经生成的选品与评分。若该阶段显示出稳定增益，再另建输出目录实现评分、评论和记忆联合生成。

每个假用户需要一次目标评论调用，并在每轮分别调用一次普通评论批处理和一次传播批处理。992 个假用户、5 轮时，忽略重试约为 10,913 次请求/目标；它使用独立用量账本，不与评分阶段的请求计数相加。

## 三用户真实 API pilot

`configs/review_pilot_3users.yaml` 使用独立输出目录，将假用户限制为 3 人、2 轮、每轮 2 个 filler。评分阶段预计 26 次请求，评论阶段预计 16 次，合计约 42 次，不含重试。它不会读取或覆盖正式 992 用户的请求缓存。

```powershell
.venv\Scripts\python.exe -m agentsa --config configs/review_pilot_3users.yaml estimate
.venv\Scripts\python.exe -m agentsa --config configs/review_pilot_3users.yaml estimate-reviews
.venv\Scripts\python.exe -m agentsa --config configs/review_pilot_3users.yaml generate-agent --target 1920
.venv\Scripts\python.exe -m agentsa --config configs/review_pilot_3users.yaml generate-reviews --target 1920
```
