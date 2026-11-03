# 实验协议与原文差别

## 数据与划分

用户确认采用 Amazon 2014 5-core Pet Supplies 全时间范围。原文仅称选一个年度，但没有说明数据版本、年份。此项目不承诺逐项复现表格数字。

按 user-item 去重保留时间最晚记录，同时间后出现记录优先；再做迭代 5-core。先随机按 8:1:1 划分，后把验证/测试中 train 未覆盖的用户/物品对应边移回 train，避免无训练特征。记录移回数量和最终比例，因此最终比例并非严格 8:1:1。正式目标从 train 中出现且有商品标题的物品抽取；每个目标独立注入一批假用户。

只由真实 train 计算曝光流行度与基线评分统计，不向 Agent 暴露真实用户历史、真实评分矩阵、验证/测试内容。公开商品信息可以供 Agent 使用；真实评论文本不会参与此阶段的推荐训练或 Agent 输入。

## 模型

- NMF：非负用户/物品因子内积，masked observed-rating MSE 与潜因子 L2，优化后投影回非负。未观测项不当作零分训练；不额外添加偏置。采用 Adam 投影梯度，是工程优化选择，不是宣称完全恢复 Lee-Seung 乘法更新。
- NeuNMF：根据 Dziugaite--Roy 的 NNMF 建立 `MLP([U, V, U' * V'])`，取 K=1。默认 D=10、D'=60、3 层各 50 sigmoid 单元，线性评分输出；额外的乘积特征是所引 NNMF 结构的一部分。使用交替的潜因子/网络参数更新及 RMSprop；为了全量实验使用小批量，原 NNMF 实验使用全批量。最后输出层 bias 初始化为真实 train 均值是新增约定；不等同于 NeuMF/NCF。
- 同一训练种子的真实用户因子、物品因子和网络初始化，在 clean/poison 中相同；新增假用户不会改变原有初始化流。使用真实 train 均值初始化，poison 不重新计算该值。
- 全部训练配置共享，早停只看真实 validation 的原始预测 RMSE；目前没有完成正式超参数搜索。默认 CPU，CUDA 可配置但不同硬件/版本不能保证完全一致。
- 参数 L2 目标为 `MSE + lambda * sum(latent^2) / observed_train_count`。模型代码与训练代码变化后，旧 checkpoint 会被拒绝。

## 攻击

默认 `floor(真实用户数 × 5%)` 假用户，每人对目标最高分。六基线为 Random、Average、Bandwagon、Segmented、Mixed、AoP。Random 用 train 全局高斯统计；Average/AoP 用 train 物品统计，标准差为总体标准差，采样后四舍五入并截断 [1,5]。AoP 从全部物品流行度前 10%（向上取整）排除目标后抽取。热门 selected 数默认 filler 总数的 20%（向下取整，至少 1），属于论文未指定的工程参数。Mixed 按四类循环分配，人数差至多 1；不改变总交互预算。

新增 target_only 对照故意只生成目标交互，不称为预算匹配方法。预算匹配时传统攻击沿用 Agent 的每用户 filler 数量，但不使用其选中物品或评分。

Agent 核心为目标分析→画像推断与验证→分组多样性筛选→目标评分→多轮 filler 选品/评分。默认候选数 2 倍、每组最终选至多 16 个画像，是为了限制上下文的工程近似。独立假用户的不同回合串行，所有请求当前也串行，方便缓存和限额审计。

记忆存评分和商品 ID/轮次，用 Transformer 商品元数据嵌入的余弦相似度取 top-3，再取最近 3 条，按交互索引取并集；不生成评论，不保留评分/评论两套记忆。因此是**评分阶段实现**，不能将结果解释为完整评论型 AgentSA 的验证。默认保留全部短历史。

曝光 `p(i) ∝ train_count(i)^alpha`、alpha=1；每轮最多 20 个不同商品，排除目标与已交互，5 轮，最多每轮选 4 个、人均 filler 上限 20。软参考每轮 2 个，允许不选。这些不是论文报告的完整曝光参数。

`ablation: no_memory` 清空评分请求中的检索记忆，仍保留内部已交互集合以去重。`no_profile` 使用与目标无关的简化通用画像，属于初步附加消融，不等价于作者随机画像的精确实现。改变 ablation 时使用新输出路径。

## API 协议

供应商和模型均由环境变量指定，不固定模型。只实现 chat-completions 请求格式；各供应商的 token 参数、JSON mode、采样限制或认证协议可能不同。网络/429/5xx 有界重试；字段不合法时有界重新请求，不会换成随机攻击。合法 ID/评分/重复交互都在程序侧校验。

模型缓存保存完整请求（不含 Key）、原始模型文本、供应商返回模型名、token 和耗时。Key 仅在 HTTP header 中。固定 request/hash 与 checkpoint 可恢复，但供应商在相同模型别名后升级模型仍可能改变结果；正式实验须记录可用的版本标识。

## 指标

所有真实用户中排除 train 已对目标有交互的用户，构成固定评估群体；候选为全物品并屏蔽各用户 train 已见物品。相同分数按递增物品 ID 打破平局。HR@10 是目标排名不超过 10 的用户比例。默认不截断预测；若开启截断必须对所有方法一致，并知晓它会增加并列。

有符号 Shift=`mean(poison_score-clean_score)`；绝对 Shift=`mean(abs(poison_score-clean_score))`。原文称 absolute change，但表中有负数，因此两种都保留。真实 test 的 RMSE/MAE 在 clean/poison 各计算一次。RMSE 变化小不等于已证明检测隐蔽性。

同一目标数据复用到 3 个训练种子，只覆盖训练随机性；汇总的目标-种子均值/标准差是描述统计，不能冒称独立生成重复或显著性检验。当前只提供一个生成种子配置；评估生成不确定性时应分别改变 Agent/attack 种子及输出目录。

## 参考

- [AgentSA 作者站点 PDF](https://jhliu0807.github.io/paper/AgentSA.pdf)，§4、§5、表 1–4。
- [Neural Network Matrix Factorization](https://arxiv.org/html/1511.06443)，§2、§3、§5。
- [Amazon 2014 官方数据页](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html)。
- [Sentence Transformers 相似度接口](https://www.sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html)。
