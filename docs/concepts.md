# S1 / S2 概念注解


## 引擎为什么是环境和奖励函数

在强化学习里，智能体需要在一个环境中反复试错。四川麻将规则引擎就是这个环境：它负责发牌、推进回合、判断动作是否合法、结算输赢，并给出每个玩家的分数变化。

分数变化就是奖励函数。胡牌得分、点炮扣分、杠钱、查叫、退杠钱都会影响奖励。如果规则或结算错了，后续 AI 会学习错误目标。

## State / Action / Reward / Episode

- **State（状态）**：当前牌局信息，例如自己手牌、牌河、副露、剩余牌墙、定缺、是否已胡、当前轮到谁。
- **Action（动作）**：玩家能做的选择，例如换三张、定缺、打牌、碰、杠、胡、过。
- **Reward（奖励）**：动作导致的分数变化，例如自摸三家付分、直明杠收 2 分、点炮扣分。
- **Episode（回合/一局）**：从发牌开始，到提前结束或荒庄结算结束的一整局。

## 合法动作生成为什么重要

AI 训练时通常只在 `legal_actions` 中选择动作。如果合法动作生成漏掉胡牌、错误允许有缺门时胡牌、海底仍允许杠，AI 会在错误环境中学习，训练数据也会被污染。

## 零和守恒为什么重要

麻将分数结算应当零和：有人得分就必须有人等额失分。每局结束检查总分为 0，可以快速发现结算 bug，尤其是杠转水、荒庄退杠钱、一炮多响这类复杂场景。

## 不完美信息博弈

麻将不是完全信息游戏。引擎内部需要上帝视角来判定规则，但喂给 AI 的玩家视角只能包含自己手牌和公开信息，不能泄露其他玩家手牌或换三张暗选内容。

因此 S1 的 `state` 需要为后续 S5 预留两种视角：

- **全知视角**：引擎裁判内部使用，所有牌可见。
- **玩家视角**：策略模型使用，只包含该玩家可合法观察的信息。

## S2 为什么要显式区分“没有”和“不知道”

S2 的核心是 `ObservedValue`。同一个字段可能有三种语义：

- `observed`：确定事实，例如自己没有副露、牌墙剩余 0 张。
- `estimated`：估计事实，例如视觉识别认为牌河有 `5W`，置信度 0.8。
- `unknown`：信息缺失，例如中途接入前的换三张记录不知道。

这三者不能混用。空列表、`0`、`false` 是合法事实；`unknown` 才表示模型不应把它当作“没有”。编码器会为 unknown/estimated 写入标志位和置信度，让模型学习“信息质量”本身。

## facts / statistics / beliefs 三层分工

- `facts`：可观察事实与其可观测性状态，只记录能从玩家视角、视觉或人工输入得到的信息。
- `statistics`：可验证的硬编码统计，例如已见牌、剩余牌、未知池拆分；这些必须能用 oracle 对账。
- `beliefs`：对未知牌位置、听牌概率、弃牌危险度的估计；先用数学先验，后续由 S4/S5 学习模型替换。

硬编码统计不能夹带行为学经验分；belief 可以学习，但不能反向污染 facts/statistics。

## sim2real gap 与观测退化

训练时的 S1 引擎天然知道完整状态，但部署时常来自残缺视觉流：可能中途接入、漏识别、误识别或缺少规则字段。`observation_degradation.py` 用退化算子把完美观测变成接近部署分布的观测：

- `MidGameSnapshot`：模拟中途接入，只保留接入后的事件历史。
- `MaskExchange` / `MaskField`：模拟特定字段缺失。
- `VisionNoise`：模拟视觉误识别和漏识别，并把相关字段降级为 `estimated`。

这样训练样本能覆盖“知道得不完整”的情况，避免模型只会处理完美引擎状态。

## belief 标签为什么可以用上帝视角

训练标签可以调用 S1 引擎上帝视角，因为标签是监督信号，不会作为策略输入泄露给模型。约束是：

- policy/belief 输入只允许使用 `S2ProtocolState` 玩家视角。
- `generate_belief_labels(...)` 这类标签函数可以读取 `GameState`，但输出只用于训练目标或离线评估。
- 测试必须验证对手暗手牌仍是 `unknown`，且隐藏牌变化不会改变编码输入。

## 软计数、动作掩码与定长编码

视觉或退化观测会产生软计数：一张置信度 0.8 的公开牌只贡献 0.8，而不是硬塞成 1。剩余牌统计、belief prior 和编码器都应保留这种不确定性。

`legality.py` 是训练/部署同源的合法动作来源。当关键条件未知但动作可能成立时，动作可带 `conditionally_legal`，表示需要后续规则或真实交互确认。

`encoder.py` 把 `s2.v4` 转为固定长度向量。固定 shape 对批量训练很重要；unknown 标志和 confidence 保证“没有/不知道/大概是”在张量里可区分。

## 向听数为什么是 S3 的核心

向听数表示一手牌距离听牌还差几步：`0` 是已经听牌，`1` 是还差一次有效改良。S3 不自己重写向听算法，而是通过 `policies/shanten.py` 复用 S2 `hand_analysis`，保证状态分析、数据生成和后续训练只认同一套牌理定义。

四川麻将还有定缺约束：手里有缺门牌时不能胡，所以缺门牌必须优先清掉。S3 的出牌规则先打缺门，再选择打出后向听数最优的牌；如果向听相同，再用有效进张数、孤张、对子等基础启发式做稳定 tie-break。

## 启发式策略和学习策略的区别

S3 的 `RulePolicy` 是启发式策略：规则由人写死，目标是“会打、不蠢、可复现”。它能提供稳定陪练和可解释数据，但不会从长期收益里自动学到防守、弃和、诱导、针对对手风格等复杂行为。

S4/S5 的学习策略相反：它们从数据或自博弈奖励中学习参数。学习策略的上限更高，但前期需要可靠数据、稳定环境和明确基准。因此 S3 是训练链路的地基，不是最终高手。

## 为什么 S3 是基线而不是终点

S3 明确不实现高级战术：不做安全牌推断、不根据局势弃和、不动态规划大牌方向，也不针对单个对手建模。这些限制避免把人写的偏见硬编码进系统，给 S5 强化学习留下提升空间。

后续评测时，S3 作为固定基线有两个作用：

- 若学习策略连 S3 都长期打不过，说明训练或状态表示有问题。
- 若学习策略只会克制自己、遇到 S3/贪心/随机就退化，说明对手分布太窄。

## 对手池为什么要有强度梯度

`policies/opponent_pool.py` 把三档标准对手封装为统一 `BasePolicy` 接口：

- `random`：弱对手，随机选合法动作，用于压力测试合法性和鲁棒性。
- `greedy`：中等对手，优先胡牌、清缺门和直接推进，不做完整 S3 牌理。
- `s3_rule`：基线对手，使用 S3 规则策略，是 S4/S5 的主要固定参照。

多样化对手池可以缓解“只跟自己打导致只会克制自己”的分布差。S5 会继续加入历史版本快照；S3 交付的三档对手提供最早的强度梯度。

## S3 如何为 S4 提供数据

`selfplay/data_recorder.py` 在每个决策点记录玩家视角的 `s2.v4` 协议状态、合法动作、实际动作、终局分数和单独的上帝视角标签。输入和标签必须分离：策略输入不能包含对手暗牌，belief 标签可以用上帝视角批改。

S4 可以用这些数据做两件事：训练 belief 网络预测未知牌与听牌风险；训练模仿策略先学会 S3 的基本打法。之后 S5 再从这个冷启动策略出发，通过自博弈继续提升。

## 监督学习和模仿学习

监督学习是“给输入，也给标准答案”的训练方式。模型看到很多样本，例如某个玩家视角状态、合法动作集合、老师实际选择的动作，然后学习把相似输入映射到相似答案。

模仿学习是监督学习在决策任务里的常见形式：老师可以是人类牌谱、S3 规则策略或更强模型。S4 的策略网络把 S3 自博弈记录当作老师，目标不是立刻超过老师，而是先学会稳定地产生合法、接近老师的动作。

模仿学习的天花板通常受老师限制：如果 S3 不会防守、不懂弃和，学生只看 S3 数据也很难凭空学会这些高级战术。但 S4 仍然必要，因为它提供三件基础设施：

- **冷启动策略**：S5 强化学习不必从随机出牌开始探索，而是从“已经会基本打法”的网络继续提升。
- **训练评估链路**：数据集、动作掩码、checkpoint、arena、评估报告都在 S4 打通，S5 可以复用。
- **belief 能力**：部署时 AI 看到的是残缺玩家视角，S4 先训练读牌/补全能力，降低 S5 的状态不确定性压力。

## belief 网络和策略网络各自解决什么问题

belief 网络回答“我不知道的信息可能是什么”。例如中途接入时不知道早期弃牌，或者视觉识别置信度不足，belief 网络要根据公开牌、自己手牌、定缺、副露、牌河等信息，估计未知牌在哪里、对手是否可能听牌。

策略网络回答“现在我应该做什么”。它接收玩家视角编码、合法动作掩码，以及必要时接收 belief 输出，最后在合法动作里选择换三张、定缺、打牌、碰、杠、胡或过。

二者分开训练和验收，是为了避免问题混在一起：

- belief 错，说明读牌/状态估计有问题。
- 策略错，说明动作选择或老师模仿有问题。
- 如果合在一起只看输赢，很难判断到底是“看错局面”还是“看对了但决策差”。

## 为什么 belief 用“退化观测输入 + 精确真值标签”

训练 belief 时，输入故意使用退化后的玩家视角，例如字段 unknown、estimated、置信度不足或中途缺历史；标签则来自引擎上帝视角的精确真值。这个配方类似老师批改盲填题：学生只能看残缺题面，但答案必须按完整事实评分。

这样做不会泄露信息，因为真值只用于损失函数和离线评估，不进入模型输入。长期训练后，模型学到的是从残缺线索中推断隐藏状态的统计规律，而不是在运行时偷看对手暗牌。

## 校准是什么，为什么概率必须可信

校准（calibration）衡量模型说出的概率和真实频率是否一致。若模型在很多样本上都说“70% 可能”，这些事件最终大约 70% 发生，才叫校准好。

辅助模型的概率是给决策和人看的，所以不能只追求排序正确。如果系统提示“某家 70% 听牌”，真实却只有 30%，玩家会过度防守；如果真实是 90%，又会低估风险。S4 因此要报告 ECE 等校准指标，必要时用温度缩放等方法让概率更可信。

## 为什么 train / val / test 要按局划分

麻将一局内部的样本高度相关：同一副牌、同一批玩家、连续决策点共享大量上下文。如果把同一局的前半放进训练集、后半放进验证集，模型可能只是记住了这局的局面分布，验证分数会虚高。

按局划分可以避免局间泄漏：一整局只能属于 train、val 或 test 之一。

- **train**：用于更新模型参数。
- **val**：用于调参、选择 checkpoint、早停。
- **test**：只在最后报告泛化效果，不参与调参。

早停（early stopping）是在验证集指标长期不提升时停止训练，防止模型继续记训练集细节却损害泛化。

## S5 从哪里接手

S4 结束时，S5 接手的不是最终高手，而是一套可训练、可评估、可复现的起点：

- 会从残缺观测中输出可校准估计的 `LearnedBelief`。
- 会稳定选择合法动作、行为接近 S3 的模仿策略网络。
- 能对比随机、贪心、S3、策略网络的 arena。
- 能在完美观测和退化观测两条赛道上报告一致率、非法动作、得分和鲁棒性。

S5 在此基础上用自博弈和奖励信号继续优化长期收益，重点学习 S3 没有手写进去的高级能力：防守、弃和、风险控制、对手适应和局势权衡。

## S5：从 S4 冷启动到自博弈强化学习

S5 不是再次模仿 S3。它以冻结的 S4 Policy 作为初始策略，以冻结的 S4 Belief 作为不完全信息的读牌能力，然后在 S1 引擎的完整结算中自博弈。因此，策略可以为了长期净得分而学习防守、风险控制和大牌取舍，而不会被 S3 的规则上限锁死。

### 自博弈、终局奖励与 GAE

每一局只有一个学习者座位，其余三个座位从对手池中抽样。rollout 仅记录学习者在每个决策点的特征、合法动作掩码、动作、旧策略 log-prob 和价值。中间奖励恒为 0；只有一局完全结算后，最后一步才获得经归一化的终局净得分。胡牌不是 episode 终点，仍要等待材钱、转水、查叫等所有结算完成。

这种设计避免人为中间奖励让策略“刷分”（reward hacking）。价值网络估计从当前决策点到终局的期望得分；GAE（Generalized Advantage Estimation）用这个估计将终局信号分配到轨迹各步，在方差与偏差之间取稳定折中。

### PPO 与合法动作边界

PPO 以旧策略的轨迹为样本，限制新策略每次更新的幅度（clip），同时优化策略损失、价值损失和熵。训练早期还对冻结 S4 Policy 施加逐渐衰减的 KL 约束，防止冷启动策略突然退化成随机出牌。所有 logits 都先由 S1/S2 同源的 legal mask 过滤：被掩码的动作既不能被采样，也不能参与 log-prob 、KL 或 PPO 更新。

### League 与观测退化课程

只与最新的自己对战会造成循环克制。league 混合最新策略、可保留的历史快照、S3 规则对手、贪心对手和随机对手。新快照入池前要对池中成员做反遗忘评测；不能用“打赢最新自己”代替这个门槛。

观测退化课程从完美观测或轻微噪声开始，再逐步增加中途接入、缺失字段和视觉误识别的比例。只有学习者的 S2 观测受退化；对手保持完美观测。这既模拟部署条件，也避免通过同时弱化对手来虚高胜率。

### 不完全信息安全与冻结 S4 依赖

学习者的路径固定为“退化后的 S2 观测 → 冻结 S4 Belief → 编码 → 策略”。它不接触 `GameState`、其他玩家暗手牌、牌墙顺序、真值 label 或环境洗牌 seed。S4 Belief 在 S5 的第一阶段强制冻结：RL 不能反向改写已经校准的读牌能力。如要尝试联合微调，必须作为单独对照实验，并重跑全部 S4 Belief 验收。

四人不完全信息博弈没有已知的理论最优解保证。工程上用多样对手、安全的观测边界和不同难度的课程来缩小分布偏差，而不将训练集胜率误认为真实牌力。

### 健康监控、双赛道与操作流程

长时间 RL 可以“不报错地失败”。训练会监控熵塌缩、KL 爆炸、价值损失发散、对 S3 胜率回退和非法动作。任一警报都要保存可诊断 checkpoint，包含模型、优化器、RNG、league、课程、KL horizon 和训练曲线，以便可重现地继训或回滚。

每个里程碑都跑两条 arena 赛道：完美观测赛道测理想牌力，残缺观测赛道测部署可用性。两者都与 S3 和 S4 模仿策略做同条件对照，报告平均净得分、胜率、非法动作率及置信区间。完美赛道好、残缺赛道差时，不能视为完成。

操作时先在 CPU 上运行可重现的小规模 smoke，验证合法性、零和结算、checkpoint 继训和双赛道报告；再打包上云并做 GPU smoke。正式长训前固定 S4 产物版本、对手池与课程配置；训练中保留检查点和评测报告；训练后输出高手行为审阅材料，供人工依照八条清单打分。





## S5 Expert Review Export Boundary

Expert review is post-training human diagnosis only: it is not a PPO reward, policy input, or backpropagation signal. A review item starts from the learner-visible `LearnerView` and its retained `TrajectoryStep`; the exporter never accepts `GameState`, a generic mapping, or an arbitrary JSON payload.

The fixed review categories are: defensive exchange direction, disadvantage-opening escape, post-kong discard safety, seven-pairs value, endgame defensive fold, early-tenpai self-draw, dead-tile inference, and favorable-position big-hand choice. A source must explicitly select one `ExpertBehaviorCategory`; the exporter never infers a category from privileged state.

`ExpertReviewSource` accepts only a fixed category, `LearnerView`, retained trajectory, and canonical S3/S4 action-space indices. It accepts no review ID, prose, labels, summaries, or mappings. `generate_expert_review_records()` returns fresh plain JSON-safe mappings (not an exportable record object); it assigns deterministic `review-000001`-style IDs only from source order. The JSON export has exactly `record_id`, `category`, `public_observation`, `policy_action`, `s3_action`, `s4_action`, and `belief_summary`. `public_observation` is derived only from an observed safe phase and canonical public river tiles. Actions are derived with `state.action_space.index_to_action`, so each has only `index` and canonical `action`; caller labels cannot enter the export. `belief_summary` is derived only from structurally valid frozen learned-Belief output: a learned source, estimated tile-location/tenpai/danger outputs with finite bounded probabilities, canonical S4 tile/player schemas, and no oracle-shaped fields.

The exporter rejects non-learned or malformed Belief data, unobserved/unsafe public fields, non-canonical tiles, and invalid indices. Use `select_review_sources()` to sample one category, then `generate_expert_review_records()` for reviewer JSON. Human findings are training/state diagnostics only; never write them directly into S3 rules or a shaped per-step reward.
