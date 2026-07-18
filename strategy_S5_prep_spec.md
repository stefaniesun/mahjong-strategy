# S5 开训前准备规格:资产入库 + belief 分桶补考 + RL 基建适配

> 交给 AI 编码助手执行的独立文档。三个任务相互独立,按顺序执行(先保资产,再补考试,最后适配基建)。
> **本文档不包含 S5 正式训练**——只做开训前的准备与验证,全部在本地完成,不需要云。

---

## 背景(执行前必读)

S4 已收官:模仿策略在 500 局同种子门禁上达到 **-0.066±0.42 分/局**(基线 S3 为 -0.044,统计上不可区分),五代迭代史:

| 版本 | 配方 | 门禁分差 |
|---|---|---|
| v1 | 1000 局 | -2.56 |
| v2 | 5 万局 + encoder.v3(对手弃牌/副露) | -2.47 |
| v3 | + DAgger① | -1.30 |
| v4 | + DAgger② | -1.26 |
| **v5** | **+ encoder.v4(候选弃牌特征)** | **-0.07 ✅ 通过** |

**S5 种子三件套(最终定稿)**:
- 编码器:`ENCODER_VERSION = "s2.v4.encoder.v4"`,893 维;
- checkpoint:`training_artifacts/S4/v5_20260718_encoder_v4/checkpoints/{belief_s4.pt, policy_s4.pt}`;
- 门禁复现命令:`learning/eval/arena.py`,seed=90000,500 局,0 号位模型 + 3 个 RulePolicy。

---

## 任务 1:资产入库(git,先做——保护这几天的成果)

当前未提交/未跟踪的资产(以 `git status` 实际输出为准):

1. **代码**:`state/hand_analysis.py`(`_best_blocks` 缓存上限修复——这是修过线上事故的,有注释说明)、`tools/cloud_train_s4_50k_cached.py`(缓存并行训练器 + 多数据集支持)、`tools/generate_dagger_data.py`(DAgger 数据生成器);
2. **产物**:`training_artifacts/S4/v2_*/ v3_*/ v4_*/ v5_*/`(每版 checkpoint 约 6MB + 报告,合计约 25MB,可入库);
3. **文档**:`strategy_S2_encoder_upgrade_spec.md`、`strategy_S2_encoder_v4_candidate_features_spec.md`、`strategy_local_training_infra_spec.md`、本文档。

要求:
- 若无 `.gitignore`,先补(至少排除:`__pycache__/`、`*.out`、`.pytest_cache/`、数据分片目录如 `data_dagger*/`、`cloud_outputs/*/data/`);
- 分主题提交(代码修复 / 工具 / 产物归档 / 文档),提交信息说清动机,不要一锅炖;
- **新建 `docs/s4_gate_history.md`**:记录上面五代门禁表、每代的诊断结论(数据量 → 分布漂移 → 表示能力)与门禁复现命令——这是项目最重要的实验记录,必须落档;
- 提交后 push 到 origin。

**验收**:`git status` 干净(除有意忽略项);`git log` 主题清晰;GitHub 上能看到全部资产。

## 任务 2:belief 分桶补考(S4 spec 欠下的考试③④)

S4 spec 要求的四场 belief 考试,①(赢先验)②(校准)已过,**③(尾盘提升最大)④(退化画像分桶)从未跑过**。现在用 v5 的 belief checkpoint 补上。

**做什么**:写 `tools/eval_belief_buckets.py`:

1. 数据:从本地 5 万局数据包(`s4_50k_cloud_training_package_*.zip` 解出的分片;若本地没有,按 `tools/` 下生成器的种子区间重新生成若干局即可)抽取验证集样本 ≥2 万条(按 game_id 哈希取 val 划分,与训练侧 `_split_name` 同口径);
2. **阶段分桶**:按记录中牌墙剩余数分三桶——开局(>40 张)、中盘(20~40)、尾盘(<20);
3. **退化画像分桶**:perfect / light_noise / midgame / heavy 四桶(用训练同款 `DatasetBuildConfig` 施加);
4. 每桶计算:belief 模型 tile log-loss、先验 tile log-loss、提升幅度(prior − model);
5. 产出 markdown 报告到 `training_artifacts/S4/v5_20260718_encoder_v4/reports/belief_bucket_report.md`。

**验收判定**:
- 考试③:**尾盘桶的提升幅度 > 开局桶**(尾盘信息多,可推断空间大;若尾盘反而不如开局甚至不如先验,按 S4 spec 属"照妖镜"级异常,停下报告人类,疑标签错位);
- 考试④:**四个退化画像桶全部满足 model log-loss < prior log-loss**;
- 报告数字齐全、可复现(固定种子)。

## 任务 3:S5 RL 基建适配 encoder.v4 + v5 三件套冒烟

`rl/` 目录的 S5 基建(rollout / ppo_trainer / league / curriculum / reward / value_net / checkpoints / train_rl)编写于 encoder.v3 时代,需适配 893 维并用 v5 三件套跑通冒烟。**只做适配与冒烟,不启动正式训练。**

要求:

1. **维度全面动态化**:排查 rl/ 与 learning/ 中一切 input_size 来源——必须从 `encode_state`/编码结构表动态推导,不得有 806/263 等硬编码残留(全局 grep 数字确认);value_net 的输入维度同样动态;
2. **v5 三件套装载**:
   - 策略网络:从 `v5_20260718_encoder_v4/checkpoints/policy_s4.pt` 初始化 PPO 的策略头(权重加载,encoder_version 校验必须通过);
   - belief:同目录 `belief_s4.pt` 以 `LearnedBelief` 冻结装载(S5 全程不训它);
   - **KL 锚**:确认 ppo_trainer 的 KL 正则引用的参考策略就是 v5 模仿策略(冻结副本),且系数可按训练进度衰减——这是 S5 spec 的硬要求;
3. **冒烟(全部本地)**:
   - rollout:1 学习者(v5 策略)+ 3 对手池成员跑 ≥50 局,轨迹 schema 校验、零非法动作、零和守恒;
   - PPO:用冒烟轨迹跑 ≥10 个更新步,loss 有限、无 NaN、熵/KL 数值合理;
   - checkpoint:存档 → 读档 → 继续 5 步,曲线衔接(断点续训验证);
   - league:v5 入池、采样权重生效;
   - 观测退化:学习者座位按 curriculum 施加退化画像,对手保持完美观测(隔离正确性断言);
4. **性能台账**:实测本地 rollout 吞吐(局/分钟,注明机器与线程数),写进 `docs/` 下台账文件——这个数字决定 S5 正式训练放本地还是云,是人类决策的依据;
5. 全套 pytest(含既有 test_s5_*)通过;新增的适配相关断言并入测试。

**验收**:上述 5 项全绿;给出一条"正式启动 S5 训练"的完整命令(参数就绪但**不执行**),连同预计单轮时长写在总结里交人类拍板。

---

## 全局约束

- **三不动**:不改 encoder(v4 已定稿)、不重训 S4(v5 已过门禁)、不启动 S5 长训(等人类拍板);
- 可见性/零非法动作红线照旧;所有新脚本固定种子可复现;
- 任务 1 的 git 操作:只 add 明确列出的类别,提交前 `git status` 核对,不误提数据分片等大文件。

## 执行顺序

任务 1(入库保资产)→ 任务 2(分桶补考,约半小时)→ 任务 3(RL 适配冒烟,主体工作)。

## 完成标志

**git 资产全部入库且 push;belief 补考③④出报告且通过(或如实报告异常);S5 基建在 893 维 + v5 三件套下全链路冒烟通过、断点续训验证、rollout 吞吐实测在案;"启动 S5"的命令与耗时预估摆在人类面前待批。**
