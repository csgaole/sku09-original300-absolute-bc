# 第一阶段 BC：RL-100 源码对照与运行协议

## 来源锁定

- 用户提供论文 `2510.14830v4.pdf`，Imitation Learning，式 (1)–(6)：两帧条件、条件扩散噪声预测、编码器端到端训练、适用时加入 Recon/VIB。
- 官方仓库 https://github.com/Lei-Kun/RL-100 ，锁定 `6b6afceb7eb12525398207b5d6e66908863dc385`。启动前核对远程 HEAD 与该 revision 一致。
- 采用官方 `scripts/Diffusion/Offline/2D/train_policy_image_unet_chunk_two_stage.sh` 的 BC 参数覆盖值，并结合 `rl100_2d_epsilon.yaml` 的继承项。不是混用基础 YAML 的 1600 epoch / batch128 / lr1e-4 与启动脚本的覆盖值。
- 此次只运行 BC，不启动该脚本中随后进行的 IQL、动力学、PPO 或蒸馏。

## 保留的官方实现与参数

- `ConditionalUnet1D`，down_dims=[256,512,1024]，FiLM，diffusion embedding256，kernel5，GroupNorm8。
- `MultiImageObsEncoder`：每相机独立 R3M ResNet18，GroupNorm，ImageNet 标准化，resize224，无裁剪；全部视觉参数参与训练。
- R3M 权重来自 facebookresearch/r3m 官方下载地址；加载完整 checkpoint 中 `module.convnet.*`，严格核对键名，不使用随机初始化代替预训练权重。
- 官方 `RL1002D.compute_loss` 与 `_augment_rgb_obs` 两个方法未经修改，直接从锁定源文件的 AST 编译绑定。它们调用官方 encoder、UNet、normalizer、mask generator 和 scheduler。这样避免 RGB BC 因顶层无条件导入而依赖无关的 pytorch3d/RL 环境，不伪造缺失模块。
- epsilon prediction；100 个训练噪声时刻；squaredcos_cap_v2；DDIM 配置 steps_offset=1，推理目标10步（本轮训练不做蒸馏）。
- RandomShiftsAug pad4；Recon 权重0.05；VIB beta=5e-4，KL annealing关闭。损失遵从官方实现（RGB重建与VIB），不自行增加奖励或价值项。
- AdamW lr=2e-4，betas=(0.95,0.999)，eps=1e-8，weight_decay=1e-6；cosine，warmup500个优化器更新。
- 官方 EMA：inv_gamma1、power0.75、最大衰减0.9999。BC 上游训练循环没有执行梯度裁剪，本适配器也不以配置中未使用的 max_grad_norm 改写优化。
- 600 epoch、有效 batch512、seed42。14400 个训练动作块/epoch，共29个更新/epoch（最后64个样本），总计17400次更新。

## 明确的任务与运行适配

1. 三相机替代官方示例任务相机；raw320×240输入，官方 encoder resize224。输入只有图像和28维本体状态，不含特权物体状态、力、专家相位、控制参数，也不添加时钟输入。
2. n_obs_steps=2、n_action_steps=8、horizon=9。适配器提供一个结构性前缀，官方 `no_pre_action` 丢弃它，最终损失严格覆盖当前时刻起的8个真实未来动作；不跨回合、不错位。
3. 输出14维绝对关节位置目标，含固定12mm夹爪列；拟合实际生效动作。不是末端 delta pose；论文仅在适用时使用末端增量。
4. 现有数据是仿真特权脚本搜索生成的示教，非论文的人类遥操作数据；不是原论文任务/结果的复现。
5. HDF5 回合适配器替代官方任务数据加载器。只使用240条训练、30条验证，30条测试不读取。归一化只用训练回合，采用官方 limits normalizer。
6. 单张 RTX5090，FP32、关闭TF32。microbatch8，通过按样本数正确加权的梯度累积得到有效 batch512；不是将优化器 batch 改小。GroupNorm无跨batch统计，但随机增强/浮点求和顺序与原生大batch不逐位相同。
7. 使用独立训练循环承载官方组件和损失，而不是直接运行官方完整训练入口。保存完整 optimizer/scheduler/RNG/EMA 状态；每个epoch保存 last，按完整验证集EMA总BC损失保存 best，每50epoch保存当时最优权重。
8. 验证关闭随机图像增强，EMA处于eval模式，固定噪声种子919；验证不推进训练RNG。最终成功率需要冻结模型后的独立闭环仿真测试；BC loss不是任务成功率。

## 执行与断点恢复

工作目录为本仓库，Python 环境为原工作区 `work/il_venv`。安装的训练依赖见 `requirements-bc.txt`。

```bash
python src/train_bc.py --microbatch 8
python src/train_bc.py --microbatch 8 --resume
```

启动前进行一次真实完整batch的前向/反向/优化器/EMA检查，并保存首步checkpoint；正式进程从该断点继续，不将预检作为已完成训练报告。

运行文件位于 `runs/bc_stage1_seed42/`：

- `config.json`：冻结配置、参数量、预训练权重哈希。
- `status.json`：PID、阶段、epoch、更新步数与速度；异常写入失败原因。
- `metrics.jsonl`、`validation.jsonl`：训练与完整验证集损失。
- `last.pt`：可恢复完整训练状态；`best.pt`：按验证损失选定的EMA策略。
- `training.log`：后台进程输出。

600 epoch来自选定官方启动配方，不代表论文对所有新任务统一规定的收敛阈值。不会未经说明缩减网络或预算，也不会将以上明确适配隐去后称作原论文严格结果复现。

## 用户授权修订：开发集阶段评估与提前停止（development_v2）

此段覆盖前文固定跑满600轮与仅按验证损失最终选模的规定。第100轮评估，此后每50轮一次，每次固定100个新开发种子880000–880099。候选为截至该阶段的验证损失最优EMA。成功数相对改善参考增加至少2/100，或验证总损失比改善参考下降至少1%，任一成立清零耐心计数；参考仅在该指标满足阈值时更新。首轮设基线，连续3次两项均未达到阈值则提前停止（最早250轮），最多600轮。这些是工程阈值，不是统计显著性判据。最终按开发成功数最高选择候选，同分取较低验证损失，再同分取较早阶段。最终冻结 selected_development.pt 后才运行900000–900999独立测试。保留原600轮cosine调度，不重启学习率。工程接线检查使用870000系列，与开发及测试隔离；不用于选模。开发评估在训练进程内同步触发，评估期间不继续训练，故无需等待自动跟进才触发。全部阶段候选、报告和停止决策均持久化；崩溃后可从阶段断点继续。
