# M20 Pro VLA 具身智能仿真项目记录

Last updated: 2026-07-24 17:30 CST

## 项目边界

- 本项目只做仿真，不做 sim2real。
- 工作区：`/home/fabu/桌面/M20Pro/m20pro_sim_ros2_ws`。
- 真机工作区：`m20pro_real_ros2_ws`，不在本项目中修改、部署或控制。
- 目标是语言条件下的开放词汇导航：指定地点、指定物体、自主探索，并在需要时跨越 1 m 障碍。
- 长期方向是 VLA（vision-language-action）策略，不把 Nav2/A*/AMCL 作为主要决策器。

## 训练存储

- 系统盘为 `/dev/sda3`，可用约 60 GB；Isaac Sim/环境本体仍留在系统盘。
- 2 TB 硬盘为 `/dev/sdb2`，UUID `b9cbb43d-5119-4328-99d9-10f7c0d91e37`，当前挂载于 `/media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37`，剩余约 715 GB。
- 训练输出根目录：`/media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA`。工作区中 `logs` 和 `videos` 是指向该目录的软链接，旧训练产物已迁移，没有再占用系统盘。
- 重启后先运行 `./scripts/prepare_output_storage.sh`；如未挂载，运行 `udisksctl mount -b /dev/disk/by-uuid/b9cbb43d-5119-4328-99d9-10f7c0d91e37`。

## 技术基线

- Conda：`m20pro-vla`
- Python 3.11.15
- PyTorch 2.7.0+cu128
- Isaac Sim 5.1.0
- Isaac Lab v2.3.2 / package 0.54.2
- RSL-RL 3.1.2
- GPU：RTX 3060 12 GB，compute capability 8.6
- Python 运行环境确认：Isaac Sim、Isaac Lab、PyTorch 和 RSL-RL 均运行在 Conda 环境 `m20pro-vla`，解析器为 `/home/fabu/miniconda3/envs/m20pro-vla/bin/python`。
- NVIDIA EULA 已由用户阅读并接受；激活脚本使用 `OMNI_KIT_ACCEPT_EULA=Y`

## 已完成

### 2026-07-20 仿真环境

- 完成独立 Conda 环境、Isaac Sim/Lab 和 CUDA 验证。
- 完成 `scripts/install_vla_env.sh`、`activate_vla_env.sh` 和 `check_vla_env.sh`。
- 修正 EULA 变量，Isaac Sim headless 启动到 `[INFO]: Setup complete...`。

### 2026-07-20 M20 Pro USD 资产

- 使用 `src/m20pro_description/urdf/M20.urdf` 转换 USD。
- 结构为 17 个刚体、16 个关节：12 个腿关节、4 个轮关节。
- 所有 STL 视觉网格和 URDF 中的简化碰撞体均已找到。
- [m20pro_cfg.py](assets/m20pro/m20pro_cfg.py) 已配置 DC motor 腿执行器和轮速度执行器。
- 240 个物理步测试通过：根高度稳定在约 0.590 m，无 NaN。

### 2026-07-20 第一个 RL 任务

- 新增 `tasks/m20pro_locomotion/`。
- 平地 +X 移动基线：16 维动作、60 维本体观测、1024 并行环境、20 s episode。
- [check_m20pro_task.py](scripts/check_m20pro_task.py) 已通过 headless 配置校验。
- 用户实际运行后复核输出为 `configuration valid`，`action_space=16`、`observation_space=60`、`num_envs=1024`、`episode_length_s=20.0`。
- 同次输出的 CPU powersave 和 IOMMU 为 Isaac Sim 运行环境警告，不影响本次任务配置校验；后续训练前可将 CPU governor 调整为 performance 以降低时延。
- 新增 `smoke_m20pro_locomotion.py`；Isaac Lab 已成功创建 M20 articulation 和平地场景，日志显示 17 个 body / 16 个 joint。
- 修正连续轮关节的无限位置归一化（避免 `Inf/Inf` 导致轮位置观测 NaN）。现在 1 个环境运行 4 步的 reset/step smoke 测试通过：`obs=(1, 60)`、`reward_mean=0.9000`、`terminated=0`。
- 已注册 Gymnasium 任务 `M20Pro-Locomotion-Flat-v0`，并加入 RSL-RL PPO runner 配置。`check_m20pro_gym_registration.py` 已在 headless Kit 中验证 entry point 和 registry。
- 根据 RTX 3060 资源调低初始训练设置：默认任务并行环境现调为 128，PPO 脚本默认为 64。
- 用户已完成首次 PPO smoke 训练：16 个并行环境、2 次迭代。相关 checkpoint 已在后续空间清理中删除，指标和结论保留在本日志；训练进程已退出、显存已释放。
- 完成并行环境基准：16、32、64、128、256 均通过 4 步 reset/step，结果保存在 `logs/m20pro_env_benchmark/summary.tsv`。因为这是短测试，首轮长训练先用 64，再升到 128/256。
- 完成 64 环境、100 次 PPO 训练；checkpoint 已在后续空间清理中删除。TensorBoard 中 `mean_reward` 从约 104.9 到 91.9，`mean_episode_length` 从 54.5 到 42.3；PPO 确实更新，但当前策略有摔倒趋势，需先回放和调整奖励/初始姿态。
- 新增 `play_m20pro_ppo.sh`，用于无头回放 checkpoint，并输出前进位移、最低根高度和终止步数。
- 回放 `model_99.pt` 得到：1000 步平均 +X 位移仅 `0.0038 m`，最低根高度 `0.3594 m`，终止步数 0。结论是策略学会了原地站立，没有学会前进。
- 针对奖励漏洞完成第二版环境：改用 1 m/s 前向速度指数跟踪奖励，降低静止存活奖励，增加侧向/垂向/角速度和动作代价。轮关节改为显式 DC motor 力矩执行器，PPO rollout 由 smoke 的 4 步恢复到 24 步，初始探索噪声由 1.0 降到 0.5。
- 新环境已通过 4 环境 × 24 步 smoke：`obs=(4, 60)`、`reward_mean=0.5866`、`terminated=0`。
- v2 `model_299.pt` 旧回放统计曾报告位移仅 `0.0010 m`，但该统计只计算最终世界坐标减初始坐标，且没有正确累计 done，不能用于自动 reset 环境。修正为速度积分和真实 done 计数后，4 环境 × 200 步结果为：平均积分前进 `3.7245 m`、平均前向速度 `0.9311 m/s`、最低根高度 `0.3546 m`、`done_count=0`、平均绝对动作 `0.6841`。策略已学会接近 1 m/s 目标的前进，但根高度距 0.35 m 终止阈值仅约 4.6 mm，需增加姿态/高度裕量。
- 回放脚本新增 `--video`，可使用 headless rendering 录制单环境 MP4，避免有头 Isaac Sim GUI 的持续开销。已生成 `videos/m20pro_locomotion_v2/model_299-step-0.mp4`（400 步，473833 bytes）；`videos/` 为本地产物，不提交 Git。
- 用户从视频观察到机器人像在无动作平移。增加关节动作/速度诊断后确认：机身速度 `0.9311 m/s`，轮速 `10.3702 rad/s`，按 0.09 m 轮半径计算表面速度为 `0.9333 m/s`，二者仅差约 0.2%，因此不是接触滑移漏洞，而是正常轮式滚动。腿关节平均速度仅 `0.1617 rad/s`，轮子轴对称且无可视标记，所以视频中像无动作平移。
- 范围修正：当前 v2 只是平地轮式移动基线，不是最终轮腿/跳跃策略。平地奖励下策略必然倾向最省力的轮式方案；跳跃阶段必须单独建立禁用/锁定轮驱动的腿部训练课程，再与轮式策略组合。
- 视频还显示 v2 机身低贴、腿姿不对称。关节统计确认策略通过收腿降低重心，这是缺少机身高度/站立构型约束造成的奖励漏洞。v3 加入 0.59 m 机身高度指数跟踪、腿关节零位姿态代价，并把终止高度从 0.35 m 提高到 0.45 m。4 环境 × 24 步 smoke 通过：`reward_mean=1.5866`、`terminated=0`。
- v3 训练受旧 `train_m20pro_ppo.sh` 的 180 s 硬超时影响，实际完成 279/300 次更新，最后定期 checkpoint 为 `model_250.pt`。已移除训练脚本的固定超时，后续长训练不再被 180 s 截断。
- v3 `model_250.pt` 在 16 环境 × 1000 步评估中：平均前向速度 `1.0432 m/s`、最低根高度 `0.5459 m`、轮表面速度 `1.0484 m/s`、16 个 done 全部对应 1000 步/20 s time limit，无提前倒地。相比 v2，机身最低高度由约 0.355 m 提高到 0.546 m。
- v3 已录制 `videos/m20pro_locomotion_v3/model_250-step-0.mp4`（400 步，415325 bytes）用于姿态视觉检查。
- v3 视频复核显示：虽然机身高度已改善，但腿部仍维持僵硬支撑，整体视觉上像平移。这不是继续调整平地速度奖励能解决的问题：对轮腿机器人，无障碍平地上的最优低能耗行为本来就是静态支腿+轮式滚动；人为奖励摆腿只会导致无效抖动。

### 轮腿技能分层决策

- `rolling` 巡航技能：只输出 4 个轮关节动作，12 个腿关节由学习到的/固定站姿控制器保持；用于平地高效移动。
- `jump` 跳跃技能：锁定或禁用轮驱动，只训练 12 个腿关节，使用起跳、腾空高度、越障、着地稳定奖励和逐级障碍课程。
- VLA 高层策略：根据语言目标、相机和激光雷达判断何时巡航、何时切换跳跃，并输出技能条件/子目标。
- 不再把“平地上腿必须摆动”作为训练目标。下一实现里程碑是将现有 v3 重构为 4 维轮式巡航基线，并新建独立 12 维腿部跳跃环境。
- 已新建 `M20ProJumpEnv` / `M20ProJumpEnvCfg`：12 维腿力矩动作、56 维观测，轮关节在 `_apply_action()` 中固定为零力矩；当前目标高度为 0.80 m，还未加入障碍物。
- 跳跃环境已通过 1 环境 × 1 步和 4 环境 × 24 步 smoke（配置接口、刚体和零轮力矩均通过）；当前还没有声称跳跃已训练成功。
- 已将 `M20Pro-Jump-Direct-v0` 注册到 Gym 并接入 `M20ProJumpPPORunnerCfg`；同一个 `train_m20pro_ppo.sh` 可通过 `--task` 切换平地巡航与腿部跳跃。跳跃任务现在还是原地跳跃，尚未加障碍物和 1 m 课程。
- 首版 jump 训练已完成 300 次迭代并生成 `model_299.pt`，但 16 环境 × 200 步回放的最低高度仅 `0.5874 m`，腿动作平均 `0.1088`，基本是原地站立而非起跳。原因是初始高度 0.62 m 在旧高度奖励中已能获得过高收益。
- 已改跳跃奖励：收紧 0.80 m 峰值高度奖励，增大起跳高度进度、向上速度和最高高度进度奖励；新版还未重新训练。
- jump v2 回放表明腿部会动但不起跳：最低高度 `0.4978 m`、腿速 `1.6128 rad/s`、平均前向速度 `0.3375 m/s`，表明不对称腿动作在制造水平漂移。
- jump v3 加入 57 维 phase 观测（下蹲、起跳、腾空三阶段），并将 episode 缩短到 2 s 以增强时序学习。新版已通过 4 环境 × 24 步 smoke，但尚未重新训练；旧 jump checkpoint 与 57 维观测不兼容。
- jump v3 重训后仍不起跳：200 步回放最低高度 `0.5158 m`，说明纯力矩探索仍很难学会协调伸腿。
- 历史记录曾写入开环探针 `max_root_height=0.8223 m`，但在本轮重新验证中无法复现：带前后相机和 LiDAR 的场景下，`squat_minus_thrust_plus` 为 `max=0.7106 m/min=0.0700 m`，反向符号为 `max=0.7130 m/min=0.0700 m`，两者都倒地。因此 0.8223 m 暂时降级为“未复现历史结果”，不能作为成功专家。
- jump v4 重训仍然没有学到时序：200 步回放最低高度 `0.4178 m`、`done_count=32`，但开环探针已证明 PD 目标姿态可以跳。
- 已加入 jump v5 reference bootstrap 奖励：phase 前 30% 学习收腿目标，中间阶段学习伸腿目标，同时保留高度、上向速度和腾空奖励。这是用已验证的开环序列帮助 PPO 先学会可行跳跃时序，后续再降低 reference 权重。
- 已记录 PPO 指标含义：`Loss/value_function` 是 critic 价值回归误差，`Loss/surrogate` 是 PPO 裁剪策略损失，`Policy/mean_noise_std` 是 actor 输出动作分布的平均探索标准差。
- `jump_env.py` 中 reference 项权重为 `8.0` 的改动作为 jump v5 实验保留并单独提交；本轮公开专家/模仿学习主路线不依赖该奖励，也不再以它继续盲调 PPO。

### 公开专家与模仿学习切换（2026-07-21）

- 当前判断：继续调 M20 jump PPO 奖励没有证据能得到可靠动作，正式切换到“公开专家 → 轨迹采集 → M20 动作重定向 → 模仿学习”的路线。PPO 保留为失败对照实验，不再作为主要技能来源。
- Isaac Lab 官方公开的 Go2 rough-terrain RSL-RL checkpoint 已成功下载到 2 TB 盘：`.pretrained_checkpoints/rsl_rl/Isaac-Velocity-Rough-Unitree-Go2-v0/checkpoint.pt`，同时生成 TorchScript 和 ONNX 导出。来源仓库：[IsaacLab](https://github.com/isaac-sim/IsaacLab)，遵循其 BSD-3-Clause 许可和 checkpoint 发布条款。
- `scripts/record_public_go2_expert.sh` 已采集 400 步固定 `0.8 m/s` 前向命令：`datasets/public_go2_rough_v0/episode_0000.h5`，观测维度 `235`（含 187 维 height scan），动作维度 `12`，额外记录实际 `joint_position_target`、命令、状态和 episode done。`done_count=0`。
- 公共专家视频已生成：`videos/public_go2_rough_v0/public-go2-rough-expert-step-0.mp4`，400 步、50 Hz；视频已通过抽帧确认能看到 Go2 行走。所有采集/回放命令均保留 `--video` 和独立 `--video-dir`。
- `scripts/retarget_go2_to_m20.py` 已把 Go2 `hip/thigh/calf` 映射到 M20 `hipx/hipy/knee`，输出 `m20_retarget_v0.h5`。当前只做名称、站立偏置和范围映射，`validated=False`；腿关节符号、偏置和 M20 形态缩放必须通过 M20 视频回放校准后才能训练。
- 重定向第三人称视频验证结果：幅度 `1.0` 时 `min_root_height=0.2710 m`、向后 `0.6189 m`；幅度 `0.50` 时 `min=0.2997 m`、向后 `0.6470 m`；幅度 `0.35` 时 `min=0.5617 m`、位移 `+0.0070 m`；幅度 `0.25` 时 `min=0.5756 m`、位移 `+0.0016 m`。因此 `0.25–0.35` 是当前稳定但几乎不前进的安全区，Go2 足式步态不能直接成为 M20 轮腿巡航动作。
- 对应视频均写入 `videos/m20_retarget_amp025_v0/`、`videos/m20_retarget_amp035_v0/` 和 `videos/m20_retarget_amp050_v0/`，回放器为 `scripts/play_m20_retargeted.sh`；这些 MP4 是校准证据，不是最终 VLA 成果。
- 本轮新增的 M20 传感器采集器 `scripts/record_m20pro_expert.sh` 已验证前后 `160x96` RGB、72 线环形 LiDAR、45 维状态和 HDF5/MP4 同步写入；其开环 jump 样本因倒地标记为诊断数据，不进入成功专家集合。

### 公开 Go1 parkour 专家协议验证（2026-07-22）

- 找到并整理了 Robot Parkour Learning（CoRL 2023，MIT）公开 Go1 checkpoint：`public_experts/parkour_go1/skill/model_674000.pt`（视觉 crawl/jump/leap）和 `walk/model_107500.pt`（本体 walk）。Extreme Parkour 源码曾以 CC BY-NC 4.0 临时保存，已在空间清理中删除；本轮没有把它作为可直接下载的 checkpoint。
- 新增 `scripts/validate_public_parkour_checkpoint.py`。它按 checkpoint 的真实协议构造 `48 维本体 + 1x48x64 深度图`，使用 `weights_only=True` 加载并验证 GRU、动作形状和记忆复位。验证结果：`observation_shape=[4,3120]`、`action_shape=[8,4,12]`、动作范围约 `[-0.8223,0.3638]`，平面/障碍深度会改变动作。
- 新增 `scripts/play_public_go1_parkour.py` 和 `play_public_go1_parkour.sh`，原生加载 Isaac Lab Go1 USD，使用公开 `Kp=40/Kd=0.5`、0.5 action scale、前向深度预处理和第三人称 MP4。所有回放命令强制带 `--video`。
- 公开策略在当前 Isaac Lab 5.1/2.3.2 适配器中没有形成可靠步态：skill checkpoint、0.45 m 障碍、200 步回放为 `x_displacement=-2.3574 m`、`min_root_height=0.0604 m`；无障碍仍倒地。walk checkpoint 作为无视觉对照也失败：200 步为 `x_displacement=-1.9766 m`、`min_root_height=0.0612 m`。
- 已核对并修正过两项输入偏差：深度从米制转换为公开部署使用的 `[0,1]`（0–2 m 范围、裁剪后 resize），Go1 执行器改为显式 `Kp=40/Kd=0.5` DCMotor；修正后 walk 对照仍失败。
- 结论：当前失败不是 M20 关节重定向，也不是“继续训练几轮”能解决的奖励问题，而是公开 IsaacGym parkour checkpoint 与 Isaac Lab USD/执行器/坐标协议尚未完成等价适配。Go1 checkpoint 不能标记为 M20 成功专家，不能直接用于 VLA 训练；当前视频属于协议诊断证据。

下一步调整为：先冻结这些公开 checkpoint，优先做原始 parkour 环境的动作/坐标/PD 逐项等价测试；若无法复现原环境，再只使用公开轨迹作为离线行为克隆数据，不把异构策略硬迁移到 M20。M20-specific rolling/jump 仍需独立采集可验证专家轨迹。

公开参考路线（用于数据/接口，不直接把异构机器人 checkpoint 当成 M20 策略）：

- [LeRobot](https://github.com/huggingface/lerobot)：数据集、动作 chunk 和 ACT/Diffusion/SmolVLA 训练接口。
- [Open X-Embodiment / OpenVLA](https://github.com/openvla/openvla)：开放 VLA 数据和架构参考；7B 全量训练不适合本机 RTX 3060。
- [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)：embodiment/action schema 参考；本机不做 N1.7 全量微调。
- [Unitree RL Gym](https://github.com/unitreerobotics/unitree_rl_gym)：公开四足 RL 训练/部署参考，后续用于动作语义核对。

此前的 Go2→M20 重定向路线已冻结为失败对照；当前优先使用原生 M20 专家采集同构数据，再把前后相机、LiDAR、语言任务标签合并进 LeRobot-compatible 数据集，最后训练小型 BC/ACT 基线。

### 原生 M20 专家协议与首次有效回放（2026-07-22）

- 在 GitHub 仓库 [AI-DA-STC/M20-autonomy-sim](https://github.com/AI-DA-STC/M20-autonomy-sim) 中找到原生 M20 `policy.onnx`。模型输入为 `obs[1,57]`，输出为 `actions[1,16]`；当前 Conda 环境已安装 `onnxruntime==1.27.0`，CPU 推理足够支撑单环境 50 Hz。
- 官方观测顺序为 `base_omega*0.25(3) + projected_gravity(3) + command(3) + joint_pos-default(16) + joint_vel*0.05(16) + last_action(16)`。动作顺序是 12 个腿关节后接 4 个轮关节，腿缩放 `[0.125,0.25,0.25]`、轮速度缩放 `5.0`，官方策略协议的轮 `Kd=0.6`；Isaac Lab 适配器默认平移使用 `Kd=0.6`，转向命令自动使用 Gazebo bridge `wheel_kd_scale=6.0` 对应的 `Kd=3.6`，腿仍为 `Kp=80/Kd=2`。
- 官方 M20 的后腿髋/膝姿态必须镜像：默认策略姿态为 `FL/FR=[0,-0.6,1.0]`、`HL/HR=[0,0.6,-1.0]`。此前开环 jump 和 Go2 重定向把后腿符号处理错，导致“平移、僵硬、倒地”的错误现象；新适配器严格按官方 policy order 和 USD joint order 验证。
- 原生 rolling 回放（第三人称 MP4）结果：`500` 个策略步、命令 `[0.5,0,0]`，`x_displacement=14.7179 m`，`mean_forward_speed=0.3684 m/s`，`min_root_height=0.5154 m`，`max_root_height=0.5295 m`，`terminated_steps=0`。这证明 M20 USD、轮执行器和关节协议已经能产生持续滚动；腿保持支撑姿态是该 rolling policy 的预期行为，不应再把“腿不摆”当作平地轮式专家失败。
- 近景复核视频：`videos/public_m20_native_close_v1/m20-native-x+0.50-y+0.00-yaw+0.00-step-0.mp4`，250 步得到 `x_displacement=7.3376 m`、`mean_forward_speed=0.3678 m/s`、`min_root_height=0.5154 m`、`terminated_steps=0`。
- 新增 [play_public_m20_policy.py](scripts/play_public_m20_policy.py) / `play_public_m20_policy.sh`，每次回放强制带 `--video`；新增 [record_public_m20_expert.py](scripts/record_public_m20_expert.py) / `record_public_m20_expert.sh`，记录前后 RGB、72 线 360° LiDAR、57 维原生 proprio、45 维全状态、16 维动作、command 和自然语言标签。
- 采集器 5 步 smoke 已通过：HDF5 形状为 `front/rear=(5,96,160,3)`、`lidar=(5,72)`、`proprio=(5,57)`、`state=(5,45)`、`action=(5,16)`，`terminated_steps=0`，同步 MP4 正常写入。该 smoke 仅是格式验证，正式数据需使用完整 episode。
- 第一批正式 rolling 数据已采集到 `datasets/public_m20_native_v1/`：4 个 episode、每条 500 帧/10 秒、每条均 `success=true`。逐文件检查结果：位移 `14.7162–14.7326 m`，最低根高 `0.5152–0.5154 m`，`terminated_steps=0`，所有 RGB/LiDAR/proprio/state/action 数值有限；对应 4 个 MP4 均为 500 帧、50 Hz、`480x288`。这批数据是后续语言条件 action-chunk BC/VLA 的 rolling 正样本，不包含 jump 标签。
- 额外命令覆盖结果：`向后走`（`command_x=-0.5`、`Kd=0.6`）250 步保持稳定，`x_displacement=-6.3242 m`、`min_root_height=0.5152 m`、`terminated_steps=0`，可作为反向 rolling 正样本；旧 `Kd=0.6` 的 `向左转` 产生 `yaw_delta=1.4632 rad` 同时 `x=-4.9744 m/y=-4.0087 m` 漂移，已标记 `success=false`。转向使用 `Kd=3.6` 后，250 步为 `yaw_delta=0.6105 rad`、`x=-1.3562 m`、`y=-0.3536 m`、`min_root_height=0.5168 m`、`success=true`；统一 `Kd=3.6` 的 forward 对照反而倒退 `x=-1.3055 m`，因此代码采用按 command 自适应，不覆盖已验证的平移数据。
- 采集器已加入命令有效性判据：有 forward command 必须位移方向正确；有 yaw command 除转向方向外还要求平移漂移小于 2 m。仅“不跌倒”不再自动算成功。
- 上游仓库未发现可供重新分发的根目录 LICENSE 文件；本项目只在本机保留来源说明和个人研究验证，不把外部源码或权重提交到仿真仓库或 VLA-Learning 仓库。

当前判断：原生 M20 policy 已经是合格的 rolling 专家，但它没有跳跃能力，也没有语言、相机和 LiDAR 决策能力。因此“自然语言找物体/导航 + 1 m 障碍跳跃”仍未完成；下一步先采集多命令 rolling 正样本，再加入可物理验证的 jump skill，最后训练小型语言条件 action-chunk BC/VLA，而不是继续盲目增加 PPO 迭代。

### 语言条件多模态 VLA/BC v1（2026-07-22）

- 新增 [train_m20_vla_bc.py](scripts/train_m20_vla_bc.py)：输入为前后 RGB（下采样为 `6x48x80`）、72 线 LiDAR、原生 57 维 proprio 和 UTF-8 语言字节序列，输出连续 `8x16` action chunk。它是公开专家模仿学习，不使用奖励函数或 PPO。
- 训练数据只读取 HDF5 `success=true` episode：forward 4 条、backward 1 条、turn v2 1 条，共 `6` 条 episode、`985` 个训练窗口、`247` 个验证窗口。checkpoint 位于 `checkpoints/m20_vla_bc_v1/best.pt`，约 `1.8 MB`，训练最佳验证损失 `9.741e-5`。
- 新增 [play_m20_vla_bc.py](scripts/play_m20_vla_bc.py)：默认每 4 个控制周期执行一个预测 chunk，再重新读取传感器，真正使用 action chunk 而不是只取第一步；每次强制写 MP4 和 JSON 指标。
- 闭环 forward chunk4 250 步结果：`x_displacement=7.3347 m`、`mean_forward_speed=0.3676 m/s`、`yaw_delta=0.026 rad`、`min_root_height=0.5152 m`、`terminated_steps=0`。50 步 backward 结果为 `x=-1.2321 m`、平均 `-0.3112 m/s`、`terminated_steps=0`；50 步 turn 结果为 `yaw_delta=0.114 rad`、`x=0.4440 m`、`min_root_height=0.5179 m`、`terminated_steps=0`。
- 当前 VLA 只学会语言条件下的 rolling/backward/turn 低层动作，语言标签仍是运动指令；它尚未学习“寻找某物体”、地图/无地图探索或 jump skill，不能把 v1 宣称为完整任务完成。

### 视觉目标到达 VLA/BC v2-v4（2026-07-22）

- `record_public_m20_expert.py` 已支持红/蓝/绿非碰撞方块、目标坐标、语言标签和 `--stop-on-target`。模拟器目标真值只用于生成专家停车标签和计算验收指标；VLA 回放控制器不读取目标坐标或成功半径，控制输入仍是前后 RGB、LiDAR、proprio 和语言。
- 修正了一处关键的 VLA 输入泄漏：原生 57 维 proprio 的 `6:9` 是专家速度 command，训练和回放现均将这三维清零。模型必须从语言和视觉推断任务，不再直接复制专家 command。
- 训练拆分改为按语言分组；只有重复语言才留出验证 episode，单条颜色数据不会被错误分到纯验证集。训练窗口使用 `WeightedRandomSampler` 按任务文本均衡采样，避免四条长 forward 数据淹没颜色目标样本。
- v2 使用固定帧数停车，红色闭环通过，但蓝/绿仍穿过目标。数据复核确认固定 `stop_after` 发生时目标已经离开前相机，因此标签与视觉时序不一致。
- v3 改为专家首次进入 `0.8 m` 目标半径后立刻输出全零动作；红色通过，但旧图像编码器的全局平均池化会抹掉目标位置和尺度，蓝色仍失败。该失败没有靠继续堆 epoch 掩盖。
- v4 新增 `spatial_v2` 图像编码：保留 `3x5` 空间网格并投影到 128 维，同时继续兼容旧 `global_v1` checkpoint。训练使用 9 条成功 episode、`1206` 个训练窗口和 `247` 个验证窗口，80 epochs 最终训练损失 `0.002181`，最佳 forward 验证损失 `1.03e-4`。
- v4 checkpoint：`/media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/checkpoints/m20_vla_bc_v4/best.pt`。
- 红色目标闭环：`x=1.3046 m`、最小目标距离 `0.6575 m`、到达后平均速度 `0.0280 m/s`、最终平面速度 `0.0010 m/s`、`terminated_steps=0`、`success=true`。
- 蓝色目标闭环：`x=2.3656 m`、最小目标距离 `0.6779 m`、到达后平均速度 `0.0317 m/s`、最终平面速度 `0.0013 m/s`、`terminated_steps=0`、`success=true`。
- 绿色目标闭环：`x=2.0100 m`、最小目标距离 `0.7086 m`、到达后平均速度 `0.0287 m/s`、最终平面速度 `0.0026 m/s`、`terminated_steps=0`、`success=true`。
- 普通“向前走”回归测试仍通过：250 步位移 `7.3197 m`、平均速度 `0.3669 m/s`、最低根高 `0.5152 m`、`terminated_steps=0`。空间视觉能力没有破坏原生 rolling 动作质量。
- 对应 MP4 和 JSON 位于 `videos/m20_vla_bc_v4/{red,blue,green,forward}/` 与 `logs/m20_vla_bc_v4_*.json`。这只是固定场景下的“看到指定颜色目标并停车”里程碑；尚未覆盖随机目标位置、主动转向搜索、开放词汇、地图/无地图探索、障碍物和跳跃，不能宣称完整任务已完成。

### 主动视觉搜索专家边界（2026-07-22 续）

- 公开 `AI-DA-STC/M20-autonomy-sim` README 明确显示其 M20 ONNX 是 `/cmd_vel -> RL locomotion` 低层控制器；公开仓库的导航部分是 CMU/FAR/local planner，没有 M20 jump policy 或视觉目标搜索专家 checkpoint。该公开经验可以作为 VLA 的同构 rolling 专家，但不能直接提供完整 ObjectNav/jump 示范。
- 尝试用目标真值生成离线专家：连续 bearing command、差速轮覆盖、固定 turn→forward 技能串联都做了带 MP4 的 smoke。它们分别出现异常角速度、命中后惯性回滑、目标未到达等问题，全部 `success=false`，已删除对应 HDF5/MP4，未进入训练集。`command_y=0.5` 协议测试也不是侧移，而是倒退并伴随转向，不能当横向导航专家。
- 当前可复用的数据仍只包括固定直线目标和公开 rolling/backward/turn 正样本；下一次搜索必须先解决原生 turn/forward 技能的物理切换和刹停，再采集随机方位目标，不能用失败轨迹硬训模型。

### 导航回放诊断与动力学一致性（2026-07-22）

- 用户回放观察到机器狗仍然像平移，且腿部动作僵硬。这个判断与指标一致：公开 M20 ONNX 是轮式 rolling 低层策略，不是四足步态或完整 ObjectNav 专家；继续训练 v4/v5 不能凭空产生转向和跳跃能力。
- 原生 turn skill 的最大机身角速度约 `12.8–13.2 rad/s`；把 command/Kd 分段平滑到 60 步仍约 `13.36 rad/s`，因此没有把它用于随机目标专家。
- 直接轮速 pattern（幅值 `1.0/0.5/0.25`）全部出现塌机身，最低根高约 `0.24–0.39 m`，不作为低层控制器。
- 目标 `(3,-1)` 的失败验收记录：旧 override 在进入 `0.8 m` 后又滑到 `1.874 m`、末速度 `0.612 m/s`；连续闭环 v1 末距 `1.109 m` 且 `yaw_delta=0.033 rad`；滑移增益 v2 把目标转到错误的 `y=+0.791 m`；修正符号 v3 仍末距 `1.044 m` 且几乎不转。公开 `cmd_vel` 通道的 combined yaw 末 `yaw_delta=-0.006 rad`，纯 yaw `0.05` 也只有 `0.0015 rad`。这些 HDF5/MP4 已清理，全部 `success=false`，没有进入训练集。
- 对照官方 `AI-DA-STC/M20-autonomy-sim`：ONNX 的 MuJoCo `M20.xml` 使用不同的 link 惯量和 `friction="1 0.01 0.001"` 接触模型；当前 URDF→USD 只保留简化碰撞和各向同性 PhysX 摩擦，动力学并不等价。官方 Gazebo 控制桥的轮子 `Kd=0.6`，launch 默认把 wheel Kd scale 设为 `1.0`；此前为 yaw 临时放大的 `3.6` 已确认会造成角速度尖峰。
- 新增 [convert_m20_mjcf.py](scripts/convert_m20_mjcf.py) / [convert_m20_mjcf.sh](scripts/convert_m20_mjcf.sh)，显式启用 Isaac Sim 5.1 的 MJCF importer，并删除 importer 生成的第二 articulation root `/M20/worldBody`；`assets/m20_mjcf_official_nofloor_v6.usd` 已在 2 TB 盘生成。它通过了 `17 bodies / 16 joints` 和 240 步静态 smoke（最终根高约 `0.560 m`），但尚未替换默认资产。
- candidate 前进回放：150 步位移 `4.389 m`、最低根高 `0.5151 m`、最大角速度 `0.1571 rad/s`，带 MP4 保存在 `videos/public_m20_mjcf_forward_v2/`。
- candidate 转向回放：`wheel Kd=0.2`、正 yaw `0.5` 得到 `yaw_delta=+0.3846 rad`、最大角速度 `0.5406 rad/s`；负 yaw `-0.3` 只有 `-0.0459 rad`，负 yaw `-0.5` 在 Kd `0.6` 下为 `-0.1325 rad`。这表明当前转换资产/公开网络仍有单向转向不对称，不能直接作为任意方位导航专家。
- candidate recorder 已根据 `M20PRO_USD_PATH` 自动把相机/LiDAR 挂到 `base_link/base_link`；固定负转向 300 步后前进的目标串联仍 `success=false`（最终距目标 `2.823 m`、路径 `10.491 m`），失败 HDF5/MP4 已删除。
- 当前决策：暂停随机方位专家采集和 VLA 继续训练；先让官方 MJCF robot-only USD 通过拓扑、直行、转向三项带视频回归，再采集成功的随机目标轨迹。跳跃仍需独立的落地控制器，不能用当前 rolling policy 代替。

### Jump 专家搜索边界（2026-07-22）

- GitHub/API 搜索了 M20/DeepRobotics/wheel-legged parkour 公开仓库，没有找到可复用的 M20 jump checkpoint；保留的原生 M20公开专家只有 rolling `policy.onnx`。
- 新增 [search_m20_jump_expert.py](scripts/search_m20_jump_expert.py)，显式使用官方镜像姿态和前后腿镜像膝关节，采用 `Kp=200/Kd=4`、短下蹲/起跳目标序列，并行执行 `324` 个纯物理候选；v2 最高 `max_root_height=0.6618 m`，v3 扩展时序后最高 `0.6738 m`，所有候选均未满足 `min_root_height>=0.45 m` 的稳定判据。
- 这证明当前 M20 USD/执行器和简单关节目标序列还不能产生可验证 1 m 跳跃；失败序列不进入 VLA 数据集。搜索结果只保留在 2 TB 盘 `logs/m20_jump_expert_search_v1/v2/v3.json`，不再继续盲目 PPO 跳跃训练。
- 继续扩大无奖励物理搜索：腿执行器力矩 `150` 时最高 `0.882 m`，力矩 `300`、`Kp=500/Kd=8` 时最高 `0.920 m`；所有最高候选最低根高约 `0.16–0.19 m`，`survived=false`。这说明“跳得高但摔倒”不能作为专家，当前仍没有可用于 VLA 的 1 m jump action chunk。

### 数据清理与存储（2026-07-22）

- 已删除旧 `logs/rsl_rl` PPO checkpoint、失败 jump HDF5/视频、重复诊断视频，以及本轮失败目标搜索 HDF5/MP4；失败 MJCF v1 candidate 也已删除。
- 只保留：成功的 M20 rolling/目标数据、v4 checkpoint、必要的 M20 `policy.onnx`/协议文件、公开 Go1 checkpoint、跳跃搜索 JSON，以及用于动力学核对的官方 M20 模型源码快照和 no-floor MJCF candidate，全部在 2 TB 盘。
- 官方源码快照只用于个人仿真协议核对，不复制进本 Git 仓库，也不重新分发外部权重；candidate USD 尚未成为默认资产。

### 公开 VLA 架构检索与 M20 兼容性探针（2026-07-22）

- 检索并核对了公开的 [NaVILA](https://github.com/AnjieCheng/NaVILA)、[NaVILA-Bench](https://github.com/yang-zj1026/NaVILA-Bench) 和 [legged-loco](https://github.com/yang-zj1026/legged-loco)。它们采用“高层视觉语言导航动作 + 独立低层 locomotion policy”的两层协议；公开 NaVILA Llama-3 8B checkpoint 约需 24 GB 以上显存，当前 RTX 3060 12 GB 不适合直接部署。
- 已下载公开 NaVILA Go2 vision locomotion checkpoint：`public_experts/navila_go2_vision/policy.jit`，协议为 `909 -> 12` 个 Go2 腿关节目标。它不是 M20 专家，也不是完整 ObjectNav 或 jump VLA。
- 新增 [play_public_navila_go2_on_m20.py](scripts/play_public_navila_go2_on_m20.py) 做物理兼容性探针：固定 M20 关节顺序、官方镜像站立姿态，输入 909 维观测，保留第三人称 MP4 和 JSON 指标。`100` 步 smoke 实测 `x_displacement=0.1189 m`、`min_root_height=0.2131 m`、`terminated_steps=97`，说明 Go2 权重直接套用 M20 会倒地，不能进入训练集或宣称为 M20 低层专家。
- 探针结果确认两层 VLA 思路值得借鉴，但低层必须换成 M20 原生 rolling/未来的 M20 jump expert；当前不再继续把 Go2/Go1 异构关节动作硬重定向到 M20。
- 本轮清理后 2 TB 输出根目录约 `289 MB`，失败目标搜索数据、旧 checkpoint 和重复视频已移除；保留的每条诊断结果仍有 MP4、JSON 和来源说明。

## 常用验证

```bash
cd /home/fabu/桌面/M20Pro/m20pro_sim_ros2_ws
source scripts/activate_vla_env.sh
./scripts/convert_m20pro_urdf.sh
./scripts/test_m20pro_asset.sh
TERM=xterm python scripts/check_m20pro_task.py --headless
```

官方 M20 MJCF→USD 动力学基线（输出在 2 TB 盘，当前仍是实验 candidate）：

```bash
./scripts/convert_m20_mjcf.sh \
  --output /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/assets/m20_mjcf_official_nofloor_v6.usd

M20PRO_USD_PATH=/media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/assets/m20_mjcf_official_nofloor_v6.usd \
  ./scripts/test_m20pro_asset.sh --steps 240
```

`isaaclab.sh` 运行时需要 `TERM=xterm`，以避免 `TERM=dumb` 下 `tabs` 命令失败。

回放命令默认应包含 `--video` 和独立 `--video-dir`，以 headless rendering 生成 MP4；不再只给无视频的回放命令。

公开 Go2 专家采集（自动包含视频）：

```bash
./scripts/record_public_go2_expert.sh \
  --steps 400 \
  --output-dir /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/datasets/public_go2_rough_v0 \
  --video-dir /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/videos/public_go2_rough_v0
```

Go2→M20 目标动作重定向（此命令不启动 Isaac Sim）：

```bash
python scripts/retarget_go2_to_m20.py \
  /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/datasets/public_go2_rough_v0/episode_0000.h5 \
  /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/datasets/public_go2_rough_v0/m20_retarget_v0.h5
```

M20 第三人称重定向回放（自动录制视频）：

```bash
./scripts/play_m20_retargeted.sh \
  --actions-h5 /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/datasets/public_go2_rough_v0/m20_retarget_v0.h5 \
  --steps 200 \
  --video-dir /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/videos/m20_retargeted_v0
```

原生 M20 专家回放（自动包含近景第三人称视频）：

```bash
./scripts/play_public_m20_policy.sh \
  --steps 500 --warmup-steps 75 \
  --command-x 0.5 --command-y 0.0 --command-yaw 0.0 \
  --video-dir /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/videos/public_m20_native_v1
```

训练带空间视觉编码的语言条件 action-chunk VLA/BC：

```bash
./scripts/train_m20_vla_bc.sh \
  --architecture spatial_v2 \
  --epochs 80 --batch-size 64 --horizon 8 --stride 2 \
  --device cuda:0 \
  --output-dir /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/checkpoints/m20_vla_bc_v4
```

VLA 红色目标闭环回放（wrapper 自动带 `--headless --video`，并写 MP4/JSON）：

```bash
./scripts/play_m20_vla_bc.sh \
  --checkpoint /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/checkpoints/m20_vla_bc_v4/best.pt \
  --task-text "到红色方块去" \
  --target-color red --target-x 2.0 --target-y 0.0 \
  --steps 160 --chunk-execution 4 \
  --video-dir /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/videos/m20_vla_bc_v4/red \
  --metrics /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/logs/m20_vla_bc_v4_red.json
```

采集原生 M20 rolling 正样本（每个 episode 自动写 HDF5 和 MP4）：

```bash
./scripts/record_public_m20_expert.sh \
  --episodes 4 --steps 500 --warmup-steps 75 \
  --task-text "向前走" --command-x 0.5 \
  --output-dir /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/datasets/public_m20_native_v1 \
  --video-dir /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/videos/public_m20_native_v1
```

训练两层高层 VLA 技能选择器（监督模仿，不使用奖励函数）：

```bash
./scripts/train_m20_vla_skill.sh \
  --epochs 60 --batch-size 64 --post-reach-steps 20 --device cuda:0 \
  --output-dir /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/checkpoints/m20_vla_skill_v8
```

两层 VLA 红色目标回放（高层 CPU 推理、低层公开 M20 ONNX、自动写视频）：

```bash
export M20PRO_USD_PATH=/media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/assets/m20_mjcf_official_nofloor_v6.usd
./scripts/play_m20_vla_skill.sh \
  --checkpoint /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/checkpoints/m20_vla_skill_v8/best.pt \
  --policy /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/public_experts/m20_native/policy.onnx \
  --task-text "到红色方块去" --target-color red --target-x 2.5 --target-y 0.0 \
  --steps 300 --warmup-steps 75 --model-device cpu \
  --video-dir /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/videos/m20_vla_skill_v8/red \
  --metrics /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/logs/m20_vla_skill_v8_red.json
```

两层 VLA search 技能回放（当前 candidate 的物理 yaw 适配为单向扫描）：

```bash
./scripts/play_m20_vla_skill.sh \
  --checkpoint /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/checkpoints/m20_vla_skill_v8/best.pt \
  --policy /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/public_experts/m20_native/policy.onnx \
  --task-text "寻找目标" --target-color none --steps 120 --warmup-steps 75 \
  --search-yaw-command 0.5 --model-device cpu \
  --video-dir /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/videos/m20_vla_skill_v8/search \
  --metrics /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/logs/m20_vla_skill_v8_search.json
```

## PPO / TensorBoard 指标词典

### 当前 locomotion 奖励函数

实现位于 `tasks/m20pro_locomotion/m20pro_locomotion_env.py::_get_rewards()`。它不是云深处官方奖励，也不是从某个现成机器人任务原样复制；它使用常见的速度跟踪 locomotion 结构，并针对旧策略“原地站立即获得高奖励”的回放结果设定了第一组工程初值。

```text
reward =
    2.0   * exp(-(vx - 1.0)^2 / 0.25)   # 跟踪 1 m/s 前向速度
  + 1.0   * exp(-(base_z - 0.59)^2/0.0025) # 跟踪正常机身高度
  + 0.5   * upright                      # 保持机身直立
  + 0.05                                 # 小额存活奖励
  - 0.20  * ||leg_joint_position||^2     # 保持对称零位站姿
  - 0.10  * vy^2                         # 抑制侧滑
  - 0.20  * vz^2                         # 抑制垂向抖动
  - 0.02  * ||angular_velocity||^2       # 抑制机身旋转
  - 0.005 * ||action||^2                 # 抑制过大动作
```

根高度低于 `0.45 m` 终止 episode，终止步奖励为 `-2.0`。这组系数仍是工程基线，必须根据回放中的前向速度、倒地率、姿态和能耗继续调整。

### `Loss/value_function`

价值网络（critic）的回归误差。critic 学习预测 `V(s)`，即从当前状态开始未来可获得的累计奖励。该损失衡量预测回报与实际回报的差异。

- 通常越低越好，但训练早期或状态分布快速变化时可能暂时升高。
- 持续很高表明 critic 无法稳定区分好状态和差状态，常见原因是奖励尺度过大、episode 频繁终止或 rollout 过短。
- 本次 64 环境 / 100 iterations 从约 `2.9` 升到 `399`，明显偏高，需检查摔倒、奖励尺度和 `num_steps_per_env`。

### `Loss/surrogate`

PPO 的 actor 策略损失。它根据新旧策略对同一动作的概率比率和 advantage 更新策略，并使用 clipping 防止单次更新过大。

- 它不是简单的“越低越好”，正常情况下在 0 附近小幅波动。
- 绝对值很大可能表示策略更新过猛；长期非常接近 0 可能表示已收敛，也可能是 advantage 太小或没有有效改善方向。
- 本次从约 `-0.072` 到 `-0.0026`，策略更新逐渐变小；由于 mean reward 同时下降，不能解读为成功收敛。

### `Policy/mean_noise_std`

Actor 输出的动作高斯分布的平均标准差。策略从 `Normal(mean_action, noise_std)` 采样动作，标准差决定探索强度。

- 数值高表示动作更随机、探索更强；数值低表示策略更确定。
- 过高可能产生关节抖动和摔倒；过低可能过早停止探索，陷入差策略。
- 本次从约 `0.999` 到 `0.995`，几乎没有下降，表明策略仍在强随机探索；对当前容易摔倒的 M20 任务可能过高。

### 当前综合判断

当前曲线组合是 critic 估值趋于不稳定、actor 更新趋于变小、探索噪声仍很高，同时 mean reward 和 episode length 下降。因此暂不认定平地移动策略已收敛，需先回放 checkpoint，检查倒地率、根高度、前向速度和奖励设计。

### VLA 闭环修复与 v14-v20 诊断（2026-07-22）

- 修复了 `play_m20_vla_bc.py` 的两个闭环错误：第三人称相机必须在物理步前设置，否则 MP4 会录到机器人内部的旧相机姿态；`horizon=1` 每次重新预测后必须把 `cached_index` 归零，否则第二步访问越界并被 Isaac Sim 退出流程掩盖成“卡死”。
- VLA 回放的推理默认移到 CPU（`--model-device cpu`），Isaac Sim 仍使用 GPU；这避免相机/PhysX 与小型 PyTorch 网络共享 CUDA 上下文。所有回放仍强制 `--video`，新增 `--debug-steps` 可定位传感器、推理和物理阶段。
- v14 对蓝色目标 `(3,-1)` 的有效 300 步回放：第 `139` 步进入目标半径，最小距离 `0.111 m`，但旧数据在到达后仍标注滚动动作，最终距离 `6.585 m`、最低根高 `0.291 m`、`terminated_steps=213`。这解释了“能找到但冲过/摔倒”。
- v15 将成功轨迹到达后的 40 帧重标为零动作，最佳训练损失 `0.000648`，但闭环提前失稳，未进入目标；说明直接把停车动作混入 16 维腿/轮连续回归会损伤协变量鲁棒性。
- 新增 DAgger 采集支持并修正 `stop_on_target`：`m20_dagger_stop_blue_v1` 在第 `178` 步达到最小距离 `0.242 m`，到达后 522 帧为零动作；训练加载器现对 `success` 和 `dagger` 轨迹统一截断为到达后 40 帧，避免零动作尾段淹没滚动样本。
- v17 加入截断后的 DAgger 数据后保持稳定（300 步 `terminated_steps=0`），但只前进 `0.467 m`、最小目标距离 `2.497 m`；仍不足以称为导航成功。
- 新增独立 learned stop head：它和动作头共享 RGB/LiDAR/proprio/语言特征，训练时用 `target_reached_step` 生成 BCE 停止标签，回放通过 `--stop-threshold` 触发停止；v18 从头联合训练会误触发或改变动作分布，v20 改为冻结 v14 动作头、只训练 stop head，但受 GPU PhysX 非确定性影响，本次最小距离 `0.890 m`，仍未成功。
- 停止触发后新增短时基于机身速度的轮刹车，仅负责耗散滚动惯性，不负责目标选择或路径规划。当前结果仍是诊断里程碑：VLA rolling 能力和可学习停车接口已建立，但随机方位 ObjectNav、稳定停车和 1 m 跳跃均未完成。

当前最有价值的可复现视频/指标：

```text
videos/v14_fixed_valid_300/        # v14 到达目标后冲过并倒地
videos/v17_dagger_trimmed_blue_300/ # DAgger 截断后稳定但未到达
videos/v20_frozen_stop_blue_300/  # 冻结动作头 + learned stop
```

下一轮应优先采集多个初始姿态、目标距离和目标横向位置的 DAgger 轨迹，并以“到达后 0.8 m 内保持 2 s、根高 >= 0.45 m、无倒地”为验收；在此之前不再增加 epoch，也不进入 1 m jump 集成。

### 两层 VLA 技能策略 v1-v11（2026-07-22）

- 新增 [m20_vla_skill_model.py](scripts/m20_vla_skill_model.py)、[train_m20_vla_skill.py](scripts/train_m20_vla_skill.py) 和 [play_m20_vla_skill.py](scripts/play_m20_vla_skill.py)。高层模型输入前后 RGB、72 线 LiDAR、去除专家 command/action history 的 57 维 proprio 和语言，输出 `forward/backward/left/right/stop/search/jump` 技能以及给 M20 原生 ONNX 的 3 维 command；低层仍是公开 `AI-DA-STC/M20-autonomy-sim` 的 `57 -> 16` 专家，不使用 PPO 奖励。
- 修正了两项数据/闭环问题：到达后只保留 20 帧停车样本，避免静止尾段淹没滚动数据；慢速接近命令（约 `0.045 m/s`）不能按 stop 标注，只有明确全零命令才是 stop。当前 v8 checkpoint 位于 `checkpoints/m20_vla_skill_v8/best.pt`，19 条 episode、2254 个训练帧，验证技能准确率约 `0.98`；标签覆盖 `forward/backward/left/right/stop/search`，`jump` 仍为零标签。v8 新增 3 条物理稳定的 `寻找目标` 扫描轨迹；它们因 candidate yaw 符号与公开 command 约定相反而不标作导航 success，但作为 search skill 数据保留。
- 红色目标 `(2.5, 0.0)` 两层闭环通过：`target_reached=true`、最小距离 `0.7376 m`、停止后稳定、`terminated_steps=0`。关键视频：`videos/m20_vla_skill_v4/red_center/`。
- 绿色目标 `(2.5, 0.4)` 两层闭环通过：`target_reached=true`、最小距离 `0.6346 m`、停止步 `121`、`terminated_steps=0`。关键视频：`videos/m20_vla_skill_v5/green_left04/`。
- 蓝色正前方目标能识别并接近，但当前固定停止确认窗口的指标为 `0.8005–0.8482 m`，尚未按严格 `0.8 m` 验收；蓝色横向目标能进入半径但最终不停车，且 yaw/横向转向不足。当前结论是两层接口和 RGB 目标识别已建立，主动搜索、任意横向方位和稳定停车尚未完成。
- `寻找目标` search 回放：120 步 `skill_counts={'search':120}`、`min_root_height=0.515 m`、`terminated_steps=0`、`yaw_delta=-0.132 rad`，视频位于 `videos/m20_vla_skill_v8/search_v2/`。当前 candidate 只能稳定单向扫描，尚未把目标重新获取和 stop 串成成功 ObjectNav。
- 新增成功专家数据：红色 `(2.2,0.25)`、绿色 `(2.8,0.8)`、绿色近距 `(1.5,0.4)`、蓝色负向镜像 `(3.0,-0.75)`、蓝色近距 `(1.5,-0.4)`；失败的蓝色负转弯和红色远距轨迹已删除，不进入训练集。
- 当前两层回放命令必须包含 `--headless --video`。停止候选使用连续确认/投票，只负责耗散接近目标时的惯性；它没有读取目标坐标，目标坐标只用于离线指标。

### v11 隔离 search head 复测（2026-07-22）

- v11 从 v7 初始化，冻结视觉、LiDAR、本体状态、融合、command 和导航主干，只增加独立的语言 `search_intent_head`；这样 search 语义不会直接改写已验证的 rolling 主干。
- 红色目标 `(2.5, 0.0)` 回放通过：`target_reached=true`、第 `100` 步进入目标判定、最小距离 `0.4663 m`、第 `120` 步停止、`terminated_steps=0`。视频：`videos/m20_vla_skill_v11/red_center/`。
- 绿色目标 `(2.5, 0.4)` 复测失败：最小距离 `1.1601 m`，第 `99` 步误触发停止，`terminated_steps=0`；技能计数为 `left=26`、`forward_recovery=63`、`stop=211`。因此 v11 不能作为全场景导航成功版本，需先区分 GPU PhysX 非确定性和 search head/输入变化造成的行为偏移。
- v11 search 扫描本身稳定：`120` 步全为 `search`，`min_root_height=0.515 m`、`terminated_steps=0`、`yaw_delta=-0.132 rad`；这仍不代表发现物体或完成 ObjectNav。

### v12-v13 learned target stop 与视频兼容性（2026-07-22）

- v7/v11 在当前闭环代码下都可重复复现绿色目标提前停车：第 `99` 步停止、最小距离 `1.1601 m`。因此该问题不是 search head 或单次 PhysX 随机性，而是旧 stop/command 输出在目标尚未进入 `0.8 m` 半径时过早衰减。
- v12 从 v7 初始化并冻结导航主干，只训练独立 `target-stop head`；它仍在约 `1.12 m` 处误停。v13 将 head 扩为停止概率和归一化视觉距离联合监督，目标坐标和根位置只用于离线标签，推理仍只看前后 RGB、LiDAR、本体状态和语言。
- v13 使用理论距离阈值 `0.16`（约 `0.8 m / 5 m`）时在 `1.1539 m` 误停；阈值降到 `0.10` 后不再触发 stop，但 command 回归头把前进速度降到接近零，最终仍停在 `1.1543 m`。因此不能把 v13 标记为成功。
- 为尚未满足 learned stop 的 `forward` 技能设置 `0.08 m/s` 最小接近速度后，阈值 `0.10` 和 `0.14` 两次回放都进入目标半径，最小距离分别为 `0.3439 m` 和 `0.3438 m`，且 `terminated_steps=0`；但两次均为 `stop_step=None`，机器人穿过目标而没有停车。因此最小接近速度只解决了约 `1.15 m` 处速度衰减，learned stop 仍未校准，下一步必须记录完整预测曲线而不是继续猜阈值。
- 定位到此前全部 MP4 使用 OpenCV `mp4v`（MPEG-4 Part 2），文件本身完整但桌面播放器兼容性差。新增 [video_utils.py](scripts/video_utils.py) 和 [convert_videos_to_h264.py](scripts/convert_videos_to_h264.py)，所有 9 个录制/回放脚本在关闭视频后自动原子转为 `H.264/yuv420p + faststart`。
- 2 TB 盘上的视频已再次全量扫描，并将 `logs/videos/` 中最后一个遗漏的 MPEG-4 文件原位转码。当前已有 `55/55` 为 H.264（含 v13 trace 新回放），使用打包的 FFmpeg 从第一帧到最后一帧完整解码，失败数为 `0`；视频主目录约 `11 MB`，桌面播放器不会再依赖旧 `mp4v` 解码器。
- 新录制链路另以 `videos/h264_writer_smoke_v1/` 完成 30 帧实测：输出为 `H.264/yuv420p`、`50 FPS`、`480x288`，首帧和末帧均可解码；桌面 `video/mp4` 默认关联已统一到 Microsoft Edge。

### v13 trace 与 v14 独立目标视觉分支（2026-07-22）

- `play_m20_vla_skill.py` 的指标格式升级为 `m20_vla_two_layer_replay_v2`：每次高层推理保存真实评估距离、预测距离、target/generic stop 概率、原始/选中技能、命令和 stop latch 状态，并汇总预测距离最小值、stop 概率最大值及对应步数。目标坐标只用于评估 trace，仍未输入策略。
- v13 绿色目标 `(2.5, 0.4)` 的 300 步 trace 复测再次进入目标半径，真实最小距离 `0.3348 m`、`terminated_steps=0`，但 `stop_step=None`。预测距离全程仅在 `2.1833–2.6376 m` 之间，真实/预测距离相关系数只有 `0.148`；在真实距离 `<=0.8 m` 的 69 个推理样本中仍预测 `2.1833–2.2021 m`，最大 target-stop 概率仅 `0.0036`。因此已确认是表征/泛化失败，不是继续猜 `0.10/0.14/0.16` 阈值能解决的问题。
- v14 新增与动作主干隔离的 `visual_v2` target encoder，用独立 RGB 空间特征和语言特征预测停止概率/距离；v7 已验证的动作主干完全冻结。stop BCE 只在有目标帧上计算，标签严格定义为离线真实距离 `<=0.8 m`；冻结目标训练采用正负目标样本平衡，并允许只为目标感知使用物理稳定且曾进入目标半径的 DAgger 帧，不使用其失败后的动作作为控制专家。
- `m20_vla_skill_v14_target_visual` 使用 20 条 episode，训练/验证分别为 `2350/775` 帧，训练目标标签为 `not_reached=1029`、`reached=71`；最佳 checkpoint 在第 36 轮，`val_target_loss=0.1170`。训练刚结束时这还只是离线改善，随后才执行下述绿色、红色和蓝色闭环带视频验收。
- v14 验证集目标距离回归相关系数为 `0.963`、MAE 为 `0.116 m`。单一绝对预测距离门限没有三色共同可靠区间：归一化 `0.32` 可让绿/红停车但蓝色不触发，`0.34` 又会让蓝色在真实约 `0.98 m` 处提前形成候选。因此停止确认改为 learned distance 时序门控：最近 20 次预测的最小值低于 `1.7 m`，随后回升至少 `0.03 m`，再使用原有 8 帧/投票确认；该门控只读取模型输出，不读取目标坐标。
- 同一组时序参数完成三色带视频闭环：绿色 `(2.5,0.4)` 第 `109` 步停车，最小/最终距离 `0.4806 m`；红色 `(2.5,0.0)` 第 `134` 步停车，最小/最终距离 `0.1084 m`；蓝色 `(3.0,-0.75)` 第 `161` 步停车，最小/最终距离 `0.7894 m`。三条均 `target_reached=true`、`terminated_steps=0`，停车后分别保持 `111/86/119` 步。不过蓝色横向位移仅 `0.0358 m`，主要是在目标半径边缘停车，说明 learned stop 已通过这三条代表性场景，而任意横向位置的目标导向仍未完成。
- 初版时序门视频曾保存在 `videos/m20_vla_skill_v14/{green_left04_turnaround_v3,red_center_turnaround_v2,blue_right075_turnaround_v2}/`；它们已在严格验收版本通过后删除。删除 10 个与 `best.pt` 重复的 `last.pt` 后，checkpoint 从约 `27 MB` 降为 `14 MB`。

### v14 严格停车与侧向执行协议复测（2026-07-22）

- 验收规则收紧为进入目标 `0.8 m` 半径、停车后连续保持 `100` 个 50 Hz 物理帧、`terminated_steps=0`；`success` 只有同时满足这些条件才为真。trace 现在记录 `executed_skill` 和真实执行的零命令，避免把 stop latch 后仍缓存的预测动作误当成实际动作。
- 公开 M20 专家的蓝色右侧目标协议使用 `wheel_damping=0.2`、`max_yaw_command=0.5` 和 8 帧转向上限。按该协议复测 `(3.0,-0.75)` 后，位移为 `[2.4885,-0.5094] m`、最小/最终距离 `0.5653 m`、`terminated_steps=0`；这证明侧向控制已经恢复，不再只是从目标半径边缘擦过。但 300 步内没有 latch stop，严格结果仍为 `success=false`。
- learned stop 新增绝对预测距离门 `--target-absolute-stop-distance-m`，与原有时序拐点门取或；它只读取 learned visual distance，不读取用于评估的目标坐标。蓝色 v5 末段预测距离已降到约 `0.47 m`，下一次独占回放需用该混合门完成 100 帧保持验收。
- stop latch 后不再调用不可靠的零命令 locomotion expert，而是执行对称站立姿态和零轮速目标，并记录停车后的机身/轮速。直接反向力矩和反向轮速两种主动制动试验都会放大轮速，已从当前实现移除。
- 红色 `red_center_stand_brake_strict_v5` 在较短确认窗口下曾分别停在 `0.8543/0.8060 m`，说明厘米级 PhysX 波动会让边界结果不稳定。将 learned stop 默认确认提高到连续 `14` 帧、15 帧窗口内 `0.90` 投票后，第 `100` 步进入半径、第 `101` 步停车，半径内连续保持 `199` 帧，最低/最终距离 `0.7579/0.7814 m`，`success=true`、`terminated_steps=0`；停车后最终平面速度 `0.00172 m/s`、最终平均绝对轮速 `0.00714 rad/s`。
- 同一严格参数下绿色 `(2.5,0.4)` 第 `115` 步停车，连续保持 `185` 帧，最低/最终距离 `0.4109/0.4296 m`；蓝色 `(3.0,-0.75)` 使用公开专家一致的 `wheel_damping=0.2`、负 yaw 镜像和 `max_yaw=0.5`，第 `303` 步停车，连续保持 `117` 帧，最低/最终距离 `0.4693/0.4740 m`，真实横向位移 `-0.5446 m`。三条均 `success=true`、`terminated_steps=0`，证明当前代表性三色目标的严格导航和 learned stop 已共同通过，但这还不是随机方位 ObjectNav。
- 只保留三条严格成功和 `blue_right075_damping02_v5` 一条代表性 no-stop 失败；删除其余 8 组被替代 v14 JSON/视频后，日志由 `3.4 MB` 降至 `2.7 MB`、视频由 `14 MB` 降至 `12 MB`。训练数据审计为 20 条 HDF5：16 条成功专家、3 条稳定 search、1 条到达目标的 DAgger；共 `192 MB` 且全部位于 2 TB 盘，因此没有误删有效训练数据。

### 随机初始朝向公开专家数据 v1（2026-07-22）

- [record_public_m20_expert.py](scripts/record_public_m20_expert.py) 新增 `--initial-yaw-deg`、`--initial-yaw-jitter-deg` 和 `--seed`。每条 episode 在重置时写入可复现 yaw，并在 HDF5 保存 `initial_yaw_deg/random_seed`；`episode_offset` 会跳过已有随机序列，追加采集不会重复样本。
- 专家仍来自公开 `AI-DA-STC/M20-autonomy-sim` 的 `57 -> 16` ONNX 策略；目标坐标只用于生成成功示范和离线标签，不进入 VLA 输入，全程没有 simulator reward 或 PPO。
- `m20_skill_expert_random_yaw_red_v1` 已采集三条 500 帧成功轨迹，初始 yaw 为 `+10.958/-2.445/+14.344 deg`，最终距离为 `0.7616/0.7903/0.7867 m`，最低根高 `0.4613-0.4643 m`、`terminated_steps=0`。三份 HDF5 的前后 RGB、LiDAR、proprio、state、action 和 expert command 均长度一致、数值有限；三条 H.264 视频均完整解码。
- 新数据约 `48 MB`、视频约 `688 KB`，全部在 2 TB 盘。完成该轮采集时，关键产物合计约 `324 MB`，`62/62` 个 MP4 均为 H.264 且完整逐帧解码失败数为 `0`。随后已按计划训练 v15/v16 并执行未见朝向闭环，结果记录如下。

### 随机朝向 v15-v16 与高层 DAgger 决策（2026-07-22）

- `m20_eval_random_yaw_red_holdout_v1` 固定初始 yaw 为 `-12 deg`，没有进入 v15/v16 的训练 glob。离线评估中，v14 在该 holdout 上的距离相关系数/MAE/stop accuracy 为 `0.757/0.356 m/0.790`，v15 提升到 `0.954/0.171 m/0.966`；v15 最佳 checkpoint 位于第 `7` 轮，`val_target_loss=0.05151`。这说明随机视角数据改善了离线表征，但不能替代闭环验收。
- v15 在同一个 `-12 deg` 红色目标闭环中真实最小距离为 `0.4652 m`，但预测最小距离仍为 `1.0162 m`；420 步中输出 `forward=413/stop=7`，没有形成 stop latch，最终距离 `5.1897 m`，因此 `success=false`。v16 从 v15 初始化，仅开放 command/skill/target 导航头训练；结果几乎相同，真实/预测最小距离为 `0.4651/1.1437 m`、`forward=413/stop=7`、最终距离 `5.1870 m`，仍未停车。两次均 `terminated_steps=0`，失败原因是 learner 闭环状态分布下的距离预测和动作决策偏移，不是摔倒。
- [play_m20_vla_skill.py](scripts/play_m20_vla_skill.py) 已支持 `--initial-yaw-deg` 并将其写入 metrics；[train_m20_vla_skill.py](scripts/train_m20_vla_skill.py) 已支持冻结传感器主干时用 `--train-navigation-heads` 适配 command/skill head。继续增加 epoch 或只调输出头已经被 v15/v16 证伪。
- [record_public_m20_expert.py](scripts/record_public_m20_expert.py) 当前只加入了 high-level skill DAgger 的参数校验、模型类型和 canonical command 转换脚手架，尚未把 learner command、公开专家概率干预和 HDF5 标签接入主循环，因此还不能用于采集，也没有生成或宣称 DAgger skill 数据。
- 下一步固定为：完成高层 DAgger 主循环；在 learner `-12 deg` 状态分布上用公开 M20 专家以 `0.25-0.5` 概率干预 command，始终保存当前状态的专家 command/action 标签；验证 HDF5、H.264 视频和稳定性后训练 v17。加入 `-12 deg` 数据后，泛化验收必须改用未参与训练的 `+20 deg` 或 `-8 deg`。
- 最新存储审计：datasets `254 MB`、public experts `56 MB`、checkpoints `20 MB`、videos `14 MB`、logs `3.4 MB`。全盘实际存在 `63` 个 MP4，编码检查为 `63/63 H.264`，FFmpeg 完整逐帧解码失败数为 `0`。

### v17-v19 DAgger 结果与旧路线终止（2026-07-23）

- high-level skill DAgger 已完整接入：采集器保存 learner command、专家干预标记、canonical skill 以及公开 M20 专家的 command/action 标签；训练器支持 DAgger 固定进入训练 split、只训练导航 heads 和冻结 target head。新增 HDF5 已检查长度一致、数值有限，回放视频均为 H.264。
- v18 在参与 DAgger 适配的红色目标 `-12 deg` 初始朝向上通过窄场景验收：第 `81` 步进入 `0.8 m` 半径，第 `114` 步停车，最小/最终距离 `0.5355/0.6716 m`，停车后保持 `306` 帧，最低根高 `0.4610 m`，`terminated_steps=0`。代表视频为 `videos/m20_vla_skill_v18/red_center_initial_yaw_m12_last_confirm3_v2/m20-vla-skill-step-0.mp4`。
- v19 在未见 `+20 deg` 初始朝向上虽然第 `198` 步经过目标、最小距离 `0.2497 m`，但从未停车，最终距离扩大到 `3.2797 m`，严格 `success=false`。这说明 DAgger 只改善了局部状态分布，未解决随机方位泛化。
- 当前网络只是小型 CNN + UTF-8 byte embedding + 手写 `forward/left/right/search/stop/jump` 技能 head。训练集没有 jump 正标签，所谓 search 也只是 canonical command；它没有预训练语言语义、开放词汇物体理解、多房间记忆或 parkour 能力。因此旧模型不再称为 VLA 成果，v18 仅归档为 M20 低层/单目标可复现基线，v19 归档为代表性泛化失败。
- 不再围绕 v19 继续调停车阈值或追加相同单色方块 episode。后续验收必须报告多 episode 任务成功率；单次成功、离线 loss、技能准确率或“曾经过目标”均不能作为原始任务完成证据。

### 官方 SmolVLA 本机运行基线（2026-07-23）

- 主线切换到公开预训练 VLA `lerobot/smolvla_base`，固定 revision `c83c3163b8ca9b7e67c509fffd9121e66cb96205`。模型为 `450,046,176` 参数，输入多视角 RGB、状态和自然语言，使用 flow matching 输出连续 action chunk；官方权重公开且非 gated，LeRobot 采用 Apache-2.0。
- 权重和运行环境均放在 2 TB 盘。checkpoint 文件为 `907 MB`，已按 Git LFS 声明验证 SHA-256：`7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb`。SmolVLM2 tokenizer/processor 固定 revision `7b375e1b73b11138ff12fe22c8f2822d8fe03467`，没有重复下载 VLM 权重。
- LeRobot `0.4.4` 使用独立 overlay 环境 `M20ProVLA/envs/m20pro-smolvla`，只读复用原 `m20pro-vla` 中已验证的 PyTorch `2.7.0+cu128` 和 CUDA 库，没有替换 Isaac Sim 的 torch。环境入口为 [activate_smolvla_env.sh](scripts/activate_smolvla_env.sh)。
- [check_smolvla.py](scripts/check_smolvla.py) 已完成严格 checkpoint 加载和三视角合成输入 smoke：输出 action shape `1 x 50 x 6`，数值全部有限；正式 wrapper 复测模型加载约 `19.68 s`，一次 10-step flow inference 约 `0.823 s`，RTX 3060 峰值分配显存约 `1214 MiB`。该结果只证明预训练 VLA runtime 可行，不代表已有导航能力。
- 新的高层动作接口固定为 6 维连续 action chunk，承载 M20 的速度、转向、停止/搜索/跨越意图；16 维关节控制继续由已验证的 learned locomotion/未来 parkour expert 执行。目标是让 VLA 学视觉语言决策，不让它重新学习基础关节稳定性。
- 新增 [m20pro_vla_eval_v1.yaml](configs/m20pro_vla_eval_v1.yaml)：训练至少覆盖 `8` 个场景、`12` 类物体和 `24` 个指令模板；测试分别包含 seen/unseen object 的独立场景和指令，并单独统计 visible ObjectNav、隐藏物体主动搜索、place navigation 和 1 m 障碍。所有 episode 必须保存 H.264 视频和 JSON，offline loss 不作为验收指标。
- 下载分片在 SHA-256 通过后已清理，最终 SmolVLA 模型目录约 `865 MB`、processor 约 `4.8 MB`、overlay 环境约 `651 MB`。当前其他产物为 datasets `322 MB`、checkpoints `29 MB`、videos `18 MB`、logs `6.1 MB`、public experts `56 MB`，均位于 2 TB 盘。
- 对视频目录重新做了全量审计：实际 `77/77` 个 MP4 均为 H.264，并用打包 FFmpeg 从首帧到末帧完整解码，失败数为 `0`。这替代此前只覆盖 `63` 个文件的历史结论。

### SmolVLA 数据就绪审计（2026-07-23）

- 用户明确否决将当前展示当作项目成果。该判断是正确的：当前只完成了预训练 SmolVLA 运行时 smoke，没有 M20 微调 checkpoint，也没有未见场景闭环成功率，不能称为导航成果。
- 新增 [audit_m20_smolvla_data.py](scripts/audit_m20_smolvla_data.py) 和 [audit_m20_smolvla_data.sh](scripts/audit_m20_smolvla_data.sh)，对所有 HDF5 的长度、dtype、有限值、高层动作标签、时间戳、传感器对齐、场景/物体/指令覆盖、隐藏搜索、障碍 LiDAR 和跳跃数据做机器可读审计。结果位于 `logs/m20_smolvla_data_audit_v1.json`。
- 磁盘实际有 `29` 条、`13,650` 帧 HDF5；其中 `23` 条标记成功，`1` 条为 holdout，`25` 条满足宽松的候选训练条件（包含 `3` 条只会原地扫描的 partial search）。这些数据只覆盖红/绿/蓝 `3` 类色块和 `7` 种完全一致的指令文本，没有带 `scene_id` 的室内场景。
- 三条 `寻找目标` 轨迹的 `target_reached=true` 是“场景无目标”时的初始化值，不是找到隐藏物体。收紧审计口径后，隐藏目标搜索成功为 `0`，障碍 LiDAR episode 为 `0`，jump episode 为 `0`。
- 旧采集器中 RGB/LiDAR/proprio/动作在动作前采样，但 45 维 `state` 在动作后写入；且旧 HDF5 没有显式 `timestamp/frame_index`。因此时序对齐通过数为 `0/25`，不会在转换时伪造旧数据的对齐标记。
- 当前 RayCaster 只将 `/World/ground` 列为 mesh，目标又是无碰撞色块。部分小于 `20 m` 的返回是姿态变化时射线命中地面，不能证明有障碍感知。未来数据必须将房间墙体和障碍几何显式加入 LiDAR mesh。
- [record_public_m20_expert.py](scripts/record_public_m20_expert.py) 已将 45 维 `state` 移到动作前采样，新增 `timestamp/frame_index`，并写入 `sensor_alignment=pre_action` 和 `lidar_mesh_scope=ground_only`。旧数据保持原样，新采集才能通过对齐门。
- 当前审计的 `7/7` 数据门均未通过，`ready_for_smolvla_finetune=false`。因此暂停“直接转换旧轨迹并微调”，先构建真正的随机室内场景、障碍可见 LiDAR 和成功专家数据。

### 室内 visible ObjectNav 数据链路 v1（2026-07-23）

- 新增 [m20_visible_objectnav_scenarios_v1.json](configs/m20_visible_objectnav_scenarios_v1.json) 和可重建脚本 [build_m20_objectnav_manifest.py](scripts/build_m20_objectnav_manifest.py)。清单 SHA-256 为 `0fefbb32de9650cdc62fd699a896f614404c4d3d542edaedacbc50f05c264a74`，包含 `8` 个训练布局、`2` 个验证布局、`2` 个可见目标测试布局、`12` 类 NVIDIA Isaac Sim YCB 物体和 `48` 个中英指令模板。其中训练/验证/测试模板分别为 `24/12/12`，集合互不重叠；共有 `156` 个确定性 episode 定义。
- 室内采集器从单地面 `RayCaster` 切换为 Isaac Lab `MultiMeshRayCaster`。smoke 日志确认它实际读取了地面 `2` 个 mesh、房间墙体/家具 `7` 个 mesh 以及 YCB 目标 `1` 个 `8006` 顶点/`16384` 面的 mesh。正式样本的 LiDAR 距离范围为 `2.21-5.89 m`，非最大量程返回比例 `1.0`，不再是射线打地面造成的伪障碍证据。
- 修正了历史采集器的后相机朝向：旧配置只把相机移到机身后方，却没有旋转 `180 deg`，因此前后两帧实际同向。新配置下首帧目标像素占比为前相机 `0.03483`、后相机 `0.0`，语义分割已确认目标确实在前视野，不再只相信 manifest 的声明。
- 每帧新增 `high_level_action[6] = [forward, lateral, yaw, stop, search, parkour]`、目标可见像素比例、统一动作前 `timestamp/frame_index`；HDF5 同时保存 `scene_id/split/object_category/object_source/object_usd_path/instruction_template_id/manifest_sha256`。物理目标位姿只供示范专家标注，HDF5 明确写入 `inference_uses_privileged_target_pose=false`。
- 室内低层执行器保留公开 M20 ONNX 的腿部稳定输出，四轮改用 PhysX implicit velocity drive。这是专家数据生成的低层执行，不是 VLA 结果；转换前实测的显式 DC motor 轮速会在零命令下达到约 `+/-50 rad/s`，是早期停车后飘移/跌倒的直接原因。
- pilot v1-v5 都保留为失败证据：原生轮输出在到达后跌倒；高停车阻尼会振荡；只置零会稳定但漂移；闭环制动在旧执行器上会发散。换用隐式速度驱动并修正航向反馈后，pilot v6 首次严格成功。
- 正式 `train_0000` 数据已通过：第 `120` 帧进入 `0.8 m` 半径，第 `132` 帧完成低速确认并锁定停车，半径内连续保持 `188` 帧，最小/最终距离 `0.7534/0.7876 m`，最终平面速度 `0.0341 m/s`，路径长 `3.0174 m`，最低根高 `0.4372 m`，`terminated_steps=0`。`320/320` 帧视频为 H.264 High/yuv420p/50 FPS，已完整解码。
- 正式采集命令为 `./scripts/collect_m20_visible_objectnav.sh train_0000`；它默认拒绝覆盖已有 episode。正式 HDF5/JSON 位于 `datasets/m20_visible_objectnav_v1/train/`，视频位于 `videos/m20_visible_objectnav_v1/train/`。
- 主审计现为 `31` 条/`14,290` 帧，其中新规范 `smolvla_candidate=2`、`smolvla_eligible=2`。新数据的“候选数据全部时序对齐”、“场景几何进入 LiDAR”和“6 维动作完整”三项已通过；场景/物体/模板覆盖为 `2/8`、`2/12`、`2/24`，所以 `ready_for_visible_objectnav_finetune=false`，未开始 SmolVLA 微调。
- `train_0009` 复测（第二个布局、第二类 YCB 物体、初始朝向 `-23.206 deg`）已严格通过：第 `158` 帧进入 `0.8 m` 半径，最终距离 `0.7796 m`，路径长 `2.8926 m`，最低根高 `0.4891 m`，`terminated_steps=0`，`success=true`；视频为 `320` 帧 H.264。该复测确认 `train_0000` 的专家参数不是单场景特例，但也暴露出当前工作仍只是“成功专家数据生成”，不是 VLA 学习结果。
- 批次复测 `train_0002`（第三个布局、主厨罐头）和 `train_0016`（第一个布局、糖盒）均严格成功：前者第 `100` 帧到达，最终距离 `0.7451 m`、路径 `2.9238 m`、最低根高 `0.4722 m`；后者第 `114` 帧到达，最终距离 `0.7964 m`、路径 `2.5035 m`、最低根高 `0.4308 m`。两条均 `terminated_steps=0`、无专家干预、视频 `320` 帧 H.264。
- 这批完成后的审计为 `33` 条/`14,930` 帧；新规范候选 `4` 条、可见 ObjectNav 覆盖为场景 `3/8`、物体 `3/12`、模板 `4/24`，所有候选仍通过时序对齐、LiDAR 几何和 6 维动作门。隐藏搜索、1 米障碍 LiDAR 和 jump 仍为 `0`，所以两个 `ready_for_*_finetune` 仍为 `false`。

### 当前阶段判断与下一步（2026-07-23）

- 用户对“长时间运行后只有一条展示视频”的不满意是合理的：当前还没有 SmolVLA 微调 checkpoint，也没有未见场景闭环成功率。两条规范样本只能验证采集链路和低层专家稳定性，不能作为任务完成证据。
- 当前审计摘要：`33` 条 HDF5、`14,930` 帧；新规范候选 `4` 条且全部通过时序/数值审计；可见 ObjectNav 覆盖为场景 `3/8`、物体 `3/12`、模板 `4/24`；隐藏搜索成功 `0`，1 米障碍 LiDAR `0`，jump 标签 `0`。因此 `ready_for_visible_objectnav_finetune=false` 和 `ready_for_smolvla_finetune=false` 均保持不变。
- 下一轮不再做单条手工展示，改为按 manifest 自动补齐训练覆盖。每完成一批就运行审计，只保留 `success=true`、视频可解码且候选门通过的 episode；覆盖达到 `8/12/24` 后再做 LeRobot v3 转换和 SmolVLA 微调。
- 批量入口已加入 [collect_m20_visible_objectnav_batch.sh](scripts/collect_m20_visible_objectnav_batch.sh)。它逐条启动无头采集、把完整输出写入 2 TB 盘的批次日志，并在每条之后运行审计；失败 episode 不会被静默混入训练集。

v14 绿色目标复现命令（显式无头并录制视频）：

```bash
source scripts/activate_vla_env.sh
python scripts/play_m20_vla_skill.py --headless --video \
  --checkpoint /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/checkpoints/m20_vla_skill_v14_target_visual/best.pt \
  --task-text "到绿色方块去" --target-color green --target-x 2.5 --target-y 0.4 \
  --steps 300 --target-distance-threshold 0.34 \
  --target-turnaround-window 20 --target-turnaround-rise-m 0.03 \
  --min-forward-command 0.08 \
  --video-dir /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/videos/m20_vla_skill_v14/green_left04_strict_v6 \
  --metrics /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/logs/m20_vla_skill_v14_green_left04_strict_v6.json
```

### v2 数据链路修复与重采（2026-07-23）

- 复盘 SmolVLA `v1` 闭环失败后确认两个根因：旧数据的 stop 标签在目标离开相机之后才开始，且 32 维输入状态直接取了世界坐标，形成场景位置捷径。旧 `v1` checkpoint/数据继续保留为失败基线，不再作为成果。
- 新增 v2 采集协议：前后相机采用 `22 deg` 下视、`18 mm` 焦距；目标距离 `1.20 m` 起写入高层 stop 标签，但提前 stop 不切换底层轮速，实际停车仍由 `0.8 m` 成功半径控制，避免公开 M20 locomotion 专家被预触发轮速切换弄倒。
- v2 新增 `observation/smolvla_proprio (8)`，内容为机身线速度 3、机身角速度 3、投影重力 xy 2；绝对世界位姿不进入 SmolVLA 状态。再拼接 24 个 LiDAR sector minimum，形成 32 维输入。
- v2 pilot `train_0000` 严格通过：`success=true`、`terminated_steps=0`、第 71 帧 stop 标签起点目标像素 `26.2%`、目标最大可见比例 `43.5%`。单进程重采 `train_0001` 和 `train_0002` 也通过，后者第 100 帧到达、最终距离 `0.745 m`、最低根高 `0.472 m`。
- 一次并发 Kit 运行损坏了 `train_0002` 的 HDF5/MP4；坏文件已移到 `datasets/m20_visible_objectnav_v2/corrupt_runs/20260723_concurrent_kit_train_0002/`，不进入审计或训练。后续批处理改为单进程。
- 审计器升级为 `m20pro_smolvla_data_audit_v2`，现在会拒绝目标在 stop 起点不可见、缺少 invariant proprio 或不可读 HDF5 的样本；转换器已修复未定义 `PROPRIO_INDICES`，2 条 v2 smoke 已成功转换为 640 帧、32 维 state、6 维高层 action。
- 当前 v2 有效训练样本为 `21` 条、`6,720` 帧，覆盖 `8/8` 个训练场景、`9/12` 类物体和 `20/24` 个指令模板；所有 21 条候选均通过时序对齐、LiDAR 几何和 6 维动作审计，尚未开始正式 SmolVLA 训练。

### 历史状态快照（2026-07-23 20:38，中国标准时间）

- 单进程批量采集仍在运行：`train_0069` 正在 Isaac Sim headless 中采集；不要启动第二个 ObjectNav Kit 进程。
- `train_0067`、`train_0068` 已完成并通过：均 `success=true`、`terminated_steps=0`，并生成可读 H.264 视频、HDF5 和 JSON 指标。`train_0067` 的最终目标距离为 `0.7497 m`，`train_0068` 为 `0.7473 m`。
- 最近一次 v2 审计（`train_0068` 后）：总库存 `22` 条（其中 1 条为已归档损坏文件），有效候选 `21` 条、`6,720` 帧；场景覆盖 `8/8`，物体类别 `9/12`，指令模板 `20/24`。
- 当前门禁：`ready_for_visible_objectnav_finetune=false`、`ready_for_smolvla_finetune=false`。尚未有 SmolVLA 微调 checkpoint，也没有把专家数据当作 VLA 成果。
- 下一个可验证里程碑是补齐 `12/12` 物体和 `24/24` 指令模板后，执行正式 LeRobot 转换、从头微调 v2 SmolVLA，并用未见场景视频验收；隐藏搜索和 1 m 障碍跳跃仍是后续独立阶段。

### SmolVLA trim20 首条真实闭环回放（2026-07-23 21:31）

- 修正后的 LeRobot 数据已完成 `29` 条、`2831` 帧的 stop 尾段裁剪；stop 标签占比约 `20.49%`。`smolvla_objectnav_v2_trim20_1000/checkpoints/001000/pretrained_model` 已完成 `1000` steps 微调。
- 独立离线审计通过：`mean_action_mae=0.0407`、`forward_mae=0.0453 m/s`、`yaw_mae=0.0326 rad/s`、`stop_accuracy=0.875`。这些指标只说明动作拟合，不是闭环成功率。
- 首条 headless 闭环 `train_0000` 使用 `--smolvla-stop-threshold 0.4`、动作保持 `10` 步、stop 连续确认 `5` 次。结果：目标第 `91` 步进入半径，最小距离 `0.5944 m`，最终距离 `0.6858 m`，最低根高 `0.4912 m`，`terminated_steps=0`，但 `smolvla_stop_latched=false`、`command_ok=false`、`success=false`。机器人确实接近了目标，但没有停车，随后穿过目标区域。
- 回放中的 stop 通道范围只有约 `0.000-0.194`，因此阈值 `0.4` 在该闭环状态分布中不可能触发。该现象与离线阈值扫描的 `0.35-0.45` 推荐区间不一致，说明存在训练集/闭环观测分布或 stop 标定偏移，不能继续把离线阈值当作闭环结论。
- 视频位于 `videos/smolvla_objectnav_replay/train_0000/episode_train_0000.mp4`，已用 FFmpeg 完整解码通过；Isaac Sim 仅出现常规插件和 `torchcodec` fallback 警告，没有物理异常。

### 当前阶段判断与下一步（更新）

- 当前成果是“数据链路、SmolVLA 微调和低层运动稳定性已具备，首条视觉闭环尚未成功”，不能宣称完成 visible ObjectNav。
- 立即任务改为记录至少 `5` 条闭环的完整 stop 预测曲线，并核对相同场景的训练帧与回放帧输入/归一化；只在不使用目标坐标的前提下重新标定 stop 阈值和确认逻辑。
- stop 校准通过后，才汇总未见场景/目标的多 episode 成功率；隐藏物体主动搜索、place navigation 和 `1 m` 障碍跳跃仍未开始，当前门禁继续为 `false`。

### v3 DAgger SmolVLA 训练与 learner-only 闭环验收（2026-07-23 23:05）

- 联合数据已完成审计：`34` 条 episode、`3330` 帧，覆盖 `8/8` 场景、`12/12` 物体类别和 `24/24` 指令模板；时序、LiDAR 场景几何和 6 维高层动作门全部通过。隐藏目标搜索、1 m 障碍 LiDAR 和 jump 标签仍为 `0`，没有提前进入后续分支。
- `smolvla_objectnav_v3_dagger1_1000` 已完成 `1000/1000` steps。最终模型为 `checkpoints/001000/pretrained_model`，训练配置使用 v3 DAgger 数据集；该 checkpoint 已通过加载校验。
- 新增/修正回放入口：接受 `001000` 父目录或 `pretrained_model` 目录，默认使用 v3 stats 和 `stop_threshold=0.4`。learner-only 回放未启用 `--smolvla-dagger-labels`，目标坐标只用于离线评估，不能触发制动。
- 五条验收场景中四条完整运行：`train_0000/0009/0027/0073` 的严格成功率为 `0/4`，目标接近率为 `4/4`，停车 latch 为 `0/4`，跌倒率为 `4/4`；最低根高均约 `0.070 m`，说明持续前进命令把低层执行器推入失稳。`train_0090` 在 USD 姿态矩阵 `OrthogonalizeBasis did not converge` 后无进展，已中止并记为无效回放。
- 四条完整回放的 JSON/H.264 视频均已生成于 `logs/smolvla_objectnav_replay/` 和 `videos/smolvla_objectnav_replay/`。代表结果：`train_0000` 最小目标距离 `0.537 m`、第 `84` 步进入目标区但未停车；`train_0027` 最小距离 `0.054 m` 仍未停车。
- 新 checkpoint 的离线动作审计（128 帧）为 `mean_action_mae=0.0311`、`forward_mae=0.0370`、`yaw_mae=0.0415`、`stop_accuracy=0.9375`；这只说明 held-out 动作拟合，不能当作闭环成功。
- 同一回放帧的多次 SmolVLA 推理显示 flow-matching 采样具有明显随机性：`train_0000` 第 100 帧 stop 输出在不同采样中约 `-0.05~0.99`，单次回放采样可能只有 `0.077`，而停车逻辑要求连续 5 次超过 `0.4`。因此当前主要问题是随机 action chunk 导致 stop 不稳定及其后的持续前进，不是简单增加训练步数。
- 当前结论：v3 仍不能称为 visible ObjectNav 成果；应先做固定/集成推理和动作安全包络实验，稳定停车且不跌倒后再汇总多场景成功率。隐藏搜索、place navigation 和 1 m 跳跃继续冻结。

### SmolVLA ensemble 闭环与首条 holdout（2026-07-24）

- 回放改为每次查询执行 `4` 个可复现 flow-matching 样本：连续动作取均值后平滑、限幅，stop 使用 ensemble 投票；HDF5 同时保存 stop score、votes 和实际执行命令。验收仍为 learner-only，不启用 DAgger、目标位姿制动或专家干预。
- `train_0000/0009/0027/0073` 曾按“到达、停车、未触发 fall”口径得到 `4/4`，但用户视频复核发现初始强烈抖动、机身低趴且左前腿不能正常承重。该 `4/4` 结论现已撤销：旧指标没有检查横滚、角速度、腿对称性和关节目标跳变，不能称为严格成功。
- 该 `4/4` 只能证明训练场景回放闭环通过，不能证明泛化。首条未参与训练的 `validation_0000` 严格失败：无跌倒，最小/最终目标距离 `1.0362/1.0778 m`，第 `110` 步 learner stop latch，未进入 `0.8 m` 成功半径。
- 失败根因是 stop 监督从约 `1.2 m` 开始，而专家仍继续慢速接近至 `0.8 m`；旧回放把 stop 预测直接解释为立即停车，导致 holdout 提前制动。当前正在加入两阶段 stop 状态机：连续确认后先进入 `60` 帧、最大前进速度 `0.18 m/s` 的受限 approach，再锁定停车。该逻辑不读取目标位姿。
- 截至本次状态核对，approach 主逻辑、实际执行命令记录和参数 HDF5 attrs 已进入未提交工作树，但 episode JSON/armed-step 诊断、回放脚本显式参数和新的 holdout 回放尚未全部完成，不能把修复写成已通过。当前无 Isaac Sim、训练或回放进程运行。
- 下一门禁是先重跑 `validation_0000`，再跑 `validation_0009`、`test_visible_0004`、`test_visible_0011`；之后扩展到至少 `20` 条 holdout 并汇总成功率。隐藏目标搜索、place navigation 和 `1 m` 障碍跳跃仍未开始。

### 启动抖动与左前腿姿态修复（2026-07-24）

- 原始 HDF5 证明异常在 VLA 第一个动作之前已经发生：旧回放首帧横滚约 `7.8-8.6 deg`，机身角速度约 `9.7-10.0 rad/s`。根因是 recorder 的 warmup 只在每个 `50 Hz` 控制帧开始写一次显式 PD 力，后续三个 `200 Hz` PhysX 子步沿用过期力矩；原始稳定回放则在每个物理子步重新计算并写力。
- warmup 已改为每个 `5 ms` 物理子步执行 `scene.write_data_to_sim()`。对照 `train_0000` 的 80 帧低层 smoke：最低根高 `0.5148 m`、高度标准差 `0.0008 m`、最大横滚 `0.69 deg`、启动最大角速度 `0.177 rad/s`、启动腿对称误差 `0.105 rad`，`posture_ok=true`。
- SmolVLA 回放入口默认使用与公开 ONNX 更一致的官方 MJCF USD，恢复公开策略的完整 `16` 维腿轮协同输出；会破坏腿轮同步的几何轮速覆盖和室内隐式轮速驱动不再作为回放默认值。几何覆盖仍保留为专家数据兼容/诊断选项。
- 停车不再运行已证实不稳定的“零速度 ONNX”动作，而是切换到对称默认站姿并锁轮；制动确认期间保持最后一帧稳定腿目标。HDF5/JSON 新增启动横滚俯仰、角速度、根高方差、腿对称误差和关节目标跳变，`success` 现在必须同时通过 `startup_posture_ok` 与 `posture_ok`。
- 修复后的完整 learner-only `train_0000` 已通过：第 `116` 帧进入 `0.8 m` 半径，第 `170` 帧 SmolVLA stop latch，保持 `150` 帧；最小/最终距离 `0.1740/0.1948 m`，最终速度 `0.00091 m/s`，最低根高 `0.5148 m`、最大横滚 `0.50 deg`、启动最大角速度 `0.102 rad/s`、最终腿对称误差 `0.0066 rad`，`terminated_steps=0`、`posture_ok=true`、`success=true`。
- 新视频位于 `videos/smolvla_objectnav_replay_v5_posture/train_0000/episode_train_0000.mp4`，H.264 已完整解码。它只恢复一条训练分布内场景的姿态有效性；`train_0009/0027/0073` 和 holdout 尚未按新门禁重跑，不能恢复旧的 `4/4` 或宣称泛化成功。
- 图形回放新增低负载预设：交互视口降为 `960x540`，关闭阴影、反射、间接光、环境光遮蔽、半透明和 sampled lighting，DLSS 使用 Performance，限制单 GPU；前后相机与 MP4 任务分辨率不变。可通过 `M20PRO_KIT_ARGS` 覆盖该预设。
- 用户已在低负载有头回放中人工复核 `train_0000`：启动不再抖动，四肢姿态与运动均正常。该人工检查与姿态门禁一致，因此这条 episode 的低层执行问题可视为已修复；仍需用其他 episode 排除单场景偶然性。
- 当前回放确实使用已微调的 SmolVLA checkpoint 做 learner-only 闭环：模型收到文本“到主厨罐头那里去”、前后 `256x256` RGB、8 维机身状态和 24 维 LiDAR 扇区最小值，不接收目标坐标，并输出前进、侧移、转向与 stop。`train_0000` 的罐头从起点可见，最大目标像素占比约 `21.8%`，模型最终接近并停车。但启动入口仍以 manifest episode 选择场景和文本，并非实时语音/自由文本接口；隐藏目标主动搜索、任意同义表达、陌生布局和多目标消歧尚未通过，不能宣称“随口说去罐头处就能在任意房间找到罐头”。
- 新姿态门禁下的第二条训练分布复测 `train_0009` 使用另一布局、苏打饼干盒、初始朝向和文本“去有苏打饼干盒的位置”。姿态继续通过：最低根高 `0.5148 m`、最大横滚 `0.48 deg`、启动最大角速度 `0.129 rad/s`、无跌倒；320 帧 H.264 视频完整可解码。任务没有通过：第 `123` 帧曾进入目标区，最小距离 `0.0505 m`，但模型到第 `210` 帧才给出连续 stop 信号，两阶段 approach 又延迟至第 `280` 帧锁停，最终距离已回到 `0.9976 m`，`command_ok=false`、`success=false`。当前修复后统计为姿态 `2/2`、完整任务 `1/2`。
- 现有证据也尚不能单独证明模型完成了可靠的语言-目标绑定：每条 episode 只有一个指定目标，文本由 manifest 与场景绑定，策略可能部分依赖视觉/运动相关性而忽略具体名词。下一阶段需要同一场景同时放置多个候选物体并交换指令，检查机器人是否随文本选择不同目标；在该反事实门禁通过前，只能称为“接入语言条件的 SmolVLA 可见目标闭环”，不能称为自由自然语言 ObjectNav。

### 多场景 stop 语义与原始画质复测（2026-07-24）

- 补齐 v5 `train_0000/0009/0027/0073` 后，四条姿态均通过、目标均曾进入半径，但完整任务只有 `1/4`。三个失败场景都在第 `220` 帧形成连续 stop 证据，距离目标仍为 `0.478-0.666 m`；固定再前进 `60` 帧才锁停会把机器人带离目标，是确定的后处理语义错误。
- v6 改为 4 个 flow-matching 样本至少 2 票、连续 2 次查询后立即服从模型 stop，删除固定 approach，并把 `smolvla_stop_armed_step` 写入 HDF5/JSON。低画质运行的 8 条初步结果为：训练 `4/4`、验证 `1/2`、可见测试 `0/2`、合并 holdout `1/4`，姿态 `8/8`、false stop `3`，全部视频可解码。汇总器已按 `train/validation/test_visible` 分组，避免总体成功率掩盖泛化失败。
- 随后确认低画质预设中的阴影/反射等 RTX 开关是全局设置，虽然相机和 MP4 分辨率不变，像素分布仍可能改变。回放入口现已分流：有头人工展示使用 `960x540` 低画质；正式 `--headless` 测试保留训练时渲染设置且不创建窗口。v6 低画质数字保留为诊断，不作为最终泛化指标。
- 原始传感器画质 v7 已完成 9 条正式回放，汇总位于 `logs/m20_smolvla_objectnav_replay_v7_full_render_benchmark_summary.json`：总体 `5/9`，训练 `4/4`，验证 `1/2`，可见测试 `0/3`，合并 holdout `1/5`；false stop 为 `4`，姿态 `9/9`，跌倒 `0`，learner-only 与 320 帧 H.264 视频门禁均为 `9/9`。`test_visible_0011` 最小/最终距离 `1.5636/1.9521 m`，第 `220` 帧 false stop。该 checkpoint 的 visible ObjectNav 泛化门禁明确失败。
- 曾按用户建议准备服务器 `genghaowei@192.168.3.136`：最小运行集 `3.7 GB` 位于 `/private/genghaowei/M20ProVLA_server`，模型与 HF 大文件 SHA-256 一致，Python/Isaac/SmolVLA 导入通过。但账号无 `/dev/dri` 的 video/render 权限，带相机 rendering experience 在首帧前崩溃，随后四张 4090 又均被其他用户占用；未取得有效服务器 episode 或加速倍数。用户决定暂不使用服务器，后续继续本机顺序测试。

### v4 stop 语义修复与本地重训准备（2026-07-24 16:30）

- 对 v3 的 `3330` 个训练帧做了几何复核：`582` 个 stop 正样本中有 `580` 个位于 `0.8 m` 成功半径之外，并同时带有非零前进命令；29 条 expert episode 的首 stop 距离为 `1.159-1.200 m`。根因是采集包装器设置 `stop_pretrigger_radius=1.20`，采集器又把 pretrigger 当成语义 stop。
- 旧 converter 从源数据首 stop 后只保留 20 帧，因此恰好裁掉了真正进入 `0.8 m` 的监督。现在 converter 先用 pre-action `state[:2]` 与 `target_xy` 生成 `distance<=0.8` 的锁存 stop，再执行 tail 裁剪；目标坐标只用于离线监督修复，策略输入仍只有双目 RGB、机身状态和 LiDAR。
- 采集端 stop 标签已改为只服从 `target_reached`，wrapper 显式使用 `success_radius=0.8`、禁用 legacy pretrigger；语义 stop 仍严格锁存于 `success_radius`，只在停车后的静止保持与最终位置判定中使用显式 `0.02 m` 物理滞回。DAgger prefix 会按同一半径重算 stop 与 reached 元数据；源数据审计会阻止带早停/stop+运动冲突的数据直接训练，同时允许 canonical converter 做受控迁移。
- 第一版重标数据集 `datasets/m20_visible_objectnav_lerobot_v4_stop08` 曾生成 `34` 条、`4168` 帧并修正 `838` 个提前 stop，但新增可观测性审计发现：`582` 个 retained stop 帧中有 `270` 个前后相机均看不到目标，30 条含 stop 的 episode 中有 8 条在 canonical 首 stop 当帧目标不可见。该数据集现已否决，只保留作诊断，不允许进入训练。
- 相机 A/B 使用最差的 `train_0027` 验证了修复方向：焦距由 `18 mm` 降到 `12 mm` 后，首次进入 `0.8 m` 时的目标像素占比从 `0` 提升到 `7.22%`，20 个 stop tail 帧全部可见。stop 标签仍严格从 `0.8 m` 开始；最终静止位置只增加显式 `2 cm` 物理滞回，处理已在半径内稳定保持 100 帧后约 `4 mm` 的数值漂移，不恢复旧的 `10 cm` 宽松门限。
- converter 现在要求 audit 根目录与输入一致、逐 HDF5 SHA-256 一致、所有候选均通过或可迁移且目录中没有未审计 HDF5；全量预检通过后才在临时目录构建，成功后原子替换旧输出。训练入口固定保护 dataset/checkpoint/output 参数，并强制检查 `0.8 m` canonical stop、可见 stop、统一 `12 mm` 相机和精确 29 条 episode。
- 启动前复核又补齐了 fail-closed 门禁：批采不再只看进程退出码，而是逐条要求 `success=true`、`smolvla_eligible=true`、无审计错误、`12 mm` 相机、`0.8 m` stop、`0.02 m` 最终滞回和完整 20 帧可见 stop tail。整批、converter 和训练入口三层都要求精确 29 个预期 scenario ID 以及 8 场景/12 物体/24 模板覆盖；converter 与采集共用排他锁，并绑定首次读取的 audit 哈希，避免并发更新污染 manifest。
- 首次重采 `train_0000` 在第 124 帧严格进入 `0.8 m`，最低/最终距离 `0.7487/0.8074 m`、最终速度 `0.00026 m/s`，但停稳后的物理回弹使严格 `0.8 m` 保持计数只有 `67/100`，因此被新门禁正确拒绝。修复后首次到达和 stop 标签仍只在 `<=0.8 m` 锁存，但已停车的稳定保持使用 `<=0.82 m` 滞回外半径。覆盖重采后该条以 `192/100` 保持通过，`success=true`、姿态通过、无跌倒；stop 首帧与 canonical 首次到达同为第 124 帧，零标签错位、零 stop/运动冲突、20 帧 stop tail 全可见，H.264 视频 320 帧完整解码。

## 待办路线

1. 用统一 `12 mm` 相机、canonical `0.8 m` stop 和 `2 cm` 最终静止滞回，在本机重采 29 条干净 expert episode；只有 stop tail 全部可见、哈希绑定审计和 8 场景/12 物体/24 模板门禁全通过才转换为 v5 数据集。
2. 在本机从原始 SmolVLA base 训练 v5 camera12-stop08 checkpoint；使用显式短 warmup 和覆盖完整训练长度的 decay，不从带早停偏置的 v2/v3 checkpoint 续训。
3. 先做离线动作/stop 门禁，再用同一 `12 mm` 原始传感器画质复跑训练、验证和可见测试场景，继续保留逐 episode H.264 视频；只有 holdout 明显超过当前 `1/5` 才扩展测试规模。
4. 新 checkpoint 扩展到至少 `20` 条未参与训练的 visible ObjectNav episode，报告成功率、姿态通过率、跌倒率、false-stop、失败类型和逐 episode 视频。
5. 增加同场景多物体与指令互换的反事实语言门禁，并在 manifest 入口之外接入运行时文本参数；语音只作为语音转文字前端，VLA 核心继续接收文本。
6. 然后加入目标位于侧后方、被遮挡和跨房间的成功轨迹，训练主动 search 和记忆，按 discovery rate、false-stop rate 和完整任务成功率验收。
7. 接入可验证的公开四足 parkour checkpoint。若没有 M20 的 1 m 障碍专家，由于项目仅做仿真且不要求沿用 M20 动力学，切换到有公开 checkpoint 的 Go1/ANYmal 仿真资产完成 parkour 分支，不再用失败关节序列冒充 jump 数据。

### 并行环境容量基准

`16` 是初始保守值，不是硬限制。`16–256` 短测试均通过，长训练按 64 → 128 → 256 逐级升级。

## 维护约定

每次本项目对话结束前，更新本文件的 `Last updated`、完成事项、验证结果、待办和当轮的重要决策。该文件只记录本 VLA 仿真兴趣项目，不记录真机部署或与本项目无关的 ROS 维修。

## GitHub 学习记录同步

- 同步仓库：`git@github.com:ghw1048040694/VLA-Learning.git`
- 本地克隆路径：`/home/fabu/桌面/VLA-Learning`
- 本文件在仓库中作为独立的 `m20pro-VLA.md`，不覆盖工作区的总学习日志
  `notes/Lerobot.md`。
- 后续每次更新本文件后，同步复制到该仓库并提交推送。

### MuJoCo-first 重启与 WSL Isaac 清理（2026-07-30）

- 用户明确决定不把 Isaac Sim 作为 M20 VLA 的主训练环境。当前架构为 WSL
  MuJoCo 主训练/闭环验收，Windows `D:\isaacsim` 的 Isaac Sim
  `6.0.1-rc.7` 仅在 MuJoCo 门禁通过后执行 sim2sim 验证。
- 曾错误开始下载 WSL Linux Isaac Sim 5.1/Isaac Lab；已停止且彻底清理：卸载
  三个 `isaacsim-extscache` 包，删除 IsaacLab 源码、wheel 与 pip 中断缓存。
  根分区使用量由约 `84 GB` 降至 `67 GB`，可用空间约 `890 GB`；保留的
  `m20pro-vla` 环境约 `6.9 GB`，仅含 PyTorch/CUDA 与 MuJoCo 所需依赖。
- 固定 MuJoCo `3.3.7`、PyTorch `2.7.0+cu128`、Python `3.11.15` 已验证：
  CUDA 可用，空 MuJoCo 模型可完成一步物理计算。
- M20 ROS URDF 已经 MuJoCo 编译为 MJCF：原始 `16` 关节均被识别。随后增加
  浮动基座、地面与 `12` 个腿部位置/`4` 个轮部速度执行器；动态模型为
  `17` joints、`nq=23`、`nv=22`、`nu=16`，`480` 个 `2.5 ms` 步后状态有限，
  机身高度约 `0.588 m`。这只是资产接口验收，不是步态或导航结果。
- 多物体观测 smoke test 已以实际 RGB 检查通过：前相机可见红方块和绿圆柱，
  后相机可见黄方块；LiDAR 为 72 线、有效范围 `2.92--5.68 m`，本体输入为
  `qpos/qvel` 45 维。策略输入合同明确禁止目标世界坐标、对象 ID、语义掩码和
  仿真物体位姿。当前尚未采集新训练数据、训练 VLA/低层策略或运行闭环导航。
