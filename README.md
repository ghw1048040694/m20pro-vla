# M20 Pro ROS 2 Sim

这是 M20 Pro 仿真项目，只服务上位机本地仿真、RViz 调参、地图/楼层/任务逻辑验证。真机 104 启动、自检、开机自启、原始雷达 relay 和现场脚本已经拆到独立 real 仓库。

## 编译

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## 启动

带 RViz：

```bash
./scripts/start_sim.sh true
```

不带 RViz：

```bash
./scripts/start_sim.sh false
```

直接 launch：

```bash
ros2 launch m20pro_bringup m20pro_sim.launch.py
```

网页默认地址：

```text
http://127.0.0.1:8080
```

## 停止和状态

```bash
./scripts/status_sim.sh
./scripts/stop_sim.sh
```

## 仿真链路

仿真启动会拉起：

- `sim_bridge`，以 `m20pro_tcp_bridge` 节点名发布仿真位姿/odom/TF；
- `dual_lidar_simulator`，从 PCD 地图裁剪局部点云；
- `pointcloud_fusion`，生成 `/scan`；
- Nav2 map server、planner、controller、BT navigator；
- `floor_manager` 和 `floor_goal_bridge`；
- 可选 `dynamic_obstacle_simulator`；
- 可选网页前端和 RViz。

## 关键文件

```text
src/m20pro_bringup/launch/m20pro_sim.launch.py
src/m20pro_bringup/config/m20pro.yaml
src/m20pro_bringup/config/nav2_params_sim.yaml
src/m20pro_bringup/config/map_manifest.yaml
src/m20pro_bringup/config/inspection_waypoints.yaml
src/m20pro_bringup/rviz/m20pro_sim.rviz
src/m20pro_navigation/m20pro_navigation/sim_bridge_node.py
src/m20pro_navigation/m20pro_navigation/dual_lidar_simulator.py
src/m20pro_navigation/m20pro_navigation/dynamic_obstacle_simulator.py
```

## 注意

- 本仓库不控制真实机器狗。
- 本仓库不安装 systemd 自启动。
- 本仓库不访问 103/104/106 的真机服务。
- 仿真沿用 `/m20pro_tcp_bridge/...` 话题命名，是为了让网页和任务逻辑与真机接口形状一致。
- 历史开发记录与思路已经统一并入 `../../notes/Lerobot.md`；新的真机开发请到 real 仓库继续。

## VLA 具身仿真环境（当前主线）

M20 VLA 采用 MuJoCo-first：WSL 内的 MuJoCo 负责快速训练、数据采集和
闭环验收；Windows 的 `D:\isaacsim`（Isaac Sim 6.0.1-rc.7）只在 MuJoCo
门禁通过后用于 sim2sim 验证。Windows 运行时不会被 WSL Python 导入，也不在
WSL 内重复安装 Isaac Sim/Isaac Lab。

```bash
./scripts/install_vla_env.sh
source ./scripts/activate_vla_env.sh
./scripts/check_vla_env.sh
python scripts/prepare_m20_mujoco_asset.py
python scripts/build_m20_mujoco_model.py
python scripts/m20_mujoco_observation_smoke.py
```

MuJoCo 主线脚本的实际归档目录是 `scripts/mujoco/`；上面的根目录路径是
兼容入口，新实验请直接使用：

```bash
python scripts/mujoco/prepare_m20_mujoco_asset.py
python scripts/mujoco/build_m20_mujoco_model.py
python scripts/mujoco/m20_mujoco_observation_smoke.py
```

固定运行时为 Python 3.11、MuJoCo 3.3.7 和 PyTorch 2.7.0/CUDA 12.8。
`scripts/mujoco/m20_mujoco_observation_smoke.py` 生成前后 RGB、72 线 LiDAR、
`qpos/qvel` 和语言指令；策略输入严禁包含目标坐标、语义掩码或仿真物体位姿。
运行产物均在忽略的 `.runtime/` 下。

### VLA 软件结构

MuJoCo VLA 代码已经工程化为可安装包 `src/m20pro_vla/`。底层执行器的唯一实现是
`m20pro_vla.low_level.M20LowLevelController`，所有 MuJoCo VLA、未来 SmolVLA/Pi0.5
适配器、world-model MPC 和评估器都只能向它提交统一的
`[forward, lateral, yaw, stop]` 机身命令；它负责 12 个腿关节、4 个轮关节、姿态反馈、
速度限制和制动。不得在策略或实验脚本中复制另一套关节控制。

工程目录、长期接口和产物规则见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)，
完整目录职责见 [`docs/PROJECT_LAYOUT.md`](docs/PROJECT_LAYOUT.md)，
低层控制合同见 [`configs/m20pro_low_level_v1.yaml`](configs/m20pro_low_level_v1.yaml)。
新代码必须从 `m20pro_vla` 包导入；不得为单次实验复制训练或评估脚本。
历史 MuJoCo checkpoint 的 `global_v1` 视觉头与当前 `spatial_v2` 均可由回放入口自动识别。

### 单配置实验入口

每个实验只新增或复制一个 JSON 配置，不新增 Python/Shell 脚本。当前完整配置是
`configs/experiment.json`，它同时包含路径、采集、数据门槛、
训练超参、评估参数和晋级标准。

```bash
python -m pip install -e . --no-deps
m20pro-vla doctor
m20pro-vla prepare
m20pro-vla smoke
m20pro-vla low-level-gate
m20pro-vla experiment --config configs/experiment.json --stage plan
m20pro-vla experiment --config configs/experiment.json --stage collect
m20pro-vla experiment --config configs/experiment.json --stage collect-recovery
m20pro-vla experiment --config configs/experiment.json --stage curate
m20pro-vla experiment --config configs/experiment.json --stage audit
m20pro-vla experiment --config configs/experiment.json --stage convert
m20pro-vla experiment --config configs/experiment.json --stage train
m20pro-vla experiment --config configs/experiment.json --stage train-smolvla
m20pro-vla experiment --config configs/experiment.json --stage smolvla-eval
m20pro-vla experiment --config configs/experiment.json --stage eval
m20pro-vla experiment --config configs/experiment.json --stage gate
m20pro-vla report
```

每次运行会自动创建 `.runtime/runs/M20-*/`，保存 `config.json`、
`command.txt`、`status.json`、`summary.json` 及预留的数据/模型/视频目录。
`plan` 和 `--dry-run` 只检查计划，不占用训练资源。每个执行阶段自动写入 run context；
训练阶段拒绝覆盖已有 checkpoint，必须为新实验设置新的 `experiment_id` 和输出路径。
`convert` 将M20 NPZ轨迹转换为标准LeRobotDataset；`train-smolvla` 在配置指定的
`lerobot` conda环境中调用官方训练器。二者仍由同一JSON和统一CLI调度。
`collect-recovery` 从失败闭环中的安全状态采集专家续跑，只保存通过到达、姿态与零碰撞
门槛的轨迹；`curate` 过滤未移动的旧采集轨迹，并用符号链接组合有效数据，不复制视频。

### 标准训练与晋级门禁

实验配置只允许在 WSL2 + CUDA 环境中启动训练，并在训练前强制检查 episode/NPZ 配对、
布局、地形、语言、anchor/varied 起点、成功轨迹、目标发现和障碍净距。训练完成后不能凭
loss 导出，必须继续运行同一配置的 `eval` 和 `gate` 阶段。

```bash
m20pro-vla experiment --config configs/experiment.json --stage gate
```

该门禁同时检查 layout-disjoint 离线指标和 learner-only 隐藏目标闭环指标。只有
`eligible_for_shadow=true` 才允许执行 `export-policy --target-runtime shadow`；shadow
仍然只记录建议，不控制机器人。每张 DDDMR 地图不单独训练 LoRA。


### 面向 M20Pro-3D-Nav 的 VLA 导出合同

VLA 不按每张 DDDMR 地图做 LoRA。DDDMR 地图负责几何定位、三维地图和执行安全；
VLA 负责语言、视觉目标和局部短目标/机身命令。只有相机安装、场景外观、目标类别或
任务分布发生明显变化时，才考虑 domain/site 级 adapter。

训练得到的 checkpoint 先导出为实机 bridge 可读的 manifest，而不是直接拷到真机控制链路：

```bash
m20pro-vla export-policy \
  --checkpoint .runtime/checkpoints/<run>/best.pt \
  --target-runtime shadow \
  --output-dir .runtime/exports/<policy-id>
```

默认只记录 checkpoint 路径和 SHA-256，不复制大权重；需要形成自包含交付包时再显式加
`--copy-checkpoint`。导出目录包含：

- `manifest.json`：模型、观测、动作、安全和 M20Pro-3D-Nav 消费方式；
- `m20pro_real_vla_deploy_contract_v1.yaml`：两仓库握手合同；
- 可选 checkpoint copy。

实机第一消费者应是 `M20Pro-3D-Nav` 的 `m20pro_vla_bridge` shadow mode。通过 shadow
录包/日志验证后，优先让 VLA 输出短距离 `/navigate_to_pose_3d` 子目标给 DDDMR 执行；
直接速度模式必须经过 command supervisor 和 command mux。

### 当前验收边界

当前完成的是环境与观测合同，不是导航、VLA、world model 或 sim2sim 成功。
后续训练仅使用随机多物体场景的 RGB、LiDAR、本体状态和语言；每次闭环验收
必须保存 H.264 视频、逐 episode JSON 与独立随机布局汇总。统一项目记录位于
[`../../notes/Lerobot.md`](../../notes/Lerobot.md)。
