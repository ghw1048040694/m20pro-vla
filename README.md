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
历史脚本路径保留为兼容入口，但新代码必须从 `m20pro_vla` 包导入。
历史 MuJoCo checkpoint 的 `global_v1` 视觉头与当前 `spatial_v2` 均可由回放入口自动识别。

### 统一运行入口（当前已迁移的门禁）

项目不再要求为每次门禁新写一个实验脚本。先在 `m20pro-vla` 环境中安装
当前包，然后用统一 CLI：

```bash
python -m pip install -e . --no-deps
m20pro-vla doctor
m20pro-vla prepare
m20pro-vla smoke
m20pro-vla low-level-gate
m20pro-vla collect --mode randomized --episodes 1
m20pro-vla train --help
m20pro-vla eval --help
m20pro-vla play --help
m20pro-vla report
```

每次运行会自动创建 `.runtime/runs/M20-*/`，保存 `config.json`、
`command.txt`、`status.json`、`summary.json` 及预留的数据/模型/视频目录。
`--dry-run` 只检查计划，不占用磁盘。采集、审计、转换、训练、评估、播放
也可以从同一 CLI 进入；当前这些命令会记录 run context 后转发到
`scripts/mujoco/` 的兼容实现，待对应包 API 迁移并通过同一合同测试后再收口。

### 当前验收边界

当前完成的是环境与观测合同，不是导航、VLA、world model 或 sim2sim 成功。
后续训练仅使用随机多物体场景的 RGB、LiDAR、本体状态和语言；每次闭环验收
必须保存 H.264 视频、逐 episode JSON 与独立随机布局汇总。统一项目记录位于
[`../../notes/Lerobot.md`](../../notes/Lerobot.md)。

### 历史 Isaac/SmolVLA 路径（不作为当前主线）

旧的小型 CNN/字节文本行为克隆网络只保留为 M20 低层运动基线，不再作为
VLA 主线。新的高层策略使用官方 `lerobot/smolvla_base`（固定 revision
`c83c316`），LeRobot 环境位于 2 TB 盘，并复用已验证的 PyTorch
2.7/CUDA 12.8 运行库，不修改 Isaac Sim 环境。

检查官方 checkpoint 完整性、严格加载和一次三视角 action-chunk 推理：

```bash
./scripts/check_smolvla.sh
```

该命令只验证预训练 VLA 在 RTX 3060 12 GB 上可运行，不代表已经完成导航。
在转换或微调前对现有 HDF5 轨迹做结构、时序和任务覆盖审计：

```bash
./scripts/audit_m20_smolvla_data.sh
```

审计未通过时不启动 SmolVLA 微调。
可见目标 ObjectNav 的随机室内场景清单与正式采集入口为：

```bash
python3 scripts/build_m20_objectnav_manifest.py
./scripts/collect_m20_visible_objectnav.sh train_0000
./scripts/audit_m20_smolvla_data.sh
```

清单将训练/验证/测试场景和指令模板隔离，墙体、家具和 YCB
目标网格均进入 MultiMesh LiDAR。每条轨迹包含统一动作前时间戳、6 维
高层动作、目标实际可见像素验证、JSON 指标和 H.264 视频。
多房间、未见物体、隐藏目标搜索和 1 m 障碍的验收定义位于
`configs/m20pro_vla_eval_v1.yaml`，必须以闭环成功率和逐 episode H.264 视频验收。
