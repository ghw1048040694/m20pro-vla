# 键盘专家示范

从现有唯一配置启动：

```sh
python -m m20pro_vla.cli experiment --config configs/experiment.json --stage teleop-record
```

打开 <http://localhost:8765>，点击“开始录制”。点击键盘区域可重新取得控制焦点。

- W/S：前进/后退。A/D：左/右转。空格和松开方向键：停车；再次按方向键可复航。
- Shift + W：最高 .50 m/s；普通 W 为 .35；S 为 -.20，Shift + S 为 -.25。
- A/D 单独使用为原地 .40 rad/s 转向；与 W/S 组合为慢速 .18 m/s、.15 rad/s 小转。
- 发现目标后松键停车至少一秒，再“完成并保存”。重录和切换任务会保留未完成记录。
- 失焦、关闭页面或 300 ms 无控制消息时停车；前后 LiDAR 基础保护仍保留。没有视觉发现停车授权门禁或自动选路。

首批仅使用既有训练布局 6012 的黄、红、绿三个任务，未暴露目标坐标或房间真值。人体操作身体命令，腿轮仍由冻结 v8 ONNX 执行。

记录位于配置 teleoperation.output_dir，每次独立目录有 metadata.json 和 chunk_*.npz。50 Hz 仿真时间轴上，双 RGB、LiDAR、去世界坐标 proprio 在动作前捕获；action 是经平滑与方向保护后送入 v8 的实际身体命令。requested_action、输入超时、控制器恢复、关节/轮目标、接触和前后位置仅作诊断。原始块可用于复核和恢复中断录制。

为交互使用 EGL llvmpipe 渲染，关闭阴影、反射和 MSAA；与历史渲染图像不保证逐位一致。元数据明确保存该渲染合同、实际 GL renderer、动作单位和范围、配置及来源 SHA。UI 显示实测处理帧率；原始仿真间隔始终 .02 秒，不将墙钟性能冒充 50 FPS。

所有记录 training_ready=false，不以 episode_*.npz 命名，不自动进入训练。需先复核实际到达、停车、姿态、接触和人类示范质量，并适配新动作归一化、存档合同及执行范围。自动化验证轨迹另有工作区 receipt，不能作人类专家数据。
