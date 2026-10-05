# Generic MuJoCo Executors

These files implement the shared MuJoCo workflows used by `m20pro-vla`:

```text
collect_m20_mujoco_vla.py
audit_m20_mujoco_vla_distribution.py
train_m20_mujoco_vla.py
evaluate_m20_mujoco_hidden_search.py
play_m20_mujoco_vla.py
play_m20_smolvla.py
```

Asset preparation uses `prepare_m20_mujoco_asset.py` and
`build_m20_mujoco_model.py`. Do not copy any of these files for a new
experiment. Add a versioned JSON config and run `m20pro-vla experiment`.

`play_m20_smolvla.py` is the reusable closed-loop SmolVLA executor. It loads
an official LeRobot checkpoint, replans from both policy cameras and the 32D
state, and applies the M20 LiDAR safety, motion-smoothing, and visual-stop
contracts before the low-level wheel controller. Its success result requires
arrival, latched stop, stable posture, and zero MuJoCo obstacle contacts.
