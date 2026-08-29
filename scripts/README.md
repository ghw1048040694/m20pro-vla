# M20Pro Sim Scripts

The user-facing MuJoCo entry point is now the installed `m20pro-vla` command.
Install the package once in the project environment:

```bash
python -m pip install -e . --no-deps
```

The first migrated lifecycle commands are:

```bash
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

Each non-dry run creates `.runtime/runs/<run-id>/` with `config.json`,
`command.txt`, `status.json`, `summary.json`, and reserved `dataset/`,
`checkpoints/`, `videos/`, and `logs/` directories. Use `--dry-run` to check
the plan without creating artifacts. The existing files below remain
compatibility entry points while collection, training, evaluation, and
playback are migrated into package APIs. The adapters stream logs into the
run directory and preserve the wrapped process exit code.

These scripts are local simulation helpers. They do not SSH to 104, install systemd units, record real bags, or control the robot.

## Script ownership

The compatibility MuJoCo-first VLA and low-level implementations are grouped
in `scripts/mujoco/`; invoke them through `m20pro-vla` for new work:

```text
scripts/mujoco/
  prepare_m20_mujoco_asset.py       # build the local MuJoCo asset
  build_m20_mujoco_model.py         # compile/validate the model
  smoke_m20_mujoco.py               # sensor and scene smoke test
  evaluate_m20_low_level_controller.py
  collect_m20_mujoco_vla.py         # dataset collection
  train_m20_mujoco_vla.py           # compact VLA training
  play_m20_mujoco_vla.py            # closed-loop replay
```

The same filenames under `scripts/` are compatibility entry points for old
commands. New code and documentation should use the grouped paths. The other
root scripts are retained legacy ROS/Isaac Lab/data commands until their
workflow is migrated and independently verified.

```bash
./scripts/start_sim.sh true
./scripts/status_sim.sh
./scripts/stop_sim.sh
```

Use `false` as the first argument to start without RViz:

```bash
./scripts/start_sim.sh false
```

## VLA / Isaac Lab environment

Training outputs are stored on the mounted 2 TB disk. Repair the workspace
links after a reboot if needed:

```bash
./scripts/prepare_output_storage.sh
```

The current disk is `/dev/sdb2` (UUID
`b9cbb43d-5119-4328-99d9-10f7c0d91e37`). Mount it with:

```bash
udisksctl mount -b /dev/disk/by-uuid/b9cbb43d-5119-4328-99d9-10f7c0d91e37
```

The embodied-learning environment is isolated in the `m20pro-vla` Conda
environment. It uses Python 3.11, Isaac Sim 5.1, Isaac Lab 2.3.2, PyTorch
2.7/CUDA 12.8, and RSL-RL.

```bash
./scripts/install_vla_env.sh
source ./scripts/activate_vla_env.sh
./scripts/check_vla_env.sh
```

Isaac Sim needs most of the RTX 3060's 12 GB VRAM. Stop other GPU training
jobs before launching it. The first simulator launch requires accepting the
NVIDIA Omniverse license in the terminal.

Convert and validate the M20 Pro asset:

```bash
./scripts/convert_m20pro_urdf.sh
./scripts/test_m20pro_asset.sh
```

The first locomotion task configuration can be checked with:

```bash
TERM=xterm python scripts/check_m20pro_task.py --headless
```

Run the first real environment reset/step smoke test:

```bash
./scripts/smoke_m20pro_locomotion.sh --num-envs 1 --steps 4
```

The old PPO runs are retained only as documented failure baselines; new VLA
training uses the public M20 expert trajectories below. All output paths are
on the 2 TB disk.

Train the compact multimodal language-conditioned action-chunk policy:

```bash
./scripts/train_m20_vla_bc.sh \
  --epochs 40 --batch-size 64 --horizon 8 --stride 2 --device cuda:0 \
  --output-dir /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/checkpoints/m20_vla_bc_v1
```

Replay it in a closed-loop Isaac Sim run. The wrapper always records a video
and the Python script writes a JSON metrics sidecar:

```bash
./scripts/play_m20_vla_bc.sh \
  --checkpoint /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/checkpoints/m20_vla_bc_v1/best.pt \
  --task-text "向前走" --steps 250 --chunk-execution 4 \
  --video-dir /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/videos/m20_vla_bc_forward_250
```

Record additional native M20 expert trajectories only when their JSON
`success` field is true:

```bash
./scripts/record_public_m20_expert.sh \
  --episodes 4 --steps 500 --task-text "向前走" --command-x 0.5 \
  --output-dir /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/datasets/public_m20_native_v1 \
  --video-dir /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/videos/public_m20_native_v1
```

Audit all M20 HDF5 episodes before any LeRobot conversion or SmolVLA
fine-tuning. The command writes a machine-readable report under `logs/` and
fails the readiness decision when scene coverage, timestamps, hidden search,
obstacle LiDAR, or jump data are missing:

```bash
./scripts/audit_m20_smolvla_data.sh
```

Build the deterministic visible ObjectNav manifest and collect one selected
episode. The manifest contains 8 training layouts, 12 YCB object categories,
24 disjoint training instruction templates, plus separate validation and test
layouts/templates. Collection is always headless and always writes H.264:

```bash
python3 scripts/build_m20_objectnav_manifest.py
./scripts/collect_m20_visible_objectnav.sh train_0000
./scripts/audit_m20_smolvla_data.sh
```

The hidden-search variant uses the same robot and camera contract but adds an
occluding screen so the target is not visible at the start:

```bash
python3 scripts/build_m20_objectnav_manifest.py --include-hidden-search
./scripts/collect_m20_hidden_objectnav.sh hidden_train_0000
./scripts/convert_m20_hidden_objectnav_to_lerobot.sh
./scripts/train_m20_smolvla_hidden_objectnav.sh
```

The hidden manifest currently spans 4 occluder families so the search data do
not collapse to a single screen shape or lane geometry.

For a reproducible coverage run, pass several manifest IDs to the batch wrapper.
It records one log per episode and runs the audit after every completed episode:

```bash
./scripts/collect_m20_visible_objectnav_batch.sh \
  train_0001 train_0002 train_0003 train_0004 \
  train_0016 train_0024 train_0032 train_0040
```

The wrapper continues through a failed episode, returns a non-zero status at the
end, and never overwrites an existing episode unless `M20PRO_OVERWRITE=1` is set.

The indoor collector uses the public M20 ONNX expert for leg stabilization and
a PhysX velocity drive for the four wheel targets. Privileged target pose is
used only to label successful demonstrations; it is explicitly prohibited and
recorded as disabled for VLA inference.

Legacy PPO checkpoint files were removed during cleanup; their failure metrics
remain in `m20pro-VLA.md`, but they are no longer a runnable storage target.

Smoke-test the leg-only jump skill:

```bash
./scripts/smoke_m20pro_jump.sh --num-envs 4 --steps 24
```

Record NVIDIA's published Isaac Lab Go2 expert with a video and a 235-dim
observation trajectory. The checkpoint cache, HDF5 data, and MP4 are placed on
the 2 TB output disk:

```bash
./scripts/record_public_go2_expert.sh \
  --steps 400 \
  --output-dir /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/datasets/public_go2_rough_v0 \
  --video-dir /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/videos/public_go2_rough_v0
```

Retarget its 12 joint targets to the M20 leg order. The output is marked
`validated=False` until a M20 video replay confirms signs and offsets:

```bash
python scripts/retarget_go2_to_m20.py INPUT.h5 OUTPUT.h5
```

Replay the retargeted actions on the M20. This is a third-person video
calibration pass and reports root-height/displacement diagnostics:

```bash
./scripts/play_m20_retargeted.sh \
  --actions-h5 OUTPUT.h5 --steps 200 \
  --video-dir /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/videos/m20_retargeted_v0
```

Validate the released Robot Parkour Learning Go1 visual checkpoint without
launching Isaac Sim:

```bash
python scripts/validate_public_parkour_checkpoint.py \
  --json-output /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/logs/public_parkour_checkpoint_validation.json
```

Replay the native Go1 public parkour policy with a mandatory third-person MP4.
The public checkpoint was trained for roughly 0.4--0.45 m parkour obstacles;
this command is an IsaacGym-to-IsaacLab protocol diagnostic, not an M20 policy:

```bash
./scripts/play_public_go1_parkour.sh \
  --policy-mode skill --steps 200 --command-x -1.0 \
  --obstacle-height 0.45 --video-dir /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/videos/public_parkour_go1
```

Run the non-visual Go1 walk checkpoint as a dynamics/action-protocol control:

```bash
./scripts/play_public_go1_parkour.sh \
  --policy-mode walk --steps 200 --command-x -1.0 --obstacle-x 5.0 \
  --video-dir /media/fabu/b9cbb43d-5119-4328-99d9-10f7c0d91e37/M20ProVLA/videos/public_parkour_go1_walk
```
