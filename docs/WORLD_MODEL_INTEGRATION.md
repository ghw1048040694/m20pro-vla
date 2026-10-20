# M20 short-horizon world-model integration

Status: design only. No world-model inference is active in the SmolVLA loop.

## Baseline first

Train and evaluate SmolVLA on the curated LeRobotDataset. Keep the current
checkpoint as a historical comparison; its training source included 18 search
episodes with less than 0.50 m net movement. Run paired closed-loop episodes
on both training layouts and layout IDs excluded from training. Do not use the
old 1/13 result as the expected performance of the cleaned policy.

## Prediction contract

At each SmolVLA replan, score a small set of ten-step body-command chunks
(`[forward, lateral, yaw, stop]`; 25 Hz policy actions, 0.4 s horizon).
Inputs are the two RGB cameras, the 32D state, task text, recent actions, and
each candidate chunk. Outputs are predicted target progress, robot motion,
obstacle contact probability, tilt/low-height probability, and valid-stop
probability. Target XY, simulator object poses, semantic truth masks, and the
DDDMR map are training labels only, never policy or scorer inputs.

The existing `M20TrajectoryScorer` is a useful candidate-generation and
multi-head scoring reference, but it expects the older policy's 192D latent.
Its weights and feature contract cannot be plugged into SmolVLA.

## Data and training

Fork MuJoCo at safe states where the cleaned SmolVLA stalls, approaches a
wall, nears a target, or starts a risky turn. Execute multiple bounded
candidate chunks from each identical state through the canonical M20
low-level controller. Record observed progress, actual wheel response,
contact, height/tilt, and target visibility. Include both expert
continuations and unsuccessful candidates; positive expert demonstrations
alone do not train a reliable counterfactual ranker. Split by layout before
training to prevent near-duplicate forks from leaking into validation.

Start with a compact action-conditioned outcome model. A video-prediction
model adds substantial data and compute requirements without first proving
that short-horizon outcome ranking helps this task.

## Runtime and acceptance

SmolVLA proposes; the outcome model ranks the candidates; the existing LiDAR
shield and M20 controller retain authority over execution. On M20Pro-3D-Nav,
first run the ranker in shadow mode, then consider short DDDMR subgoals. It
does not publish unsupervised `cmd_vel`.

Compare the cleaned SmolVLA and SmolVLA-plus-ranker from identical starts and
seeds. Require higher success and stop-at-target rates, no worse collision or
tilt results, and measured inference latency within the 0.4 s replan budget.
Report these separately for seen and unseen layouts. Reject a ranker that
only improves the training layouts or selects actions outside the validated
body-command envelope.
