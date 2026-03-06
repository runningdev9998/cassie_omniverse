# cassie_omniverse

Training a Cassie robot to walk using **Omniverse** and **Isaac Lab**.


<video controls width="720">
  <source src="videos/cassie_walk_v0.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>


## Policy
- PPO

## RL Library
- rl_games

## Observations
- base_height
- base_lin_vel
- base_ang_vel
- base_yaw_roll
- base_angle_to_target
- base_up_proj
- base_heading_proj
- joint_pos_norm
- joint_vel_rel
- feet_body_forces
- actions

## Actions
Action values applied to **all 12 joints** of the Cassie robot.

## Controller
Joint effort controller.

---

# Training the Model

1. Place this repository in:

```
source/isaaclab_tasks/isaaclab_tasks/manager_based/classical
```

(This helps register the environment with Isaac Lab.)

2. Run:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py --task Isaac-Custom-Cassie-v0
```

---

# Playing the Trained Model

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py --task Isaac-Custom-Cassie-v0 --num_envs 1
```

---

# Environment Managers

## Action Manager

| Index | Name         | Dimension |
|------|--------------|----------|
| 0 | joint_effort | 12 |

---

## Observation Manager

**Active Observation Terms in Group: `policy` (shape: 60)**

| Index | Name | Shape |
|------|------|------|
| 0 | base_height | (1,) |
| 1 | base_lin_vel | (3,) |
| 2 | base_ang_vel | (3,) |
| 3 | base_yaw_roll | (2,) |
| 4 | base_angle_to_target | (1,) |
| 5 | base_up_proj | (1,) |
| 6 | base_heading_proj | (1,) |
| 7 | joint_pos_norm | (12,) |
| 8 | joint_vel_rel | (12,) |
| 9 | feet_body_forces | (12,) |
| 10 | actions | (12,) |

---

## Event Manager

**Mode: `reset`**

| Index | Name |
|------|------|
| 0 | reset_base |
| 1 | reset_robot_joints |

---

## Termination Manager

| Index | Name | Time Out |
|------|------|------|
| 0 | time_out | True |
| 1 | torso_height | False |

---

## Reward Manager

| Index | Name | Weight |
|------|------|------|
| 0 | progress | 1.0 |
| 1 | alive | 1.0 |
| 2 | upright | 0.1 |
| 3 | move_to_target | 0.5 |
| 4 | velocity_penalty | -1.0 |
