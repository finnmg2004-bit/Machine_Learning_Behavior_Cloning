# Behaviour Cloning via Supervised Learning

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![PyBullet](https://img.shields.io/badge/Simulation-PyBullet-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![pixi](https://img.shields.io/badge/env-pixi-yellow)

Teaching a mobile manipulator to imitate an expert controller. A **Robotnik mobile base** with a **7‑joint Kinova arm** drives to a target (the red dot) and reaches it with its hand. We collect demonstrations from a hand‑designed expert, then train models of increasing complexity (Ridge regression, an MLP and a GRU) to reproduce its behaviour from state alone, and test them by letting them drive the robot in closed loop.

Built as a group project by a team of three. The full write‑up, including MLP results, is in the PDF report in this repo.

---

## Results at a glance

The final model, a two‑stage **GRU policy** (`gru_e80`), was evaluated once on the held‑out test split:

| | GRU (`gru_e80`) | Expert |
|---|---|---|
| Test success (hand within 1 cm) | **60 / 63 (95.2%)** | 63 / 63 |
| Final error | 0.92 ± 0.41 cm | 0.86 ± 0.29 cm |
| Inference time | ~100 µs per step (1 ms control period) | – |
| Model size | 305k parameters, 2.4 MB | – |

It also generalises to targets further away than anything in training (30/30) and holds up better than the expert itself under heavy actuator noise. Full results are [further down](#detailed-results).

---

## Contents

- [How it works](#how-it-works)
- [Repository layout](#repository-layout)
- [Getting started](#getting-started)
- [Running the pipeline](#running-the-pipeline)
- [Models](#models)
- [Detailed results](#detailed-results)
- [Output files](#output-files)
- [Authors and contributions](#authors-and-contributions)

---

## How it works

```mermaid
flowchart LR
    A[Plan episodes] --> B[Collect expert demos]
    B --> C[Quality checks]
    C --> D[Build datasets]
    D --> E[Train / val / test split]
    E --> F[Train models<br/>Ridge · MLP · GRU]
    F --> G[Closed-loop evaluation]
    G --> H[Diagnose runs]
```

Every step reads the previous step's files, so any step can be re‑run on its own.

The task splits naturally into two phases, and each gets its own model:

- **Base:** drives towards the target and parks at a standoff distance. Outputs right and left wheel commands.
- **Arm:** once parked, reaches the target. Outputs 7 joint torques (optionally as a residual on top of gravity compensation).

Episodes are split into train / val / test **by episode, never by row**, so near‑identical 1 ms samples can't leak between splits. Extra pools of out‑of‑range episodes test generalisation, and DART noise injection gives the models examples of recovering from mistakes.

---

## Repository layout

| File | Purpose |
|---|---|
| `robotnik_arm_task.py` | Simulation, expert controller and `run_episode()`, used by every other script |
| `pipeline.py` | Plan, collect, quality‑check and build datasets (steps 1–4) |
| `check_datasets.py` | Sanity checks on input encodings |
| `splits.py` | Fixed train / val / test split by episode |
| `training.py` | Dummy and Ridge baselines |
| `mlp.py` | MLP policies for base and arm |
| `gru.py` | Recurrent (GRU) policy: training, evaluation, ablations, reports |
| `closed_loop.py` | Closed‑loop evaluation for Ridge models |
| `mlp_eval.py` | Closed‑loop evaluation for MLP models |
| `test.py` | Plots a single closed‑loop run against the expert |
| `run_random.py` | Watch the final model (or the expert) on a random target |

---

## Getting started

### Prerequisites

- [pixi](https://pixi.sh) for the environment
- The `models/` and `configs/` folders (robot description files) from **RoboEnv**, copied into the repo root

### Install

```bash
git clone https://github.com/finnmg2004-bit/Machine_Learning_Behavior_Cloning.git
cd Machine_Learning_Behavior_Cloning
pixi install
```

Run **every** script from the repo root through pixi:

```bash
pixi run python <file>.py
```

### Quick demo

To watch the final GRU policy reach a random target in the PyBullet window:

```bash
pixi run python run_random.py
```

Set `POLICY = "expert"` at the top of the file to watch the expert instead.

<details>
<summary><b>Troubleshooting: wrong environment</b></summary>

Check which Python is being used:

```bash
pixi run python -c "import sys; print(sys.executable)"
```

The path must be inside this repo's `.pixi` folder. If it shows `RoboEnv`, run `Remove-Item Env:PIXI_PROJECT_MANIFEST` in PowerShell, or open a new terminal.

</details>

---

## Running the pipeline

### Step 0: watch the expert (optional)

```bash
pixi run python robotnik_arm_task.py
```

One episode runs twice in the PyBullet window: first the expert does everything, then a trained Ridge model controls the arm. Each run prints its outcome, final distance to the target and duration.

<details>
<summary><b><code>run_episode()</code> arguments</b></summary>

`run_episode()` runs one episode and returns `df` (one row per 1 ms step) and `summary` (one row describing the episode).

| Argument | Meaning |
|---|---|
| `target` | Target position `[x, y, z]` in metres |
| `start_xy`, `start_yaw` | Where the base starts, and which way it faces (rad) |
| `start_arm_offset` | 7 angle offsets (rad) added to the default arm pose, clipped to stay `joint_margin` inside the joint limits |
| `seed`, `noise_seed` | Random seeds; the noise has its own seed so it never changes the episode's geometry |
| `noise_std` | DART noise level: 0 = none, 0.5 ≈ half the normal steering effort |
| `headless` | `True` = no window (fast), `False` = show the 3D window |
| `floor_friction` | Floor friction (1.0 normal; used for robustness tests) |
| `max_steps` | Safety cap on steps (1 step = 1 ms) |
| `joint_margin` | How far inside the joint limits the start pose must stay (rad) |
| `verbose` | Print extra information |
| `policy` | `None` = the expert controls everything; a `Policy` object = a trained model controls the arm and/or base |

</details>

<details>
<summary><b>Constants in <code>robotnik_arm_task.py</code></b></summary>

| Constant | Meaning |
|---|---|
| `NOISE_HOLD_STEPS` | DART noise is redrawn every 150 steps and held in between |
| `WHEEL_NOISE_SCALE`, `TORQUE_NOISE_SCALE` | Size of the noise per wheel / per joint at `noise_std = 1` |
| `TORQUE_LIMIT` | Arm torques are clipped to ±30 N·m |
| `BASE_NAV_TIMEOUT` | After 25 s the base counts as parked, wherever it is |
| `STANDOFF_*` | How far before the target the base parks, depending on target height |
| `REACH_TOLERANCE` | Success = hand within 1 cm of the target |

</details>

### Steps 1–4: data collection (`pipeline.py`)

All four steps use the same file. Edit the **settings block at the bottom**, set `STEP`, then run:

```bash
pixi run python pipeline.py
```

| Step | `STEP` | What it does | Output |
|---|---|---|---|
| 1 | `"plan"` | Chooses the target, start pose, arm offsets and noise for every episode | `data/<pool>/plan.csv`, `outputs/plan_<pool>.png` |
| 2 | `"collect"` | Runs the expert on every planned episode not yet collected | `data/<pool>/ep#####.parquet`, `summary.csv` |
| 3 | `"qa"` | Checks the collected data and prints a report | Report + `outputs/outcomes_<pool>.png` |
| 4 | `"dataset"` | Turns raw logs into model inputs `X` and labels `y` | `data/<pool>/dataset_*.npz` |

> **Notes**
> - `plan` refuses to overwrite an existing `plan.csv`. Delete it only if nothing has been collected yet.
> - `collect` is **resumable**: after a crash or Ctrl+C, run it again and finished episodes are skipped.
> - Run `collect` outside VS Code with the laptop plugged in, and don't open the live `summary.csv` in Excel (it locks the file).

<details>
<summary><b>Settings</b></summary>

| Setting | Meaning | Typical value |
|---|---|---|
| `STEP` | `"plan"`, `"collect"`, `"qa"` or `"dataset"` | Change each time |
| `POOLS` | Groups of episodes to work on; comment out a line to skip a pool | All pools |
| `METHOD` | Target spread: `"lhs"` (Latin hypercube, even coverage) or `"random"` | `"lhs"` |
| `WORKERS` | Parallel simulators during `collect` | 3–8 (fewer if RAM runs out) |
| `LIMIT` | Only collect this many episodes (quick test); `None` = all | `None` |
| `MODEL` | Dataset to build: `"arm"` or `"base"` | Change each run |
| `STRIDE` | Keep every N‑th 1 ms row (adjacent rows are near‑copies) | `10` |
| `RESIDUAL` | Arm only: label = torque minus gravity instead of full torque | `False` / `True` |

</details>

<details>
<summary><b>Episode pools</b></summary>

Each line of `POOLS` reads `"name": (episodes, ranges, noise_std)`. The ranges are in the `TRAIN` and `OOR` dictionaries at the top of `pipeline.py`.

| Pool | Episodes | Ranges | Noise | Used for |
|---|---|---|---|---|
| `main` | 420 | Normal training ranges | 0 | Train / val / test |
| `dart` | 150 | Normal | 0.5 | Extra training data with recovery examples |
| `oor_far` | 30 | Targets 2.5–3.5 m away | 0 | Test only: further than training |
| `oor_high` | 30 | Targets 1.10–1.30 m high | 0 | Test only: higher than training |
| `oor_start` | 30 | Start positions 1.5–3.0 m out | 0 | Test only: unfamiliar start |
| `oor_arm` | 30 | Arm start offsets up to ±0.5 rad | 0 | Test only: unusual arm pose |

</details>

<details>
<summary><b>Key <code>summary.csv</code> columns</b></summary>

| Column | Meaning |
|---|---|
| `stop_reason` | `reached`, `unreachable` (no 2 mm progress for 3 s), `step_cap` or `crash` |
| `final_ee_error`, `best_ee_error` | Final and best hand‑to‑target distance (m) |
| `nav_done_time` | When the base finished parking (s); 25.0 = timed out |
| `sim_duration` | Episode length (s); always at least 8 s by design |
| `applied_offset0–6` | Arm start offsets actually applied, after clipping |
| `frac_steps_pinned` | Share of steps with a joint near its limit |

</details>

<details>
<summary><b>Reading the QA report</b></summary>

| Line | Meaning | Good looks like |
|---|---|---|
| `planned X, finished Y, files Z` | Episodes planned / with a summary row / with a data file | All three equal |
| Stop‑reason shares | How each episode ended | Mostly `reached` |
| `by height` / `by distance` | Stop reasons per height or distance band | No band much worse than others |
| `MB/episode`, `s/episode` | File size and run time | For planning disk space and time |
| `RED FLAG: ...` | Crashes, NaNs, clock errors, finger drift, success under 80% | None (except `dart`, where low success is expected) |
| `QA PASSED` | No red flags | |

</details>

<details>
<summary><b>Building the datasets</b></summary>

The base dataset uses rows while the base is driving; the arm dataset uses rows after it has parked. Failed episodes are cut at their closest approach. Run the `dataset` step three times:

| Run | `MODEL` | `RESIDUAL` | Creates |
|---|---|---|---|
| 1 | `"arm"` | `False` | `dataset_arm.npz`: labels = full arm torques |
| 2 | `"arm"` | `True` | `dataset_arm_residual.npz`: labels = torques minus gravity |
| 3 | `"base"` | `False` | `dataset_base.npz`: labels = right and left wheel commands |

Model inputs (built by `make_inputs`, in the robot's own frame):

| Model | Inputs |
|---|---|
| Base | Target position relative to the robot, target height, distance and bearing to target, forward / sideways speed, turning rate |
| Arm | Target relative to the robot, 7 joint angles, 7 joint speeds, distance to limit for joints 1, 3, 5, target minus shoulder (x, y, z), target minus hand (x, y, z) |

Each `.npz` holds `X`, `y`, `episode_id`, `phase`, `feature_names` and `label_names`.

</details>

### Step 5: check the datasets (optional)

```bash
pixi run python check_datasets.py
```

Prints a one‑line verdict for each of two checks:

- **Joint wrap:** the unlimited arm joints never turn a full circle in the data, so raw angles are safe inputs.
- **Bearing:** when the target is behind the robot, the expert turns towards the target's side, so the raw bearing angle is the right input.

### Step 6: split the episodes

```bash
pixi run python splits.py
```

Splits the 420 `main` episodes into **train (294) / val (63) / test (63)** with a fixed seed, and prints a table showing the splits have similar distances and heights. It refuses to overwrite an existing split, so the test set can't change by accident. **Commit the resulting `splits/*_ids.txt` files.**

### Step 7: train

Pick the trainer for the model you want. All of them score on the **val** split only; test is kept for the end.

```bash
pixi run python training.py   # Dummy + Ridge baselines
pixi run python mlp.py        # MLP policies
pixi run python gru.py        # GRU policy (see below)
```

<details>
<summary><b>Reading <code>outputs/results_offline.csv</code></b></summary>

| Column | Meaning |
|---|---|
| `val_nmse` | Error ÷ error of always guessing the mean: 1.0 = Dummy level, 0 = perfect |
| `val_r2` | Share of variation explained: 1 = perfect, 0 = Dummy level |
| `val_rmse_<output>` | Typical error per output in real units (N·m or rad/s) |
| `val_full_*` | Arm only: scored on the full torque sent (gravity added back for the residual model) |
| `val_nmse_align/drive/orient` | Base only: error in each phase of the expert's driving |
| `alpha` | Chosen Ridge penalty |
| `n_params`, `fit_s`, `infer_us` | Model size, training time (s), time per prediction (µs; the robot needs one every 1,000 µs) |

</details>

### Step 8: closed‑loop evaluation

The trained model drives the robot on the same episodes the expert ran. The expert still runs in the background, so its commands are logged next to the model's.

```bash
pixi run python closed_loop.py   # Ridge models
pixi run python mlp_eval.py      # MLP models
```

Suggested workflow: set `GUI = True` to watch one episode, then `N_EPISODES = 3` to check the table prints without crashes, then `N_EPISODES = None` for all 63 val episodes. Use `SPLIT = "test"` only for final results.

<details>
<summary><b>Settings</b></summary>

`closed_loop.py`:

| Setting | Meaning |
|---|---|
| `NAME` | Label for this run's result files (use a new one per experiment) |
| `ARM_MODEL`, `BASE_MODEL` | Path to a saved model, or `None` to let the expert control that part |
| `SPLIT` | `"val"` while developing, `"test"` only for final results |
| `N_EPISODES` | `None` = every episode in the split, or a number for a quick try |
| `WORKERS` | Parallel simulators |
| `GUI` | `True` = watch one episode in the 3D window |
| `SAVE_STEPS` | Save every step of every episode (needed for step 9) |
| `POLICY_EVERY` | Query the model every N steps. **Keep at 1**: at 10 the arm became unstable |

`mlp_eval.py`:

| Setting | Meaning |
|---|---|
| `MODEL_PATH` | Path to the saved MLP weights (`.joblib` or `.pt`) |
| `SPLIT` | `"val"` or `"test"` |
| `GUI` | `True` = watch one episode, `False` = headless |
| `N_EPISODES` | `None` = whole split, or a number for a quick test |
| `WORKERS` | Parallel simulators |

</details>

<details>
<summary><b>Reading the printed table</b></summary>

| Column | Meaning |
|---|---|
| `success` | Share of episodes ending with the hand within 1 cm of the target |
| `median_final_error_m` | Typical final hand‑to‑target distance (m) |
| `mean_time_s_when_reached` | Average episode length when successful |
| `crashes` | Episodes that raised an error (messages printed below the table) |

Rows are the `expert` (from the collected data) and your model, on exactly the same episodes.

</details>

### Step 9: diagnose a run (optional)

```bash
pixi run python test.py
```

Set the model path and the `outputs/closed_loop/<NAME>/` folder inside the file. For one episode it plots the hand‑to‑target distance and, per joint, the model's torque against the expert's. Output goes to `outputs/diagnose_*.png`; use a new file name per run or the plots overwrite each other.

---

## Models

| Model | File | Description |
|---|---|---|
| Dummy | `training.py` | Always predicts the training mean. The floor every real model must beat |
| Ridge | `training.py` | Linear regression with an L2 penalty, strength chosen on val |
| MLP | `mlp.py` | Feed‑forward network wrapped as a `Policy`, so it plugs straight into the existing pipeline |
| **GRU** | `gru.py` | **Final model.** Recurrent policy with memory of the expert's internal state |

### MLP

Trains separate base and arm networks with mini‑batch gradient descent, tracks validation loss and keeps the best weights. `mlp.py` also contains the expert policy used for the hidden reference calculation. MLP results are in the PDF report.

<details>
<summary><b>Hyperparameters (top of <code>mlp.py</code>)</b></summary>

| Setting | Meaning |
|---|---|
| `HIDDEN_SIZES` | Hidden layer sizes, e.g. `[128, 128]` |
| `LEARNING_RATE` | Optimiser step size, e.g. `1e-3` |
| `BATCH_SIZE` | Samples per batch |
| `EPOCHS` | Passes over the training set |
| `DROPOUT` | Dropout rate (if used) |

</details>

### GRU

**Why memory?** The expert isn't a pure function of the current state. While driving it switches between *align / drive / orient* phases, and the arm controller builds up its own joint reference `q_des` over time. A single‑step model can't see either of those. The GRU is built on top of the pipeline in the same way as the MLP (no pipeline files are edited): the base GRU drives until parked, then the arm GRU takes over.

| | Base | Arm |
|---|---|---|
| Inputs | 10 features: target and parking pose relative to the robot, heading errors | 39 features: target, joint angles (sin/cos and raw), joint speeds, hand‑to‑target error |
| Outputs | 2 wheel speeds, clipped to ±3.5 rad/s | 7 torques minus gravity (gravity added back at run time, clipped to ±30 N·m) |
| Parameters | 150k | 155k |

- **Network (each):** 128 embedding → 1 GRU layer of 128 → 2×128 head that sees both the memory and the current input. Memory updates every 10 ms; the head runs every 1 ms.
- **Training:** whole episodes as sequences (every 10th step), batches of 8 episodes, MSE on standardised labels with arm steps within 5 cm of the target weighted 3×, AdamW (lr 1e‑3, cosine decay, weight decay 1e‑5), gradient clipping at 1.0. Keeps the epoch with the lowest val loss.
- **Inference:** plain NumPy, about 100 µs per step against the 1 ms control period. PyTorch is only needed for training.

#### Running `gru.py`

Set the settings block at the **bottom of `gru.py`**, then `pixi run python gru.py`.

> **Before you start:** `pipeline.py` must have collected `main`, `dart` and the four `oor_*` pools, and `splits.py` must have made the splits (steps 1–6). Then run `STEP = "pools"` once.

| `STEP` | What it does | Output |
|---|---|---|
| `"pools"` | Plans and collects two extra pools: `dart2` (500 episodes, DART 0.5) and `sel` (137 clean episodes used only for model selection) | `data/dart2/`, `data/sel/` |
| `"train"` | Trains base and arm GRUs on `main` train + the `EXTRA` pools | `models/<NAME>.joblib` |
| `"closed_loop"` | The model (or the expert, with `NAME = "expert"`) drives on `SPLIT` of `main` or another `POOL`, optionally with noise / low friction | `outputs/closed_loop_<NAME>_<split or pool><TAG>.csv` |
| `"report"` | Size, inference time, offline test NMSE, closed‑loop numbers and loss curves | `outputs/gru_report.csv`, `outputs/gru_loss_curves_<NAME>.png` |
| `"ablation"` | Development table (offline NMSE + val / sel successes per version) | `outputs/gru_ablation.csv` |
| `"dagger"`, `"ensemble"` | Two experiments that didn't make the final model | `data/gru_dagger1/`, `models/<NAME>.joblib` |

<details>
<summary><b>Settings</b></summary>

| Setting | Meaning |
|---|---|
| `NAME` | Model name in `models/`; for `closed_loop` it can also be `"expert"` |
| `EXTRA` | Extra training pools, e.g. `["dart", "dart2"]` |
| `EPOCHS` | Epochs (best val epoch is kept; no early stopping) |
| `BASE_FROM` | Copy the base GRU from this model and only train the arm (`None` = train both) |
| `NEAR_WEIGHT`, `NEAR_DIST` | Arm steps closer than `NEAR_DIST` m to the target count `NEAR_WEIGHT` times |
| `SEED` | Training seed (initial weights and batch order only) |
| `SPLIT`, `POOL` | Episodes for `closed_loop`: `SPLIT` of `main`, or `POOL = "sel"` / `"oor_far"` / `"oor_high"` / `"oor_start"` / `"oor_arm"` |
| `SENSOR_NOISE`, `ACT_NOISE`, `FRICTION` | Robustness conditions, same definitions as `mlp_eval.py` |
| `TAG` | Suffix for robustness runs so the clean result isn't overwritten |
| `WORKERS`, `GUI` | Parallel simulators; `GUI = True` shows one episode |

</details>

<details>
<summary><b>Reproducing every model in the report</b></summary>

Run `STEP = "train"` with these settings, in order. Anything not listed uses the defaults (`NEAR_WEIGHT, NEAR_DIST = 3.0, 0.05`, `SEED = 0`). All of these except the seed models are in `models/`, so you can skip straight to `"closed_loop"`.

| Model | `EXTRA` | `EPOCHS` | `BASE_FROM` | Other | Notes |
|---|---|---|---|---|---|
| `gru` | `["dart"]` | 40 | `None` | | First version, main + DART 150 |
| `gru_dart2` | `["dart", "dart2"]` | 40 | `None` | | + DART 500 (650 DART episodes total) |
| `gru_nw10` | `["dart", "dart2"]` | 40 | `"gru_dart2"` | `NEAR_WEIGHT, NEAR_DIST = 10.0, 0.03` | 10× weight on last 3 cm (discarded) |
| **`gru_e80`** | `["dart", "dart2"]` | **80** | `"gru_dart2"` | | **Final model** |
| `gru_e80_s1`, `gru_e80_s2` | `["dart", "dart2"]` | 80 | `"gru_dart2"` | `SEED = 1` / `2` | Only needed for the ensemble |
| `gru_ens` | | | | `STEP = "ensemble"`, `MEMBERS` as in file | Mean of the three 80‑epoch arms (discarded) |
| `gru_dagger1` | `["dart", "dart2", "gru_dagger1"]` | 40 | `"gru_dart2"` | First run `STEP = "dagger"` with `NAME = "gru_dart2"` | DAgger round 1 (discarded) |

</details>

<details>
<summary><b>Reproducing the results</b></summary>

Everything is deterministic: the same model on the same episode gives the same result every time.

| Result | Settings for `STEP = "closed_loop"` | File |
|---|---|---|
| Development table, val / sel | `NAME` = each model, `SPLIT = "val"`, then `POOL = "sel"` | `closed_loop_<model>_val.csv`, `_sel.csv` |
| Development table, NMSE column | `STEP = "ablation"` | `gru_ablation.csv` |
| Final test result | `NAME = "gru_e80"`, `SPLIT = "test"` | `closed_loop_gru_e80_test.csv` |
| Out‑of‑range | `POOL = "oor_far"` / `"oor_high"` / `"oor_start"` / `"oor_arm"` | `closed_loop_gru_e80_oor_*.csv` |
| Robustness, GRU | `SPLIT = "val"`, then one at a time: `ACT_NOISE = 0.25, TAG = "_act025"`; `ACT_NOISE = 0.5, TAG = "_act05"`; `FRICTION = 0.4, TAG = "_fr04"`; `SENSOR_NOISE = 0.005 / 0.01 / 0.02, TAG = "_sn005" / "_sn01" / "_sn02"` | `closed_loop_gru_e80_val_<tag>.csv` |
| Robustness, expert | Same, with `NAME = "expert"` (the expert reads true state, so only actuator noise and friction apply) | `closed_loop_expert_val_<tag>.csv` |
| Size / speed / loss curves | `STEP = "report"` | `gru_report.csv`, `gru_loss_curves_gru_e80.png` |

</details>

---

## Detailed results

### Model selection

Every GRU version was compared on closed‑loop success over 200 episodes (val 63 + `sel` 137), since 63 alone is too few to separate models that all score 90–95%. **Lower offline NMSE didn't always mean better driving.**

| Version | Arm NMSE (test, offline) | Val (/63) | Sel (/137) |
|---|---|---|---|
| `gru` (main + DART 150) | 0.050 | 51 | – |
| `gru_dart2` (+ DART 500) | 0.034 | 58 | 127 |
| `gru_dagger1` (discarded) | 0.054 | 51 | 98 |
| `gru_nw10` (discarded) | 0.047 | 58 | 130 |
| **`gru_e80` (final)** | **0.027** | **62** | **133** |
| `gru_ens` (discarded) | 0.024 | 61 | 132 |

Retraining with a different seed moved the score by about ±5 out of 200, so the ensemble and the 10× near‑target weighting weren't real improvements. The final model scored 62/63 on val and 133/137 on `sel`.

### Generalisation

30 episodes per pool, none seen during training:

| Pool | GRU |
|---|---|
| `oor_far` (targets 2.5–3.5 m away) | 30 / 30 |
| `oor_high` (targets 1.10–1.30 m high) | 12 / 30 |
| `oor_start` (start 1.5–3.0 m out) | 29 / 30 |
| `oor_arm` (arm offsets up to 0.5 rad) | 28 / 30 |

### Robustness

Val split, 63 episodes, expert rerun under the same conditions:

| Condition | GRU | Expert |
|---|---|---|
| None | 62 | 63 |
| Actuator noise σ = 0.25 / 0.5 | 59 / 53 | 57 / 41 |
| Floor friction μ = 0.4 | 62 | 63 |
| Sensor noise σ = 0.005 / 0.01 / 0.02 rad | 62 / 62 / 63 | – |

A comparison against the MLP on the same test episodes is in `outputs/gru_vs_mlp.csv` (the `test + ...` rows are earlier robustness runs on the test split, kept for that comparison).

### Findings

- **DAgger made things worse** (98/137 on `sel` vs 127 without). The expert's torque depends on its own `q_des`, which winds up to its 0.15 rad limit while the GRU is driving, so the labels it records are one long saturated push rather than what it would normally do from that state.
- **The slow test episodes** are ones where the base hit the 25 s navigation timeout and the arm then reached from wherever the base stopped. Mean time to reach is 11.8 s (10.0 s excluding those 6).
- **`dart2` settings were rebuilt** from `outputs/plan_dart2.png`: noise 0.5, IDs 710–1209. `sel` uses IDs 1210–1346.

---

## Output files

```
data/<pool>/plan.csv                    planned episodes
data/<pool>/ep#####.parquet             expert episode, one row per 1 ms step
data/<pool>/summary.csv                 one row per expert episode
data/<pool>/dataset_*.npz               model inputs (X) and labels (y)
splits/*_ids.txt                        episode IDs for each split
models/<model>_<dataset>.joblib         trained models
outputs/results_offline.csv             offline scores (val split)
outputs/closed_loop_<NAME>.csv          closed-loop result per episode
outputs/closed_loop/<NAME>/*.parquet    closed-loop step logs (incl. expert's commands)
outputs/*.png                           coverage, outcome and diagnosis plots
```

Collected data and trained models are included in the repository, so you can skip straight to evaluation. To rebuild everything from scratch, follow steps 1–6 above.

---

## Authors and contributions

| Author | Contribution |
|---|---|
| **Noah** | Core pipeline: `robotnik_arm_task.py`, `pipeline.py`, `splits.py`, `training.py`, `closed_loop.py`, `test.py` |
| **Finn** | MLP policy and evaluation: `mlp.py`, `mlp_eval.py` |
| **Sam** | GRU policy: `gru.py` (final model) |

`robotnik_arm_task.py` was extended to accept a policy for training and closed‑loop testing.

**Tools and AI use.** The core pipeline was built from scikit‑learn, pandas and NumPy tutorials and earlier course labs, with minimal AI use. Claude Code assisted with the MLP files, and AI tools assisted with the GRU.
