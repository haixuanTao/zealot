# Arm-motion dataset (`~/sonic-motions`)

Retargeted mocap clips (LAFAN1 → G1 joint space, from the public
[lvhaidong/LAFAN1_Retargeting_Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset),
converted to the GR00T-WholeBodyControl / SONIC export format) that training replays on the PD-held waist + arm joints as
an **unobserved moving-mass disturbance** — under both stand and walk
commands, the legs must balance while the upper body gestures. This is *not*
an imitation objective: the policy never observes or controls the upper body.

## How training consumes it

Arm motion is **on by default**: when `BIPED_ARM_MOTION` is unset the env
itself uses `$HOME/sonic-motions` if it exists (and logs that it did) — a
box without the dataset used to silently train with frozen arms.
`scripts/train.sh` additionally refuses to launch when the dataset is
missing. Options:

- `BIPED_ARM_MOTION=<dir>` — use a dataset at another path.
- `BIPED_ARM_MOTION=off` (or `0`) — explicitly opt out (arms hold the home
  pose).

Every env construction logs one of `arm-motion playback ENABLED: N clips …`
or `arm-motion playback DISABLED …`, so any training log records which mode
it ran in. Knobs (see `src/biped/biped_env_nexus.rs`):

- `BIPED_ARM_MOTION_P` (default 0.7) — probability that each command window
  plays a clip; rolled on every command resample, stand and walk alike.
- `BIPED_ARM_MOTION_SCALE` (default 1.0) — amplitude blend home→clip;
  lower it if full-amplitude clips topple early training.
- `BIPED_ARM_MOTION_FPS` (default 30) — clip frame rate.

## File format

One CSV per clip (any filename, all files in the directory are loaded).
Header row:

```
Frame,root_translateX,root_translateY,root_translateZ,root_rotateX,root_rotateY,root_rotateZ,<joint>_dof,...
```

Angles are **degrees**, root translation centimetres (root columns are
ignored here). Joint columns are matched to the model's held joints by name
as `<name>_dof` (e.g. `left_shoulder_pitch_joint_dof`); extra columns the
model doesn't have are ignored, but a held joint *missing* from the file is
a hard error at load. Parser: `zealot-env/src/motion.rs`.

## Getting the data

1. **Copy from a machine that already has it** (any training box with the
   dataset, or the G1's Orin, which runs the full GR00T-WholeBodyControl
   stack):

   ```sh
   rsync -av <box>:~/sonic-motions/ ~/sonic-motions/
   ```

2. **Regenerate from source** (no token needed — the dataset is public):

   ```sh
   python3 scripts/make_motions.py   # downloads + converts into /workspace/sonic-motions
   ```

   The script pulls the walk/run/sprint clips from the LAFAN1 G1 retargeting
   dataset and converts headerless-radians CSV to the header/degrees/cm
   format above.

Sanity-check a copy before a long run — the loader validates joint columns,
and a one-clip smoke test replays in MuJoCo via
`S2S_ARM_MOTION=<clip.csv> scripts/sim2sim_g1_mujoco.py`:

```sh
head -1 ~/sonic-motions/*.csv | grep -c shoulder   # every file should hit
```
