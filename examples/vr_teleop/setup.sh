#!/bin/bash
# One-time setup for the VR teleop demo: fetches the mujoco_playground G1
# scene, the menagerie G1 meshes it references, and the v28 checkpoint.
set -e
cd "$(dirname "$0")"

if [ ! -d playground_g1 ]; then
  echo "fetching mujoco_playground G1 scene..."
  git clone -q --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_playground pg_tmp
  git -C pg_tmp sparse-checkout set mujoco_playground/_src/locomotion/g1
  mv pg_tmp/mujoco_playground/_src/locomotion/g1 playground_g1
  rm -rf pg_tmp
fi

if [ ! -d menagerie_g1 ]; then
  echo "fetching mujoco_menagerie unitree_g1 meshes..."
  git clone -q --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie men_tmp
  git -C men_tmp sparse-checkout set unitree_g1
  mv men_tmp/unitree_g1 menagerie_g1
  rm -rf men_tmp
fi

# repoint the playground XMLs' menagerie mesh paths at the local clone
python3 - <<'EOF'
import glob
for p in glob.glob("playground_g1/xmls/*.xml"):
    s = open(p).read()
    s2 = s.replace("../../../../../mujoco_menagerie/unitree_g1/assets/",
                   "../../menagerie_g1/assets/")
    if s2 != s:
        open(p, "w").write(s2)
        print(f"patched mesh paths: {p}")
EOF

mkdir -p checkpoints/velocity_v28
for f in env.yaml policy.safetensors; do
  if [ ! -f "checkpoints/velocity_v28/$f" ]; then
    echo "fetching v28 checkpoint: $f"
    curl -sL -o "checkpoints/velocity_v28/$f" \
      "https://huggingface.co/haixuantao/unitree-g1-velocity-v28/resolve/main/velocity_v28_iter12300/$f"
  fi
done

echo "setup complete — run: python3 g1_vr_stand.py --host <robot-ip>"
