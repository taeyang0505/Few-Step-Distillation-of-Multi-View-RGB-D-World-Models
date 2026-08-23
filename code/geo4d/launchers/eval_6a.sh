#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate video_policy
cd ~/4dgen
while pgrep -f "[g]eo4d_dmd_train" >/dev/null; do sleep 30; done
sleep 10
for s in 1000 1600; do
  python notebooks/bench_student_sweep.py --student_ckpt ~/Geo4D/dmd_6a/dmd_gen_step$s.pt --sampler renoise --steps 3 2 1 --tag _dmd6a_s$s
done
python notebooks/bench_student_sweep.py --student_ckpt ~/Geo4D/dmd_6a/dmd_gen.pt --sampler renoise --steps 3 2 1 --tag _dmd6a_s2000
echo EVAL_CHAIN_DONE
