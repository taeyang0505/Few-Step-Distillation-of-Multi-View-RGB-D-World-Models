#!/bin/bash
# H3(3호출: student 2 + teacher 1)·H4(4호출) 하이브리드 평가. 스모크(2배치) 통과 후에만 본 실행.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate video_policy
cd ~/4dgen
echo "[smoke] H3b 2배치"
python notebooks/bench_precise_6a.py --student_ckpt ~/Geo4D/dmd_6a/dmd_gen_step1600.pt --configs T25 H3b --n_batches 2 --n_div 1 --fast --tag _hyb_smoke > ~/hyb_smoke.log 2>&1
if ! grep -q "hybrid" ~/hyb_smoke.log || grep -q "Traceback" ~/hyb_smoke.log; then
  echo "SMOKE_FAILED"; tail -30 ~/hyb_smoke.log; exit 1
fi
echo "[smoke] OK — 본 실행"
python notebooks/bench_precise_6a.py --student_ckpt ~/Geo4D/dmd_6a/dmd_gen_step1600.pt --configs T25 S3b H3b H4b --n_batches 20 --n_div 3 --fast --tag _hybrid
python notebooks/bench_policy_proxy.py --student_ckpt ~/Geo4D/dmd_6a/dmd_gen_step1600.pt --configs T25 H3b H4b --n_batches 20 --fast --tag _hybrid
echo HYBRID_DONE
