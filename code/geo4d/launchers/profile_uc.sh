#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate video_policy
cd ~/4dgen
python notebooks/bench_profile_student.py 2>&1 | grep -a "^\[\|^   \|^=====\|Error"
echo PROFILE_DONE
