#!/bin/bash
# E1 게이트: lbm_eval 설치(CPU, Docker 불필요) + 태스크 목록에서 우리 3개 태스크 존재 확인
# 디스크 ~6G, 소요 ~20-40분(다운로드 포함). GPU 무관 — 지금 실행 가능.
set -e
source ~/miniconda3/etc/profile.d/conda.sh
if ! conda env list | grep -q "^lbm_eval "; then
  conda create -y -n lbm_eval python=3.12
fi
conda activate lbm_eval
mkdir -p ~/lbm_eval_wheels && cd ~/lbm_eval_wheels
BASE=https://github.com/ToyotaResearchInstitute/lbm_eval/releases/download/1.1.0
for w in robot_gym-1.1.0-py3-none-any.whl lbm_eval-1.1.0-py3-none-any.whl lbm_eval_models-1.1.0-py3-none-any.whl lbm_eval_scenarios-1.1.0-py3-none-any.whl; do
  [ -f $w ] || wget -q --show-progress $BASE/$w
done
pip install -q robot_gym-1.1.0-py3-none-any.whl lbm_eval-1.1.0-py3-none-any.whl lbm_eval_models-1.1.0-py3-none-any.whl lbm_eval_scenarios-1.1.0-py3-none-any.whl
echo "== 설치 완료. 태스크(스킬) 목록:"
evaluate --skill_type=help 2>&1 | tee ~/lbm_eval_skills.txt | head -60
echo "== 우리 3개 태스크 검색:"
grep -i "apple\|cereal\|spatula" ~/lbm_eval_skills.txt || echo "NOT_FOUND_IN_LIST"
echo E1_SETUP_DONE
