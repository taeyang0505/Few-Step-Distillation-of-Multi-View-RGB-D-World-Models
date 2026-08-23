# 논문 초안 (draft v1, 2026-08-23)

Geo4D 논문 구성을 따른 초안. 본문은 `sections/S1..S7`(마크다운), 그림은 `figures/`, 조립본은 `draft_v1.docx`.
수치는 전부 `FACTS.md`(= `results/quantitative/`에서 옮긴 사실 시트)에서만 가져왔고, 미실행 항목(자기 앵커 loss, 정책 대리 지표)은 본문에 "not run"으로 적었다.

- 구글 문서본: https://docs.google.com/document/d/13cBBLYDuXbG-677QfgVWl67yI0iLC_wZtZzzfz7fSi4/edit
- 다시 조립: `python assemble.py` (docx), `python to_html.py` (구글 문서 붙여넣기용 HTML). python-docx, matplotlib, pillow 필요.
- 그림 재생성: `figures/make_fig*.py` (결과 파일 경로는 스크립트 상단).

확인할 것: 소속 표기(NAIS Lab), 참고문헌 중 arXiv id로만 확인한 항목(DreamDojo, RoboWorld, MAGI-1, StreamDiT, TeaCache, MagCache, SVDQuant)의 학회명.
