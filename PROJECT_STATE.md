# Project State

## Project
**Multi-Method ML Research Agent**

## Goal
Build an end-to-end research intelligence system across:
traditional ML → deep learning → Transformer → BERT → RAG → LLM report generation.

## Current Stage
**Stage 1 (Traditional ML baseline): Completed**

## Roadmap Status
- [x] Stage 0: Project setup
- [x] Stage 1: Traditional ML baseline
- [ ] Stage 2: PyTorch deep learning model
- [ ] Stage 3: Mini Transformer from scratch
- [ ] Stage 4: BERT fine-tuning
- [ ] Stage 5: RAG document QA system
- [ ] Stage 6: LLM-based report generation
- [ ] Stage 7: Streamlit demo and interview packaging

## Stage 1 Snapshot
- Stage 1 runner now supports real CSV datasets via CLI arguments.
- Added sample dataset: `data/sample/research_queries_sample.csv`
- Dataset specs: **70 rows**, **7 balanced research-task categories**, columns: `text`, `label`.
- Validation command run successfully:
  `python scripts/run_stage1_ml_baseline.py --data data/sample/research_queries_sample.csv --text_col text --label_col label`
- Best model on sample dataset: **Logistic Regression**
- Best macro F1: **~0.852**
- Reports output directory: `reports/stage1`

## Next Step (for tomorrow)
Start **Stage 2** planning/implementation (PyTorch deep learning baseline), reusing the Stage 1 dataset interface and report structure for consistent comparisons.
