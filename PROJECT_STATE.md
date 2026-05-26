# Project State

## Project
**Multi-Method ML Research Agent**

## Goal
Build an end-to-end research intelligence system across:
traditional ML → deep learning → Transformer → BERT → RAG → LLM report generation.

## Current Stage
**Stage 2 completed; next step Stage 3 Mini Transformer from scratch.**

## Roadmap Status
- [x] Stage 0: Project setup
- [x] Stage 1: Traditional ML baseline
- [x] Stage 2: PyTorch deep learning baseline
- [ ] Stage 3: Mini Transformer from scratch
- [ ] Stage 4: BERT fine-tuning
- [ ] Stage 5: RAG document QA system
- [ ] Stage 6: LLM-based report generation
- [ ] Stage 7: Streamlit demo and interview packaging

## Stage 1 Snapshot
- Task: research task classification.
- Dataset: `data/sample/research_queries_sample.csv`.
- Dataset specs: 70 rows, 7 balanced research-task categories, columns `text`, `label`.
- Features/models: TF-IDF + Logistic Regression / Naive Bayes / Random Forest.
- Best Stage 1 model: Logistic Regression.
- Best Stage 1 macro F1: ~0.852.
- Reports output directory: `reports/stage1`.

## Stage 2 Snapshot
- Reused Stage 1 dataset and label system for fair comparison.
- Implemented PyTorch Dataset/DataLoader pipeline.
- Implemented `MeanPoolingTextClassifier` (Embedding + masked mean pooling + MLP).
- Trained with `CrossEntropyLoss` and Adam optimizer.
- Stage 2 macro F1 on this small demo split: ~0.495.
- Reports output directory: `reports/stage2`.
- This is a small neural baseline for learning/comparison, not a production benchmark.

## Next Step
Stage 3 Mini Transformer from scratch planning/implementation.
