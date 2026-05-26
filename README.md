# Multi-Method ML Research Agent

An end-to-end research intelligence system that integrates traditional machine learning, deep learning, Transformer-based models, retrieval-augmented generation, and LLM-based report generation.

## 1. Project Motivation

This project is designed as a hands-on learning and engineering project for building a complete machine learning and AI application pipeline.

Instead of only calling a large language model API, this project starts from traditional machine learning baselines and gradually extends to deep learning, Transformer models, BERT fine-tuning, RAG, and LLM-based report generation.

## 2. Core Tasks

- Text classification
- Sentiment or risk analysis
- Traditional machine learning baselines
- PyTorch deep learning models
- Mini Transformer implementation from scratch
- BERT fine-tuning
- Document retrieval
- RAG-based question answering
- LLM-based research summary generation
- Streamlit demo application

## 3. Technical Roadmap

### Stage 0: Project Setup

- Create repository and structure
- Set up Python environment
- Add README and project tracking

### Stage 1: Traditional ML Baselines

- Task: research task classification
- Data loading from CSV via CLI
- TF-IDF features + Logistic Regression / Naive Bayes / Random Forest
- Evaluation and reports in `reports/stage1`

### Stage 2: PyTorch Neural Baseline (Completed)

- Reused the same Stage 1 dataset: `data/sample/research_queries_sample.csv`
- Same columns/labels (`text`, `label`) for fair comparison
- Pipeline: tokenizer → vocabulary → token ids → padding → Dataset/DataLoader
- Model: Embedding → masked mean pooling → MLP classifier
- Training: epoch-level train/validation loss and metrics
- Outputs saved in `reports/stage2`
- Note: Stage 2 is an educational neural baseline on a small demo dataset and may not outperform Stage 1 Logistic Regression

### Stage 3: Mini Transformer from Scratch (Upcoming)

- Token embedding + position embedding
- Self-attention, multi-head attention
- Feed-forward network, residual, layer norm
- Transformer encoder block

## 4. Current Status

Current stage: **Stage 2 completed / Stage 3 upcoming**.

Completed:
- Stage 0 setup
- Stage 1 traditional ML baseline for research task classification
- Stage 2 PyTorch neural text classification baseline

Stage 2 outputs:
- `reports/stage2/model_results.csv`
- `reports/stage2/classification_report.txt`
- `reports/stage2/training_log.csv`
- `reports/stage2/stage2_summary.txt`

## 5. Learning Goals

- End-to-end ML workflow
- Traditional ML and neural NLP baselines
- Transformer architecture understanding
- Practical model comparison and reporting

## 6. Tech Stack

- Python
- pandas
- NumPy
- scikit-learn
- PyTorch
- HuggingFace Transformers (planned)
- FAISS/BM25 (planned)
- Streamlit (planned)
