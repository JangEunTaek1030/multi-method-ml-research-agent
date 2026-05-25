# Multi-Method ML Research Agent

An end-to-end research intelligence system that integrates traditional machine learning, deep learning, Transformer-based models, retrieval-augmented generation, and LLM-based report generation.

## 1. Project Motivation

This project is designed as a hands-on learning and engineering project for building a complete machine learning and AI application pipeline.

Instead of only calling a large language model API, this project starts from traditional machine learning baselines and gradually extends to deep learning, Transformer models, BERT fine-tuning, RAG, and LLM-based report generation.

The goal is not only to build a working AI application, but also to understand the key ideas behind modern machine learning systems.

## 2. Core Tasks

The system aims to support the following tasks:

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

- Create GitHub repository
- Build project structure
- Set up Python environment
- Add README and project state tracking files

### Stage 1: Traditional Machine Learning Baselines

- Data loading
- Text preprocessing
- TF-IDF feature extraction
- Logistic Regression
- Random Forest
- XGBoost / LightGBM
- Model evaluation

### Stage 2: Deep Learning Models

- PyTorch Dataset and DataLoader
- Embedding layer
- MLP / TextCNN
- Training loop
- Validation loop
- Loss curve visualization

### Stage 3: Mini Transformer from Scratch

- Token embedding
- Position embedding
- Self-attention
- Multi-head attention
- Feed-forward network
- Residual connection
- LayerNorm
- Transformer encoder block

### Stage 4: BERT Fine-tuning

- HuggingFace tokenizer
- Input IDs and attention mask
- CLS representation
- Fine-tuning for text classification
- Model comparison with previous baselines

### Stage 5: RAG System

- Document chunking
- Embedding generation
- FAISS vector search
- BM25 retrieval
- Hybrid search
- Reranking
- Context construction
- Question answering with source references

### Stage 6: LLM-based Research Assistant

- Prompt engineering
- Research summary generation
- Report generation
- Streamlit demo
- Final GitHub project packaging

## 4. Project Structure

    multi-method-ml-research-agent/
    |
    |-- data/
    |   |-- raw/
    |   |-- processed/
    |
    |-- notebooks/
    |
    |-- src/
    |   |-- data/
    |   |-- features/
    |   |-- models/
    |   |-- training/
    |   |-- evaluation/
    |   |-- rag/
    |   |-- llm/
    |
    |-- scripts/
    |
    |-- tests/
    |
    |-- docs/
    |   |-- study_notes/
    |
    |-- app/
    |
    |-- README.md
    |-- PROJECT_STATE.md
    |-- requirements.txt
    |-- .gitignore

## 5. Current Status

Current stage: Stage 0 - project setup.

Completed:

- Project idea defined
- Technical roadmap defined
- Initial project structure designed
- README v0 created

Next step:

- Set up Python environment
- Add initial project files
- Make the first Git commit
- Start Stage 1: traditional machine learning baseline

## 6. Learning Goals

Through this project, I aim to understand and implement:

- End-to-end machine learning workflow
- Data preprocessing and feature engineering
- Traditional ML model training and evaluation
- Deep learning training loop with PyTorch
- Transformer architecture from scratch
- BERT fine-tuning for downstream NLP tasks
- Retrieval-augmented generation
- LLM application engineering
- GitHub-based project organization

## 7. Interview Narrative

This project demonstrates a complete learning and engineering path from traditional machine learning to modern LLM applications.

Instead of only using an LLM API, the project implements and compares multiple modeling approaches, including TF-IDF based traditional ML models, PyTorch neural networks, a Mini Transformer from scratch, BERT fine-tuning, and RAG-based document question answering.

The project is designed to show both practical engineering ability and conceptual understanding of modern AI systems.

## 8. Tech Stack

Planned technical stack:

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- LightGBM
- PyTorch
- HuggingFace Transformers
- FAISS
- BM25
- Streamlit
- LLM API

## 9. Version Plan

### Version 1.0

- Traditional ML baseline
- Basic deep learning model
- Model evaluation table
- Clean README

### Version 2.0

- Mini Transformer from scratch
- BERT fine-tuning
- RAG document QA system
- Streamlit demo

### Version 3.0

- LLM-based report generation
- Better retrieval evaluation
- Model interpretation
- Final interview notes and project packaging
