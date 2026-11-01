# RAG Search Engine

A small project for learning how search works and how Retrieval Augmented Generation (RAG) can improve search experiences.

## Purpose

This project explores the foundations of building a search system for a content-based application, using movie data as the dataset.

## Setup

1. Create and activate a virtual environment
2. Install dependencies with `uv`
3. Add the movie dataset to `data/movies.json`

## Notes

- `data/` is ignored by Git
- `.venv/` is ignored by Git

## Future Updates

- [ ] **Evaluation:** Build a small golden test set and compare re-rank methods using metrics like MRR and nDCG
- [ ] **Fine-tuning:** Fine-tune the cross-encoder on movie query-document pairs (including hard negatives) and check whether it beats the stock model
- [ ] **Nice-to-haves:** Try a larger base cross-encoder (e.g. `ms-marco-MiniLM-L6-v2`) to compare speed and quality. Use LLM-as-judge to generate extra training labels (spot-check by hand). Cache cross-encoder scores for repeated queries.