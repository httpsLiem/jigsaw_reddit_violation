# jigsaw-reddit-violation

**Solution for Kaggle Competition: Jigsaw - Agile Community Rules**  
[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-blue)](https://www.kaggle.com/competitions/jigsaw-agile-community-rules)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview
This repository contains a complete solution for the **Jigsaw - Agile Community Rules** Kaggle competition. The task is to classify whether a Reddit comment violates a given subreddit rule (binary classification: `rule_violation` probability).

### Key Highlights
- **Data**: `train.csv` and `test.csv` with columns: `body` (comment), `rule`, `subreddit`, `rule_violation` (label in train), and few-shot examples (`positive_example_1/2`, `negative_example_1/2`).
- **Approach**:
  - **Data Augmentation**: Flatten few-shot examples from train/test into training data. Extract URL semantics as keywords.
  - **Models**: Fine-tune multiple Transformers (DistilBERT, ALBERT, DeBERTa, etc.) + LLM inference with Qwen 14B (GPTQ quantized) using LoRA.
  - **Semantic Search**: Use embedding-based search for rule-comment similarity.
  - **Ensemble**: Rank-based blending of 7+ submissions with weighted averages for final prediction.
  - **Probing**: Tool to probe submissions (invert predictions for gain analysis).
- **Performance**: Achieved strong leaderboard score via ensemble (exact LB position depends on run; designed for top-tier blending).
- **Environment**: Tested on Kaggle GPU (Tesla T4 x2) with PyTorch, Transformers, vLLM.

The solution is structured for reproducibility, modularity, and easy extension.

## Project Structure
