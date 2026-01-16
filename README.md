# Sentiment Classification with Word Embeddings

Machine Learning course project (HW4) implementing sentiment classification using word2vec embeddings vs. sparse one-hot features.

## Project Overview
Classification of movie reviews as positive/negative sentiment using:
- Dense word2vec embeddings (300-dim)
- Averaged perceptron with pruning
- k-NN and SVM comparison
- Systematic evaluation: one-hot vs. embeddings

## Key Results
- **Best Dev Error:** 23.4% (SVM with embeddings)
- **Best Kaggle Public:** 22.4% (Averaged perceptron)
- **Improvement:** 40.2% → 27.8% dev error (k-NN: one-hot → embeddings)

## Files Guide

### Main Implementation
- `ML_Proj_4ipynb.ipynb` - Complete implementation
  - **Part 1:** Word embedding exploration (similarity, analogies)
  - **Part 2:** Sentence embeddings, k-NN, perceptron implementation
  - **Part 3:** SVM comparison

### Key Sections to Review

**Sentence Embedding Implementation** (Part 2):
- Lines 400-450: `make_sentence_embedding()` function with OOV handling
- Lines 500-600: Averaged perceptron with smart averaging
- Lines 650-700: Vocabulary pruning implementation

**Model Comparison** (Part 3):
- Lines 1195-1250: SVM implementation and comparison

## Technical Approach

**Feature Engineering:**
- Implemented sentence embeddings by averaging word vectors
- Handled OOV tokens by checking vocabulary before averaging
- Applied vocabulary pruning (removed low-frequency words)

**Models Trained:**
- k-NN (experimented with k=1,3,...99, optimal k=73)
- Perceptron (basic + averaged variant)
- SVM with RBF kernel

**Key Learning:**
Embeddings capture semantic similarity that one-hot features miss—similar words pull sentences closer even without exact token matches.

## Running the Code
```bash
# Install dependencies
pip install gensim numpy pandas scikit-learn

# Open notebook
jupyter notebook sentiment_word_embeddings.ipynb
```

## Dependencies
- Python 3.8+
- gensim 4.4.0
- numpy
- pandas  
- scikit-learn

## Results Summary

| Method | One-hot Dev Error | Embedding Dev Error | Kaggle Public |
|--------|-------------------|---------------------|---------------|
| k-NN | 40.2% | 27.8% | 26.2% |
| Perceptron (avg) | 26.3% | 24.2% | 22.4% |
| SVM | N/A | 23.4% | N/A |

## Author
Revanth Mudavath  
Oregon State University - CS534 Machine Learning (Fall 2024)
