# Word2Vec Implementation

This folder contains PyTorch implementations of the Word2Vec models.

## Models Implemented

### 1. CBOW (Continuous Bag of Words)
- Implemented in `cbow_model.py`
- Predicts a target word given its context words
- Uses context window of 4 words (2 before and 2 after the target word)
- Includes word embedding layer of configurable size (default 300 dimensions)

### 2. Skip-gram
- Implemented in `skip_gram_model.py`
- Predicts context words given a target word
- Reverse architecture of CBOW
- Optimized for better performance on infrequent words


## References

- Efficient Estimation of Word Representations in Vector Space. arXiv preprint https://arxiv.org/abs/1301.3781
