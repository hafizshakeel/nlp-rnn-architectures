# Sequence-to-Sequence Models

This folder contains implementations of various sequence-to-sequence (Seq2Seq) architectures for different NLP tasks. Each subfolder contains a specific implementation:

## Contents

### 1. Image Captioning
- Implementation of an encoder-decoder architecture for generating natural language descriptions of images
- Uses CNN encoder (Inception v3) and LSTM decoder
- Based on the "Show and Tell" architecture
- See subfolder for detailed implementation

### 2. Seq2Seq with Attention
- Neural machine translation implementation using attention mechanism
- Based on paper "Neural Machine Translation by Jointly Learning to Align and Translate"
- Includes bidirectional encoder and attention decoder
- Trained on Multi30k dataset (German to English translation)

### 3. Seq2Seq without Attention
- Basic sequence-to-sequence implementation for machine translation
- Based on paper "Sequence to Sequence Learning with Neural Networks"
- Uses LSTM for both encoder and decoder
- Demonstrates the foundation of neural machine translation

## Common Features
- All implementations use PyTorch
- Include training scripts with hyperparameter configurations
- Support for checkpointing and model saving
- Evaluation metrics (BLEU score for translation tasks)
- Utility functions for data preprocessing and model evaluation