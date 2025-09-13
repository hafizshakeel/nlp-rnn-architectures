# Basic Sequence-to-Sequence Neural Machine Translation

This folder contains an implementation of the basic sequence-to-sequence model for machine translation, based on the paper ["Sequence to Sequence Learning with Neural Networks"](https://arxiv.org/abs/1409.3215).

## Architecture Overview

### Encoder
- LSTM-based encoder
- Processes source sentences sequentially
- Produces fixed-length context vector
- Multiple layers with dropout for regularization

### Decoder
- LSTM decoder using encoder's final state
- Takes previous prediction as input at each step
- Maps hidden states to vocabulary distribution
- Uses teacher forcing during training

## Implementation Details

- `seq2seq_lang_translation.py`: Main implementation file
- `utils.py`: Helper functions for training and evaluation
- Uses Multi30k dataset (German to English translation)
- Includes BLEU score evaluation

## Usage

Train the model:
```python
python seq2seq_lang_translation.py
```

## Visual Reference
See `seq2seq_language_translation.png` for architecture diagram.

## Comparison with Attention Model
This implementation serves as a baseline to demonstrate the benefits of adding attention mechanisms. Key differences from the attention model:
- No explicit alignment between source and target
- Fixed context vector as bottleneck
- Generally lower performance on longer sequences