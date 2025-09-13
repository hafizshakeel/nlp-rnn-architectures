# Neural Machine Translation with Attention

This folder contains an implementation of the sequence-to-sequence model with attention mechanism for machine translation, based on the paper ["Neural Machine Translation by Jointly Learning to Align and Translate"](https://arxiv.org/abs/1409.0473).

## Architecture Overview

### Encoder
- Bidirectional LSTM encoder
- Processes source sentences in both forward and backward directions
- Creates richer context representations for attention mechanism
- Projects concatenated bidirectional states through linear layer

### Attention Decoder
- Computes attention weights for each encoder state
- Uses weighted sum of encoder states as context vector
- Combines context vector with current decoder state
- Single-layer LSTM with attention mechanism

## Implementation Details

- `seq2seq_lang_trans_attention.py`: Main implementation file
- `utils.py`: Helper functions for training and evaluation
- Uses Multi30k dataset (German to English translation)
- Includes BLEU score evaluation

## Usage

Train the model:
```python
python seq2seq_lang_trans_attention.py
```

## Visual Reference
See `seq2seq_lang_translation_attention.png` for architecture diagram.