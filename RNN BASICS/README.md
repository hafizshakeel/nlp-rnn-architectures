# RNN Architectures Implementation

This folder contains implementations of fundamental Recurrent Neural Network (RNN) architectures for sequence processing tasks. The implementations include vanilla RNN, GRU, LSTM, and Bidirectional LSTM models.

## Overview

The `rnn_gru_lstm_bilstm_models.py` script provides clean, educational implementations of:

- **Vanilla RNN**: Basic recurrent neural network implementation
- **GRU (Gated Recurrent Unit)**: RNN variant with gating mechanisms to better handle long-term dependencies
- **LSTM (Long Short-Term Memory)**: Advanced RNN architecture with memory cells and gating mechanisms
- **BiLSTM (Bidirectional LSTM)**: Processes sequences in both forward and backward directions

## Implementation Details

- Each model is implemented using PyTorch's nn.Module
- Models are designed to work with sequence data (demonstrated using MNIST dataset)
- Input dimensions: sequence_length=28, input_size=28 (suited for MNIST rows)
- Configurable hyperparameters for hidden size, number of layers, etc.
- Includes training loop and accuracy evaluation


## Visual Aids

The `architecture_images` folder contains detailed diagrams of each architecture:
- RNN.png: Basic RNN cell structure
- GRU.png: Gated Recurrent Unit architecture
- LSTM.png: Long Short-Term Memory cell structure
- BiLSTM.jpg: Bidirectional LSTM architecture