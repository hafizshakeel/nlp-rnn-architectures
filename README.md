# NLP RNN Architectures

A collection of PyTorch implementations for various Recurrent Neural Network (RNN) architectures and their applications in Natural Language Processing.

## Repository Structure

### 1. RNN Basics ([RNN BASICS/](RNN%20BASICS/))
- Fundamental RNN architectures implementation
- Includes RNN, GRU, LSTM, and Bidirectional LSTM
- Example application using MNIST dataset
- Visualizations of each architecture

### 2. Word2Vec Models ([Word2Vec/](Word2Vec/))
- Implementation of CBOW and Skip-gram architectures
- Based on "Efficient Estimation of Word Representations in Vector Space"
- Includes word embedding generation and context prediction
- Training and evaluation utilities

### 3. Sequence-to-Sequence Models ([SEQ2SEQ MODELS/](SEQ2SEQ%20MODELS/))
- Multiple seq2seq implementations:
  - Basic seq2seq for machine translation
  - Seq2seq with attention mechanism
  - Image captioning using CNN-LSTM architecture

## Getting Started

1. Clone the repository:
```bash
git clone https://github.com/hafizshakeel/nlp-rnn-architectures.git
cd nlp-rnn-architectures
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Navigate to specific implementations:
```bash
# For RNN basics
cd RNN\ BASICS/
python rnn_gru_lstm_bilstm_models.py

# For Word2Vec models
cd Word2Vec/
python cbow_model.py  # or skip_gram_model.py

# For Seq2Seq models
cd SEQ2SEQ\ MODELS/
cd image_captioning/  # or seq2seq_with_attention/ or seq2seq_without_attention/
```

## Requirements
- Python 3.6+
- PyTorch
- See requirements.txt for complete list of dependencies

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
Email: hafizshakeel1997@gmail.com