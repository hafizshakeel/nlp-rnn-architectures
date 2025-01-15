"""
Natural Language Processing Model Implementation

Efficient Estimation of Word Representations in Vector Space
(Skip-gram) Implementation: https://arxiv.org/abs/1301.3781

This script contains the implementation of the Skip-gram model.

Implementation by: Hafiz Shakeel Ahmad Awan
Email: hafizshakeel1997@gmail.com
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim
import spacy
import json
from utils import predict_context_words


class SkipGram(nn.Module):
    def __init__(self, embedding_size=300, vocab_size=-1):
        super().__init__()
        # Embedding layer: maps input word (center word) indices to embedding vectors
        self.embeddings = nn.Embedding(vocab_size, embedding_size)  
        # Linear layer: maps center word embedding to vocab size for prediction
        self.linear = nn.Linear(embedding_size, vocab_size)  

    def forward(self, center_word):
        # center_word: Input tensor with shape (batch_size, 1), representing the center word in the SkipGram model
        embeddings = self.embeddings(center_word)  # Shape: (batch_size, embedding_size)
        # Pass the embedding through the linear layer to predict context words in the vocab
        return self.linear(embeddings)  # Shape: (batch_size, vocab_size)
        # --> Output: A probability distribution across the vocabulary for each input center word


def create_dataset():
    # Similar to CBOW, but here we will create a dataset where each center word predicts multiple context words.
    with open("raw_text.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    nlp = spacy.load("en_core_web_sm")
    tokenized_text = [token.text for token in nlp(raw_text)]
    vocab = set(tokenized_text)

    word_to_idx = {word: i for i, word in enumerate(vocab)}
    idx_to_word = {i: word for i, word in enumerate(vocab)}

    # Generates data with center words predicting context words
    data = []
    for i in range(2, len(tokenized_text) - 2):
        center_word = tokenized_text[i]
        context_words = [
            tokenized_text[i - 2],
            tokenized_text[i - 1],
            tokenized_text[i + 1],
            tokenized_text[i + 2],
        ]  # given the center word i , what will be 4 context words?

        center_idx = word_to_idx[center_word]
        context_idxs = [word_to_idx[w] for w in context_words]

        for ctx_idx in context_idxs:
            data.append((center_idx, ctx_idx))

    return data, word_to_idx, idx_to_word


def save_mappings(word_to_idx, idx_to_word):
    # Convert idx_to_word keys to integer for consistency
    idx_to_word_int = {int(k): v for k, v in idx_to_word.items()}
    with open("mappings_skip.json", "w") as f:
        json.dump({"word_to_idx": word_to_idx, "idx_to_word": idx_to_word_int}, f)


def load_mappings():
    if not os.path.isfile("mappings_skip.json"):
        raise FileNotFoundError(
            "The mappings.json file does not exist. Make sure to train the model and save the mappings first.")

    with open("mappings_skip.json", "r") as f:
        mappings = json.load(f)
        word_to_idx = mappings["word_to_idx"]
        idx_to_word = {int(k): v for k, v in mappings["idx_to_word"].items()}
        return word_to_idx, idx_to_word


def load_skip_model(embedding_size, vocab_size):
    model = SkipGram(embedding_size=embedding_size, vocab_size=vocab_size)
    model.load_state_dict(torch.load("skipgram_model.pth"))
    model.eval()
    return model


def main():
    EMBEDDING_SIZE = 300
    data, word_to_idx, idx_to_word = create_dataset()
    save_mappings(word_to_idx, idx_to_word)
    loss_fn = nn.CrossEntropyLoss()
    net = SkipGram(embedding_size=EMBEDDING_SIZE, vocab_size=len(word_to_idx))
    optimizer = optim.Adam(net.parameters(), lr=1e-4)

    # use center words as input and context words as labels.
    center_data = torch.tensor([ex[0] for ex in data], dtype=torch.long)
    context_data = torch.tensor([ex[1] for ex in data], dtype=torch.long)

    dataset = torch.utils.data.TensorDataset(center_data, context_data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(100):
        for center, context in dataloader:
            output = net(center)  # predicts multiple context words from a center word.
            loss = loss_fn(output, context)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("Epoch: {}, loss: {}".format(epoch, loss.item()))

    print("saving checkpoints")
    torch.save(net.state_dict(), "skipgram_model.pth")
    return net, word_to_idx, idx_to_word


if __name__ == "__main__":
    train = False
    if train:
        net = main()
    else:
        print("Loading checkpoints...")
        word_to_idx, idx_to_word = load_mappings()
        net = load_skip_model(embedding_size=300, vocab_size=len(word_to_idx))
        center_word = "about"
        predicted_context_words = predict_context_words(net, center_word, word_to_idx, idx_to_word)
        print(f"Predicted context words: {predicted_context_words}")
