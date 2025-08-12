"""
Natural Language Processing Model Implementation

Efficient Estimation of Word Representations in Vector Space
(CBOW) Implementation: https://arxiv.org/abs/1301.3781

This script contains the implementation of the CBOW model.

Implementation by: Hafiz Shakeel Ahmad Awan
Email: hafizshakeel1997@gmail.com
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim
import spacy
import json
from utils import predict_center_word


class CBOW(nn.Module):
    def __init__(self, embedding_size=300, vocab_size=-1):
        super().__init__()
        # Embedding layer: maps vocab indices to embedding vectors of size 'embedding_size'
        self.embeddings = nn.Embedding(vocab_size, embedding_size)  
        # --> The number of rows is the vocab size, and the number of columns is the embedding size.
        #     Each word index is mapped to a vector of size 'embedding_size'.
        
        # Linear layer: maps the mean of context word embeddings to the vocab size for prediction
        self.linear = nn.Linear(embedding_size, vocab_size)  
        # --> This predicts the center word, where vocab size is the number of possible center words.
        #     We will use cross-entropy loss for training to compare predictions with the true center word.

    def forward(self, context):
        # Inputs: context is a tensor of shape (batch_size, 4), where 4 is the number of context words
        embeddings = self.embeddings(context)  # Shape: (batch_size, 4, embedding_size)
        
        # Compute the average of context word embeddings across the context dimension (dim=1)
        embeddings = embeddings.mean(1)  # Shape: (batch_size, embedding_size)
        # --> Taking the mean of the context word embeddings reduces the dimension from 4 to 1,
        #     resulting in a tensor of size (batch_size, embedding_size).
        #     This step aggregates the information from the context words.

        # Pass the averaged embeddings through the linear layer to predict the center word
        return self.linear(embeddings)  # Shape: (batch_size, vocab_size)
        # --> Output shape is (batch_size, vocab_size), where each row represents the 
        #     prediction distribution over the vocab for the corresponding center word.


def create_dataset():
    # read text file from raw_text.txt as utf-8 in a string
    with open("raw_text.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    # tokenize raw_text with spacy lib
    nlp = spacy.load("en_core_web_sm")  # en_core_web_sm is a small English pipeline trained on written web text
    # (blogs, news, comments), that includes vocabulary, syntax and entities --> get dictionary
    tokenized_text = [token.text for token in nlp(raw_text)]
    vocab = set(tokenized_text)

    # create word to index and index to word mapping
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    idx_to_word = {i: word for i, word in enumerate(vocab)}

    # generate training data with two words as context and two words after as target
    data = []
    for i in range(2, len(tokenized_text) - 2):
        context = [
            tokenized_text[i - 2],
            tokenized_text[i - 1],
            tokenized_text[i + 1],
            tokenized_text[i + 2],
        ]
        target = tokenized_text[i]  # given 4 context words, what will be the center word i?

        # map context (str) and target to indices and append to data
        context_idxs = [word_to_idx[w] for w in context]
        target_idx = word_to_idx[target]
        data.append((context_idxs, target_idx))

    return data, word_to_idx, idx_to_word


# Save and Load the Mappings to ensure mapping consistency
# Use the same mappings for both training and testing.
def save_mappings(word_to_idx, idx_to_word):
    with open("mappings_cbow.json", "w") as f:
        json.dump({"word_to_idx": word_to_idx, "idx_to_word": idx_to_word}, f)


def load_mappings():
    if not os.path.isfile("mappings_cbow.json"):
        raise FileNotFoundError(
            "The mappings.json file does not exist. Make sure to train the model and save the mappings first.")

    with open("mappings_cbow.json", "r") as f:
        mappings = json.load(f)
        word_to_idx = mappings["word_to_idx"]
        idx_to_word = mappings["idx_to_word"]
        return word_to_idx, idx_to_word


def load_cbow_model(embedding_size, vocab_size):
    model = CBOW(embedding_size=embedding_size, vocab_size=vocab_size)
    # model.load_state_dict(torch.load("cbow_model.pth"))
    model.load_state_dict(torch.load("cbow_model.pth", weights_only=True))
    model.eval()
    return model


def main():
    EMBEDDING_SIZE = 300
    data, word_to_idx, idx_to_word = create_dataset()
    save_mappings(word_to_idx, idx_to_word)  # Save mappings for consistency while testing the model
    loss_fn = nn.CrossEntropyLoss()
    net = CBOW(embedding_size=EMBEDDING_SIZE, vocab_size=len(word_to_idx))
    optimizer = optim.Adam(net.parameters(), lr=1e-4)

    # use context words as input and target words as labels.
    context_data = torch.tensor([ex[0] for ex in data], dtype=torch.long)
    labels = torch.tensor([ex[1] for ex in data], dtype=torch.long)

    # create dataset for tensors x,y and dataloader
    dataset = torch.utils.data.TensorDataset(context_data, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # Run a single forward pass to print dimensions for CBOW
    # for context, labels in dataloader:
    #     output = net(context)
    #     break  # Exit after one batch for testing dimensions

    # train and save CBOW model
    for epoch in range(100):
        for context, labels in dataloader:
            output = net(context)  # predicts a single target word from context words.
            loss = loss_fn(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("Epoch: {}, loss: {}".format(epoch, loss.item()))

    print("saving checkpoints")
    torch.save(net.state_dict(), "cbow_model.pth")
    return net, word_to_idx, idx_to_word


if __name__ == "__main__":
    train = False
    if train:
        net = main()
    else:
        print("Loading checkpoints...")
        word_to_idx, idx_to_word = load_mappings()  # Load mappings
        net = load_cbow_model(embedding_size=300, vocab_size=len(word_to_idx))
        test_context = ["We", "are", "to", "study"]
        predicted_word = predict_center_word(net, test_context, word_to_idx, idx_to_word)
        print(f"Predicted center word: {predicted_word}")
