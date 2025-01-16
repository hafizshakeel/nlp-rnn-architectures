"""
Neural Network Models Implementation for Natural Language Processing

This script provides implementations of several types of recurrent neural networks. The models include:

- RNN: A basic RNN that processes sequential data with a single hidden layer.
- GRU: A GRU network that leverages gating mechanisms to handle sequences.
- LSTM: An LSTM network that uses memory cells to retain information over longer sequences.
- BiLSTM: A Bidirectional LSTM that processes sequences in both forward and backward directions.

Each model is designed to work with sequences of length 28 and input features of size 28,
processing sequences through different types of recurrent neural networks to classify input data.

The script also includes code to train and evaluate the models using the MNIST dataset.

Implementation by: Hafiz Shakeel Ahmad Awan
Email: hafizshakeel1997@gmail.com
"""


# Imports
import torch  # Main PyTorch library
import torch.nn.functional as F  # Parameterless functions, like activation functions
import torchvision.datasets as datasets  # Standard datasets like MNIST, CIFAR-10, etc.
import torchvision.transforms as transforms  # Transformations we can perform on our dataset for augmentation
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn  # All neural network modules (like layers, loss functions)
from torch.utils.data import DataLoader  # Manages data loading, batching, shuffling, etc.
from tqdm import tqdm  # For a nice progress bar to track training progress

# Set device (CUDA for GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters for the RNN, GRU, and LSTM models
# Input Nx1x28x28 represents 28-time steps (sequences) with 28 features at each step (28x28 image)
input_size = 28  # Number of features (image row size)
sequence_length = 28  # Number of time steps (i.e., the height of the image, processing one row at a time)
hidden_size = 256  # Number of units in the hidden layer of RNN/GRU/LSTM
num_layers = 2  # Number of stacked RNN/GRU/LSTM layers
num_classes = 10  # Number of output classes (e.g., 10 classes for digit classification)
learning_rate = 0.001  # Learning rate for the optimizer
batch_size = 64  # Batch size for training
num_epochs = 2  # Number of times to iterate over the full dataset during training


""" Create a simple/basic RNN """

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size  # Size of the hidden layer
        self.num_layers = num_layers  # Number of RNN layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)  # RNN layer
        # batch_first=True --> (batch_size, sequence_length, input_size)

        self.fc = nn.Linear(hidden_size * sequence_length, num_classes)  # Fully connected layer to map to final output
        # Since we're concatenating all hidden states (28 in total), input to fc is hidden_size * sequence_length
        # If you only use the last hidden state, fc should have hidden_size as input as below.

        self.fc_lh = nn.Linear(hidden_size, num_classes)  # FC layer for using only the last hidden state (alternative approach)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(
            device)  # Initialize hidden state (h0) with zeros
        # h0: shape (num_layers, batch_size, hidden_size)
        # x.size(0): batch size; self.hidden_size: size of hidden state for each timestep

        out, _ = self.rnn(x, h0)  # Pass input (x) and hidden state (h0) through the RNN
        # out: RNN output for each time step (shape: batch_size, sequence_length, hidden_size)

        out_lh = out[:, -1, :]  # Extract the last hidden state (last time step) for each sample in the batch
        # out_lh: shape (batch_size, hidden_size) -> single hidden state for each example

        out = self.fc_lh(out_lh)  # Apply the fully connected layer to get the final output
        return out


""" Create a simple/basic GRU """

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size  # Size of the hidden layer
        self.num_layers = num_layers  # Number of GRU layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)  # GRU layer definition
        # Similar to RNN, but uses GRU instead, which is a type of gated RNN that avoids vanishing gradients

        self.fc = nn.Linear(hidden_size * sequence_length, num_classes)  # Fully connected layer to map to output

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)  # Initialize hidden state (h0)
        out, _ = self.gru(x, h0)  # Pass input through GRU
        out = out.reshape(out.shape[0], -1)  # Flatten the output for fully connected layer
        out = self.fc(out)  # Apply the fully connected layer to get final output
        return out


""" Create a simple/basic LSTM """

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size  # Size of the hidden layer
        self.num_layers = num_layers  # Number of LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)  # LSTM layer definition
        # LSTM is an advanced version of RNN that prevents vanishing gradients by having memory cell (cell state)
        self.fc = nn.Linear(hidden_size, num_classes)  # Fully connected layer for output from the last hidden state

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)  # Initialize hidden state (h0)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)  # Initialize cell state (c0)
        # c0: Cell state for LSTM, which is necessary in addition to hidden state

        out, _ = self.lstm(x, (h0, c0))  # Pass input through LSTM (hidden and cell states as inputs)
        out = self.fc(out[:, -1, :])  # Use only the last hidden state for final prediction (batch_size, hidden_size)
        return out


""" Create a simple/basic Bidirectional LSTM Not suitable for all types of data. Also slow training! """

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size  # Size of the hidden layer
        self.num_layers = num_layers  # Number of LSTM layers
        # BiLSTM processes data in both directions (forward and backward)
        self.bi_lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        # bidirectional=True allows LSTM to process in both forward and backward directions
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # Multiply by 2 due to bi-directionality

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)  # Initialize hidden state for both directions
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)  # Initialize cell state for both directions
        out, _ = self.bi_lstm(x, (h0, c0))  # Pass input through BiLSTM
        out = self.fc(out[:, -1, :])  # Use the last hidden state from both directions for the final output
        return out


# Load Data
train_dataset = datasets.MNIST(
    root="dataset/", train=True, transform=transforms.ToTensor(), download=True
)
test_dataset = datasets.MNIST(
    root="dataset/", train=False, transform=transforms.ToTensor(), download=True
)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialize network
# model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)
# model = GRU(input_size, hidden_size, num_layers, num_classes).to(device)
# model = LSTM(input_size, hidden_size, num_layers, num_classes).to(device)
model = BiLSTM(input_size, hidden_size, num_layers, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
        # Get data to cuda if possible
        data = data.to(device=device).squeeze(1)  # batch, everything else. see forward method in RNN
        targets = targets.to(device=device)

        # Forward
        scores = model(data)
        loss = criterion(scores, targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient descent or adam step
        optimizer.step()


# Check accuracy on training & test to see how good our model
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    # We don't need to keep track of gradients here, so we wrap it in torch.no_grad()
    with torch.no_grad():
        # Loop through the data
        for x, y in loader:
            # Move data to device
            x = x.to(device=device).squeeze(1)
            y = y.to(device=device)

            # Forward pass
            scores = model(x)
            _, predictions = scores.max(1)

            # Check how many we got correct
            num_correct += (predictions == y).sum()

            # Keep track of number of samples
            num_samples += predictions.size(0)

    model.train()
    return num_correct / num_samples


# Check accuracy on training & test to see how good our model
print(f"Accuracy on training set: {check_accuracy(train_loader, model) * 100:.2f}")
print(f"Accuracy on test set: {check_accuracy(test_loader, model) * 100:.2f}")
