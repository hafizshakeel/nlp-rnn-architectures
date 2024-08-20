"""
Neural Network Models Implementation for Natural Language Processing

This script provides implementations of several types of neural networks:
1. Recurrent Neural Network (RNN)
2. Gated Recurrent Unit (GRU)
3. Long Short-Term Memory (LSTM)
4. Bidirectional LSTM (BiLSTM)

Implementation by: Hafiz Shakeel Ahmad Awan
Email: hafizshakeel1997@gmail.com

Each model is designed to work with sequences of length 28 and input features of size 28,
processing sequences through different types of recurrent neural networks to classify input data.

The models include:
- RNN: A basic RNN that processes sequential data with a single hidden layer.
- GRU: A GRU network that leverages gating mechanisms to handle sequences.
- LSTM: An LSTM network that uses memory cells to retain information over longer sequences.
- BiLSTM: A Bidirectional LSTM that processes sequences in both forward and backward directions.

The script also includes code to train and evaluate the models using the MNIST dataset.
"""


# Imports
import torch
import torch.nn.functional as F  # Parameterless functions, like (some) activation functions
import torchvision.datasets as datasets  # Standard datasets
import torchvision.transforms as transforms  # Transformations we can perform on our dataset for augmentation
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn  # All neural network modules
from torch.utils.data import DataLoader  # Gives easier dataset management by creating mini batches etc.
from tqdm import tqdm  # For nice progress bar!

# Set device cuda for GPU if it's available otherwise run on the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
# input Nx1x28x28, which can be viewed as 28 time sequences and each sequence contains 28 features so
input_size = 28
sequence_length = 28  # taking one row at a time and sending into the RNN at each timestep
hidden_size = 256
num_layers = 2
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 2

""" Create a simple / basic RNN """


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)  # input-size --> num of features
        # for each timestep and we don't have to explicitly say how many sequences we want to have,
        # the RNN will work for any num of sequences that we send in. just in this case it will be 28 sequences and
        # hidden_size --> num of nodes in the hidden at each time step
        # num_layers --> num layers for RNN
        # batch_first=True --> N x sequence x features

        self.fc = nn.Linear(hidden_size * sequence_length, num_classes)  # since we've 28 time sequences and
        # we'll concatenate all of those sequences and send into the linear layer, so it will use information
        # from evey hidden state; you could also take just the absolute last hidden state

        # Note: to use only the last hidden state which contains info from all the previous hidden state
        # we don't need concatenation. so we can define fc as
        self.fc_lh = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # hidden state --> num_layers, batch, nodes
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # Forward Prop.
        out, _ = self.rnn(x, h0)  # here we're not storing hidden state as every example has its own hidden state
        # so ignore that output
        # out = out.reshape(out.shape[0], -1)  # keep the batch and concatenate everything else
        out_lh = out[:, -1, :]  # all examples, last hidden state, all features
        # out = self.fc(out)
        out = self.fc_lh(out_lh)
        return out


""" Create a simple / basic RGU """


# Train for all hidden states
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * sequence_length, num_classes)

    def forward(self, x):
        # hidden state --> num_layers, batch, nodes
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # Forward Prop.
        out, _ = self.gru(x, h0)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out


""" Create a simple / basic LSTM """


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)  # train for only the last hidden state

    def forward(self, x):
        # hidden state h0, cell state c0 --> num_layers, batch, nodes
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # Forward Prop.
        out, _ = self.lstm(x, (h0, c0))  # second arg --> send h0 and c0 as a tuple
        out = self.fc(out[:, -1, :])  # all examples, last hidden state, all features
        return out


""" Create a simple / basic Bidirectional LSTM
Not suitable for all types of data. Also slow training!"""



class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bi_lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # train for only the last hidden state and hidden_size are
        # multiplies by 2 because we've one forward LSTM and one backward LSTM

    def forward(self, x):
        # hidden state h0, cell state c0 --> num_layers, batch, nodes
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)  # here num_layers are
        # multiplies by 2 because we've one forward LSTM and one backward LSTM and same for cell state
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        # Forward Prop.
        out, _ = self.bi_lstm(x, (h0, c0))  # second arg --> send h0 and c0 as a tuple
        out = self.fc(out[:, -1, :])  # all examples, last hidden state, all features
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
