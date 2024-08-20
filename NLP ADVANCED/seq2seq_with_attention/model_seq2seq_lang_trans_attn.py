import torch
import torch.nn as nn
import torch.optim as optim
import spacy
import random
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

from torch.utils.tensorboard import SummaryWriter
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint

# some details from here: https://github.com/bentrevett/pytorch-seq2seq/tree/main
spacy_ger = spacy.load("de_core_news_sm")
spacy_eng = spacy.load("en_core_web_sm")


# create a tokenizer function for both languages
def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]


def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


# data/text preprocessing
german = Field(tokenize=tokenize_ger, lower=True, init_token="<sos>", eos_token="<eos>")
english = Field(tokenize=tokenize_eng, lower=True, init_token="<sos>", eos_token="<eos>")

# split our text corpus into train, val, and test - check default format for these splits
train_data, val_data, test_data = Multi30k.splits(exts=(".de", ".en"), fields=(german, english))

# word embeddings
german.build_vocab(train_data, max_size=10000, min_freq=2)  # min freq of word to be included in our vocab
english.build_vocab(train_data, max_size=10000, min_freq=2)


# Define Network


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):  # input_size --> size of german vocab
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p)  # here not using since num_layers = 1

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, bidirectional=True)

        # For Attention Part
        # since now (hidden, cell) output extra dim as the network is bidirectional and to decide whether to send in
        # forward or backward hidden/cell state to the decoder for attn this is being done through linear layer
        # which sort of allow network choose itself the most relevant i.e., forward or backward part.
        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)  # hidden_size*2 --> forward, backward
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x):  # x is inp vector of indices
        # x shape: (seq_len, N) where N is batch size
        embedding = self.dropout(self.embedding(x))  # embedding shape : (seq_len, N, embedding_size)
        encoder_states, (hidden, cell) = self.rnn(embedding)  # out shape : (seq_len, N, hidden_size)
        # and here context vector
        # --> (hidden, cell) is imp and needs to be returned. see fig.

        # For Attention Part
        # hidden shape: (2, N, hidden_size)
        hidden = self.fc_hidden(torch.cat((hidden[0:1], hidden[1:2]), dim=2))  # forward & backward hidden state
        cell = self.fc_cell(torch.cat((cell[0:1], cell[1:2]), dim=2))  # forward & backward hidden state
        # (eq.5 in paper h_j).

        return encoder_states, hidden, cell  # imp to return encoder_state because we want hidden state for
        # each timestep and this hidden is only outputting the rightmost hidden one, and
        # it doesn't have a hidden state for each timestep that what's encoder_states have 


class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, p):  # input_size -->
        # size of the English vocab and output_size is going to be same as input size, we're outputting some sort of
        # vector with some probability corresponding to values for each word which is in our vocabulary.
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p)

        self.embedding = nn.Embedding(input_size, embedding_size)
        # For Attention Part
        self.rnn = nn.LSTM(hidden_size * 2 + embedding_size, hidden_size, num_layers)  # because we've the
        # encoder states which has a forward & backward one & both of them are included in the first hidden state
        # (meaning context vector from encoder) and embedding_size is normal embeddings.
        self.energy = nn.Linear(hidden_size * 3, 1)  # now design a small neural network to learn attn
        # hidden_size*3 because first we want to add the hidden states from the encoder (fwd, bwd) and 
        # then also take the hidden state from previous step in our decoder. see eq.6, e_ij
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, encoder_states, hidden, cell):  # x --> take one word at a time. see diag
        # x shape: (N) where N is batch size, we want it to be (1, N), seq_length
        # is 1 here because we are sending in a single word and not a sentence. word by word. so,
        x = x.unsqueeze(0)
        embedding = self.dropout(self.embedding(x))  # shape: (1, N, embedding_size)

        # For Attention Part
        sequence_length = encoder_states.shape[0]
        h_reshaped = hidden.repeat(sequence_length, 1, 1)  # we've encoder_states for all of our hidden time steps
        # but to add them together we need to have this the same dim along that axis, so we need to repeat
        # that hidden states from our decoder sequence_length times to be able to concatenate them together
        # (weighted sum).
        energy = self.relu(self.energy(torch.cat((h_reshaped, encoder_states), dim=2)))  # hidden_size*3 -> (1, 2)
        attention = self.softmax(energy)  # shape: (seq_length, N,  1)
        # now to get context vector c_i, we'll element wise multiply attention with the encoder_states
        attention = attention.permute(1, 2, 0)  # shape: (N,  1, seq_length)
        encoder_states = encoder_states.permute(1, 0, 2)  # shape: (N, seq_length, hidden_size*2)
        context_vector = torch.bmm(attention, encoder_states).permute(1, 0, 2)  # shape: (N, 1, hidden_size*2) (Eq.5)

        rnn_input = torch.cat((context_vector, embedding), dim=2)
        outputs, (hidden, cell) = self.rnn(rnn_input)  # shape: (1, N, hidden_size)
        prediction = self.fc(outputs)
        # prediction_shape: (1, N, length_target_vocabulary) to send it to loss fn.
        # we want it to be (N, len_target_vocab) so remove the first dim
        prediction = prediction.sequeeze(0)

        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):  # send in source --> the German sentence
        # and target sentence / correct translation --> English sentence
        # & teacher_force_ratio --> since prediction is not always correct so 50% of the time we send predicted
        # word from the previous cell
        # and 50% of time we send target ones. it can't be 1 (mean always use target) otherwise at test time the words
        # it predicts might be completely different from it might see at a training time.
        batch_size = source.shape[1]  # second dim
        target_len = target.shape[0]  # first dim
        target_vocab_size = len(english.vocab)

        outputs = torch.zeros(target_len, batch_size, target_vocab_size)
        encoder_states, hidden, cell = self.encoder(source)  # for attn, return encoder_states and send to decoder
        # at every time step and at the first step we'll use hidden, cell then it will be overwritten
        # after predicting the first word ...

        # Grab the first input to the Decoder which will be <SOS> token and send in to decoder word by word (loop)
        x = target[0]

        for t in range(1, target_len):
            # Use previous hidden, cell as context from encoder as start
            output, hidden, cell = self.decoder(x, encoder_states, hidden, cell)
            # store next output prediction
            outputs[t] = output  # previously predicted words (paper)
            # get the best word the decoder predicted (index in the vocab)
            best_guess = output.argmax(1)  # out shape: (N, english_vocab_size)
            # With probability of teacher_force_ratio we take the actual next word
            # otherwise we take the word that the Decoder predicted it to be.
            # Teacher Forcing is used so that the model gets used to seeing
            # similar inputs at training and testing time, if teacher forcing is 1
            # then inputs at test time might be completely different from what the
            # network is used to. This was a long comment.
            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs


"""Training part - Seq2Seq Model"""

# Hyperparameters for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inp_size_encoder = len(german.vocab)
inp_size_decoder = len(english.vocab)
output_size = len(english.vocab)
encoder_embedding_size = 300
decoder_embedding_size = 300
hidden_size = 1024  # Needs to be same for both RNNs
num_layers = 2
encoder_dropout = 0.5
decoder_dropout = 0.5
num_epochs = 2
lr_rate = 0.001
batch_size = 64
load_model = False

# Plot loss using tensorboard
writer = SummaryWriter(f"runs/plot_loss")
step = 0

# start the batching process to build an iterator for our training, validation and testing data split.
train_iter, val_iter, test_iter = BucketIterator.split(
    (train_data, val_data, test_data), batch_size=batch_size, sort_within_batch=True, sort_key=lambda x: len(x.src),
    device=device)

# load Net, optimizer, loss
encoder_net = Encoder(inp_size_encoder, encoder_embedding_size, hidden_size, num_layers, encoder_dropout).to(device)
decoder_net = Decoder(inp_size_decoder, decoder_embedding_size, hidden_size, num_layers, decoder_dropout).to(device)

model = Seq2Seq(encoder_net, decoder_net).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr_rate)

pad_idx = english.vocab.stoi["<PAD>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)  # we've padded so that all the examples in the batch are
# similar length and we don't want to pay anything for that in our loss function.
if load_model:
    load_checkpoint("my_checkpoint.pth.tar", model, optimizer)

sentence = "ein boot mit mehreren männern darauf wird von einem großen pferdegespann ans ufer gezogen."

for epoch in range(num_epochs):
    print(f"[Epoch {epoch} / {num_epochs}")

    checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
    save_checkpoint(checkpoint)

    model.eval()

    translated_sentence = translate_sentence(model, sentence, german, english, device, max_length=500)
    print(f"Translated example sentence: \n {translated_sentence}")

    model.train()

    for batch_idx, batch in enumerate(train_iter):
        # Get input and targets and get to cuda
        inp_data = batch.src.to(device)
        target = batch.trg.to(device)

        # Forward Prop
        output = model(inp_data, target)
        # Output is of shape (target_len, batch_size, output_dim) but Cross Entropy Loss
        # doesn't take input in that form. For example if we have MNIST we want to have
        # output to be: (N, 10) and targets just (N). Here we can view it in a similar
        # way that we have output_words * batch_size that we want to send in into
        # our cost function, so we need to do some reshaping. While we're at it
        # Let's also remove the start token output[1:] while we're at it
        output = output[1:].reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)  # (target_len, batch_size)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()  # Back Prop

        # Clip to avoid exploding gradient issues, makes sure grads are
        # within a healthy range
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        # Gradient decent step
        optimizer.step()

        # Plot to tensorboard
        writer.add_scalar("Training loss", loss, global_step=step)
        step += 1

score = bleu(test_data[1:100], model, german, english, device)
print(f"Bleu score {score * 100:.2f}")
