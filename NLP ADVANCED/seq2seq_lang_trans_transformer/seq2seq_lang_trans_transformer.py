import torch
import torch.nn as nn
import torch.optim as optim
import spacy
from utils import *
from torch.utils.tensorboard import SummaryWriter
from torchtext.datasets import Multi30k  # Ger to Eng.
from torchtext.data import Field, BucketIterator

""" Seq2Seq using Transformers --> Pytorch inbuilt Transformer modules """

# Load Vocabulary (German & English)
spacy_ger = spacy.load("de_core_news_sm")
spacy_eng = spacy.load("en_core_web_sm")


# create a tokenizer function for German and English Language
def tokenizer_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]


def tokenizer_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


# data/text preprocessing
german = Field(tokenize=tokenizer_ger, lower=True, init_token="<sos>", eos_token="<eos>")
english = Field(tokenize=tokenizer_eng, lower=True, init_token="<sos>", eos_token="<eos>")

# split our text corpus into train, val, and test - check default format for these splits
train_data, val_data, test_data = Multi30k.splits(exts=(".de", ".en"), feilds=(german, english))

# word embeddings
german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)


# Define builtin transformer network
class Transformer(nn.Module):
    def __init__(self, embed_size, src_vocab_size, trg_vocab_size, src_pad_idx, num_heads, num_enc_layers,
                 num_dec_layers, fwd_expansion, dropout, max_length, device):
        super(Transformer, self).__init__()
        self.src_word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.src_position_embedding = nn.Embedding(max_length, embed_size)  # since transformers are
        # permutation invariant so we add positional embedding
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.trg_position_embedding = nn.Embedding(max_length, embed_size)

        self.device = device
        self.tranformer = nn.Transformer(embed_size, num_heads, num_enc_layers, num_dec_layers, fwd_expansion, dropout)
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx

    def make_src_mask(self, src):
        src_mask = src.transpose(0, 1) == self.src_pad_idx  # transpose because src shape is (src_len, N) and
        # output is (N, src_len) which PyTorch transformer wants to be
        return src_mask

    def forward(self, src, trg):
        src_seq_length, N = src.shape
        trg_seq_length, N = trg.shape

        # create positions for position embedding and expand so that we've it for every example we send in
        src_positions = torch.arange(0, src_seq_length).unsqueeze(1).expand(src_seq_length, N).to(self.device)
        trg_positions = torch.arange(0, trg_seq_length).unsqueeze(1).expand(trg_seq_length, N).to(self.device)

        # word embedding and position embedding to make network aware of order of words
        embed_src = self.dropout(self.src_word_embedding(src) + self.src_position_embedding(src_positions))
        embed_trg = self.dropout(self.trg_word_embedding(trg) + self.trg_position_embedding(trg_positions))

        src_padding_mask = self.make_src_mask(src)  # create masking
        trg_mask = self.tranformer.generate_square_subsequent_mask(trg_seq_length).to(self.device)  # trg mask built-in

        out = self.tranformer(embed_src, embed_trg, src_key_padding_mask=src_padding_mask, trg_mask=trg_mask)
        out = self.fc_out(out)

        return out


""" Training and Model Hyperparameters """

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
src_vocab_size = len(german.vocab)
trg_vocab_size = len(english.vocab)
embed_size = 512
num_heads = 8
num_enc_layers = 3
num_dec_layers = 3
dropout = 0.10  # usually lower for seq2seq model
max_len = 100  # max sentence length for src and trg. Use for positional embedding
fwd_expansion = 4  # number of nodes - needs to check default nodes in transformer
src_pad_idx = english.vocab.stoi["<pad>"]

load_model = True
save_model = True
num_epochs = 5
learning_rate = 3e-4
batch_size = 32

# Tensorboard to get nice loss plot
writer = SummaryWriter("runs/loss_plot")
step = 0

train_iterator, val_iterator, test_iterator = BucketIterator.splits(
    (train_data, val_data, test_data),
    batch_size=batch_size,
    sort_within_batch=True,  # make sure each example in a batch of equal length
    sort_key=lambda x: len(x.src),  # sort by x.src, src sentence length
    device=device
)

model = Transformer(embed_size, src_vocab_size, trg_vocab_size, src_pad_idx, num_heads, num_enc_layers,
                    num_dec_layers, fwd_expansion, dropout, max_len, device).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
pad_idx = english.vocab.stoi("<pad>")
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

# load model and test
# score = bleu(test_data[1:100], model, german, english, device)
# print(f"Bleu score {score*100:.2f}")
# import sys
# sys.exit()

sentence = "ein pferd geht unter einer brücke neben einem boot."

for epoch in range(num_epochs):
    print(f"[Epoch {epoch}/{num_epochs}]")

    if save_model:
        checkpoint = {"stat_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
        save_checkpoint(checkpoint)

    model.eval()
    translate_sentence = translate_sentence(model, sentence, german, english, device, max_length=100)
    print(f"Translated example sentence \n {translate_sentence}")

    model.train()

    for batch_idx, batch in enumerate(train_iterator):
        inp_data = batch.src.to(device)
        target = batch.trg.to(device)

        # forward prop
        output = model(inp_data, target[:-1])  # when we send the first elem of input to be the start token
        # we want first output from the transformer to correspond to the second elem in the target

        # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
        # doesn't take input in that form. For example if we have MNIST we want to have
        # output to be: (N, 10) and targets just (N). Here we can view it in a similar
        # way that we have output_words * batch_size that we want to send in into
        # our cost function, so we need to do some reshaping.
        # Let's also remove the start token while we're at it
        # see seq2seq__lang_trans_with_attention_ptp.py for more info for below reshaping and other stuff

        output = output.reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)
        optimizer.zero_grad()

        loss = criterion(output, target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
        optimizer.step()

        writer.add_scalar("Training loss", loss, global_step=step)
        step += 1

score = bleu(test_data[1:100], model, german, english, device)
print(f"Bleu score {score * 100:.2f}")
