import torch
import torch.nn as nn

"""Attention is all you need --> Transformer from scratch"""


# Self Attention
class SelfAttn(nn.Module):
    def __init__(self, embed_size, heads):  # so we've embedding which we'll split into different parts, for instance,
        # 8 diff parts, and how many parts we split it is what we called heads so if we've embed_size 256 and heads 8,
        # then we're going to split it 8x32 parts
        super(SelfAttn, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads  # int div by heads

        # now, let's say we've embedding size 256, and we want to split it into 7 parts then that would not be possible
        # since we can't make an integer div of that, so we can throw out assert
        assert (self.head_dim * heads == embed_size), "Embed size needs to be div by heads"

        # Next, define the linear layers to send values, keys, and queries through
        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]  # Get the number of training examples
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]  # depending on where
        # we use the attention mechanism those lengths are going to be corresponding to the
        # source sentence length and the target sentence length. since we don't know exactly where this mechanism
        # is used either in the encoder or which part in the decoder those are going to vary, so we just use
        # it abstractly and say value_len, key_len, query_len, but really they will always correspond
        # to the source sentence length and the target sentence length

        values = self.values(values)  # (N, value_len, embed_size)
        keys = self.keys(keys)  # (N, key_len, embed_size)
        queries = self.queries(query)  # (N, query_len, embed_size)

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)  # self.heads, self.head_dim is where we're
        # splitting it since this was before a single dim of just embed size. similarly,
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        # now multiply queries and keys and
        # Einsum does matrix mult. for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just doing matrix multiplication & bmm

        energy = torch.einsum("nqhd, nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, heads_dim),
        # keys shape: (N, key_len, heads, heads_dim)
        # energy: (N, heads, query_len, key_len)

        # Mask padded indices so their weights become 0.
        if mask is not None:  # see masked_attention.png
            energy = energy.masked_fill(mask == 0, float("-1e20"))  # set to zero since we don't want to allow later
            # words to influence the earlier ones since otherwise they could kind of give away the answer
            # for what comes next, so we just force them to zero setting them initially to negative infinity
            # and then apply softmax to have sum of 1.

        # Normalize energy values similarly to seq2seq + attention so that they sum to 1.
        # Also divide by scaling factor for better stability
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)  # shape: (N, heads, query_len, key_len)
        # Eq.1 in paper

        out = torch.einsum("nhql, nlhd -> nqhd", [attention, values]).reshape(N, query_len, self.heads * self.head_dim)
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # out after matrix multiply: (N, query_len, heads, head_dim), then
        # we reshape and flatten the last two dimensions.

        out = self.fc_out(out)  # Linear layer doesn't modify the shape, final shape: (N, query_len, embed_size)
        return out


class TransformerBlock(nn.Module):  # Figure 1 of paper (left part)
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attn = SelfAttn(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)  # similar to BatchNorm--> takes avg across batch and the normalizes
        # whereas LayerNorm --> takes avg for every single example ( more computation)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),  # mapping it to some more nodes which is
            # dependent on the forward expansion
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attn = self.attn(value, key, query, mask)

        # Add skip connection / residual, run through normalization and finally dropout
        x = self.dropout(self.norm1(attn + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))

        return out


class Encoder(nn.Module):  # complete encoder with word embedding, position embedding,
    # and transformer block for a couple of layers
    def __init__(self, src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length):
        # max_length --> something related to positional embedding i.e., how long is the max sentence length
        # vary depending on dataset
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, heads, dropout=dropout, forward_expansion=forward_expansion)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)  # diff from paper
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))  # positional encoding

        for layers in self.layers:
            out = layers(out, out, out, mask)  # since we're in encoder so value, key, & query will be same

        return out


class DecoderBlock(nn.Module):  # Figure 1 of paper (right part)
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attn = SelfAttn(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(embed_size, heads, dropout, forward_expansion)  # same as TransBlock
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):  # trg_mask is essential and src_mask is optional
        # where if we send in a couple of examples then we need to pad it to make sure that all are equal length,
        # and then we  also send in a source mask so that we don't do unnecessary computations
        # for the ones that are padded
        attn = self.attn(x, x, x, trg_mask)  # masked multi-headed attention
        query = self.dropout(self.norm(attn + x))
        out = self.transformer_block(value, key, query, src_mask)  # since value and key from encoder block
        # and query from decoder block
        return out


class Decoder(nn.Module):  # complete decoder with word embedding, position embedding, decoder block etc.
    def __init__(self, trg_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList([
            DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
            for _ in range(num_layers)
        ])

        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)  # since enc_out, enc_out --> value, key from encoder

        out = self.fc_out(x)
        return out


class Transformer(nn.Module):  # put encoder and decoder together
    def __init__(self, src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, embed_size=256, num_layers=6,
                 fwd_expansion=4, heads=8, dropout=0, device="cpu", max_length=100):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, embed_size, num_layers, heads, device, fwd_expansion, dropout, max_length)
        self.decoder = Decoder(trg_vocab_size, embed_size, num_layers, heads, fwd_expansion, dropout, device, max_length)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    # now create function to make the source mask and the target mask
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)  # (N, 1, 1, src_len)
        return src_mask.to(self.device)  # if it is a src_pad_idx then it's going to be set to zero else set to 1

    def make_trg_mask(self, trg):  # see masked_attention.png
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones(trg_len, trg_len)).expand(N, 1, trg_len, trg_len)  # tril --> lower triangular
        # part and expand, so we have one for each training example
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out


# Test function
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)  # two examples
    # 1 is for start token, 0 is for padding, 2 is for <EOS>
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)  # diff shape and length
    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10  # since in examples 'x' we've 9 tokens
    trg_vocab_size = 10  # since in examples 'trg' we've 8 tokens
    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx).to(device)
    out = model(x, trg[:, :-1])  # trg will be shifted by one so that it doesn't have the <EOS> token
    # because we want to learn to predict <EOS>
    print(out.shape)  # [2, 7, 10]







