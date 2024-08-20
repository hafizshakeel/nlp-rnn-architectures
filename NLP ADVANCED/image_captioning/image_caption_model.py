"""
Natural Language Processing Model Implementation

This script contains the implementation of the image captioning model. Please refer to the architecture diagram.

Implementation by: Hafiz Shakeel Ahmad Awan
Email: hafizshakeel1997@gmail.com
"""

import torch
import torch.nn as nn
import torchvision.models as models


# https://koushik0901.medium.com/image-captioning-with-deep-learning-f2f2059d9e3c
class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(EncoderCNN, self).__init__()
        self.train_CNN = train_CNN
        self.inception = models.inception_v3(pretrained=True, aux_logits=True)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)  # The last linear layer
        # of the pre-trained model is replaced with our own linear layer that gives
        # an output same as the dimension of the embeddings.
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        features = self.inception(images)
        features = features.logits  # Extract the final output logits

        # Only finetune the CNN
        for name, param in self.inception.named_parameters():
            if "fc.weight" in name or "fc.bias" in name:
                param.requires_grad = True

            else:
                param.requires_grad = self.train_CNN

        return self.dropout(self.relu(features))


# Next, the output from our Encoder is passed in as the hidden state to our Decoder.
# The first time step receives the <START> token. see Image Captioning.png
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)  # word embeddings --> map the word to some
        # dimensional space to have a better representation of the words. see word2vec implementation
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)  # num_layers --> how many LSTMs we want
        # to stack on top of each other
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions))
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)  # here we need to add
        # an additional dim so that it's viewed as the timestep and then concatenate with embeddings
        # that already have a timestep for the target caption it's a some num of words,
        # and we're just adding the features sort of viewing it as the first word that we input to the LSTM.
        hidden, _ = self.lstm(embeddings)
        outputs = self.linear(hidden)
        return outputs


class CNNtoRNN(nn.Module):  # Seq2Seq class
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CNNtoRNN, self).__init__()
        self.encoderCNN = EncoderCNN(embed_size)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoderCNN(images)
        outputs = self.decoderRNN(features, captions)
        return outputs

    # When we're doing the LSTM we're using the target caption for every timestep it's making some prediction
    # which are going to be used in our loss function but actual input upto that hidden state that we're predicting
    # the target captions will be from the dataset that sort of the distinctions where we've one way when
    # training the model and another way when we're actually evaluating or doing inference on an image
    # that we don't have a target caption for.

    def caption_image(self, image, vocabulary, max_length=50):
        # take the predicted word and use that for the future words
        result_captions = []
        with torch.no_grad():
            x = self.encoderCNN(image).unsqueeze(0)  # dim for a batch
            states = None  # hidden and cell states for LSTM

            for _ in range(max_length):
                hiddens, states = self.decoderRNN.lstm(x, states)  # Initially the states are None
                output = self.decoderRNN.linear(hiddens.squeeze(0))
                predicted = output.argmax(1)  # words with the highest probability
                result_captions.append(predicted.item())
                x = self.decoderRNN.embed(predicted).unsqueeze(0)  # Next input

                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break
        return [vocabulary.itos[idx] for idx in result_captions]  # obtaining the str that corresponds to the pred.


# During training, we use Teacher Forcing i.e. we give in the correct input to the decoder
# even if it has made mistakes on the previous timestep.

# Teacher forcing is a strategy for training recurrent neural networks that uses ground truth as input,
# instead of model output from a prior time step as an input.
