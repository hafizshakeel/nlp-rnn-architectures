import os  # when loading file paths
import pandas as pd  # for lookup in annotation file
import spacy  # for tokenizer
import torch
from torch.nn.utils.rnn import pad_sequence  # pad batch
from torch.utils.data import DataLoader, Dataset
from PIL import Image  # Load img
import torchvision.transforms as transforms

# Read this blog post to understand torchtext: https://anie.me/On-Torchtext/

# We want to convert text --> numerical values
# 1. We need a vocabulary mapping each word to an index
# 2. We need to setup a PyTorch dataset to load the data
# 3. Setup padding of every batch i.e., all examples should be of same sequence len and setup dataloader


# Download with: python -m spacy download en
spacy_eng = spacy.load("en_core_web_sm")


class Vocabulary:
    def __init__(self, freq_threshold):  # freq_threshold --> something like repetition of word in our vocab
        # so we're saying if it's not repeated freq_threshold amount of time in our vocab we're going to ignore it.
        # i.e, this word is not important as it occurred rarely.
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}  # PAD --> padding, SOS --> start of sentence,
        # EOS --> end of sentence, UNK --> unknown if a word is not repeated over then freq_threshold
        # then it's going to map with this unknown token.
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):  # get len of our vocab
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):  # no initialization of self since static
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]  # "I love peanuts" --> ["i", "love", "peanut"]

    def build_vocabulary(self, sentence_list):
        frequencies = {}  # include if over the thresh_frequency
        idx = 4  # start with index 4 because 0-3 are already defined --> see init method of Vocabulary class

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1

                else:
                    frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx  # starting at index 4
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):  # covert text to numerical values
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]


class FlickerDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=5):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)
        self.transform = transform

        # Get img, caption columns --> captions.txt
        self.images = self.df["image"]
        self.captions = self.df["caption"]

        # Initialize vocabulary and build vocab
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions.tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        caption = self.captions[index]  # index of a single caption
        img_id = self.images[index]  # index of a single image
        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        numericalized_caption = [self.vocab.stoi["<SOS>"]]  # start of sentence --> start token
        numericalized_caption += self.vocab.numericalize(caption)  # converting each word to an index in our vocab
        numericalized_caption.append(self.vocab.stoi["<EOS>"])  # end of sentence --> end token

        return img, torch.tensor(numericalized_caption)


# Pad so we've equal amount of images and captions in our batch.
class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        images = [item[0].unsqueeze(0) for item in batch]  # images are at zero index and add extra dim for batch
        images = torch.cat(images, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)

        return images, targets


def get_loader(root_folder, annotation_file, transform, batch_size=16, num_workers=8, shuffle=True, pin_memory=True):
    dataset = FlickerDataset(root_folder, annotation_file, transform=transform)
    pad_idx = dataset.vocab.stoi["<PAD>"]
    loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle,
                        pin_memory=pin_memory, collate_fn=MyCollate(pad_idx=pad_idx))

    return loader, dataset


if __name__ == "__main__":
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )
    loader, dataset = get_loader("../flickr8k/images/", "../flickr8k/captions.txt", transform=transform)

    for idx, (imgs, captions) in enumerate(loader):
        print(imgs.shape)
        print(captions.shape)
        break  # Exit the loop after the first batch

# torch.Size([32, 3, 224, 224])
# torch.Size([31, 32])

# Here, the printed shapes indicate that the batch contains 32 images of size 224x224 with 3 color channels,
# and 32 corresponding captions, each with a maximum length of 31 tokens for this batch.
