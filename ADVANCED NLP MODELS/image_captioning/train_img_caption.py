import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from utils import save_checkpoint, load_checkpoint, print_examples
from load_dataset_img_caption import get_loader
from image_caption_model import CNNtoRNN
from tqdm import tqdm


def train():
    transform = transforms.Compose(
        [
            transforms.Resize((356, 356)),
            transforms.RandomCrop((299, 299)),  # since inception model take input 299x299
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_loader, dataset = get_loader(root_folder="../flickr8k/images", annotation_file="../flickr8k/captions.txt",
                                       transform=transform, num_workers=2)

    torch.backends.cudnn.benchmark = True  # Performance boost
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_model = False
    save_model = True

    # Hyperparameters
    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab)
    num_layers = 1
    learning_rate = 3e-4
    num_epochs = 2

    # for tensorboard
    writer = SummaryWriter("runs/flicker")
    step = 0

    # initialize model, loss and optim etc.
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if load_model:
        step = load_checkpoint(torch.load("my_checkpoints.pth.tar"), model, optimizer)

    model.train()

    for epoch in range(num_epochs):
        # Uncomment the line below to see a couple of test cases
        # print_examples(model, device, dataset)

        if save_model:
            checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step}
            save_checkpoint(checkpoint)

        for idx, (images, captions) in tqdm(enumerate(train_loader), total=len(train_loader), leave=False):
            images = images.to(device)
            captions = captions.to(device)

            outputs = model(images, captions[:-1])  # last caption is not included which will be used for prediction.
            # i.e., model will predict the end token
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))  # Here we're predicting for
            # each example for a bunch of different timestep. one example might have 20 words it's predicting
            # then each word has its logit corresponding to each word in the vocab, so we've 3 dim here
            # but the criterion accept 2 dim. (seq_len, N, vocab_size), target --> (seq_len, N)
            writer.add_scalar("Training Loss", loss.item(), global_step=step)
            step += 1

            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()


if __name__ == "__main__":
    train()
