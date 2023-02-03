from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from torch import nn, optim
from torch.utils.data import SubsetRandomSampler
from torchvision import datasets, models
import matplotlib.pyplot as plt
from torchvision.models import ResNet18_Weights

torch.manual_seed(17)


class Trainer:
    def __init__(self, transformation, train_dir, model_path, valid_ratio, batch_size, n_epochs, lr=0.001):
        self.model = None
        self.transform = transformation
        self.train_dir = train_dir
        self.model_path = model_path
        self.valid_ratio = valid_ratio
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.lr = lr
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.train_data = datasets.ImageFolder(self.train_dir, transform=self.transform)
        self.classes = self.train_data.classes

        self._load_pretrained_model()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.fc.parameters(), lr=self.lr, momentum=0.9)

    def train(self):
        valid_loss_min = np.Inf

        for epoch in range(1, self.n_epochs + 1):
            train_loss = 0.0
            valid_loss = 0.0

            self.model.train()
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()                         # Clear the gradient because it is saved at each step
                output = self.model(data)                          # Forward
                loss = self.criterion(output, target)              # Compute the loss
                loss.backward()                                    # Compute the gradient
                self.optimizer.step()                              # Perform updates using calculated gradients
                train_loss += loss.item() * data.size(0)

            self.model.eval()
            for data, target in valid_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                valid_loss += loss.item() * data.size(0)

            # calculate average losses
            train_loss = train_loss / len(train_loader)
            valid_loss = valid_loss / len(valid_loader)
            print(f"Epoch: {epoch} \tTraining Loss: {train_loss} \tValidation Loss: {valid_loss}")

            if valid_loss <= valid_loss_min:
                print(f"Validation loss decreased ({valid_loss_min} --> {valid_loss}).  Saving model ...")
                torch.save(self.model.state_dict(), str(self.model_path))
                valid_loss_min = valid_loss

    def _load_pretrained_model(self):
        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        print(f"Original version version\ninput features: {self.model.fc.in_features}")
        print(f"output features: {self.model.fc.out_features}")

        for param in self.model.parameters():
            param.requires_grad = False

        n_inputs = self.model.fc.in_features

        # new layers automatically have requires_grad = True
        last_layer = nn.Linear(n_inputs, len(self.classes))
        self.model.fc = last_layer

        print(f"\nAdapted version for {len(self.classes)} classes \ninput features: {self.model.fc.in_features}")
        print(f"output features: {self.model.fc.out_features}")
        self.model.to(self.device)

    def visualise_data_loader(self, data_loader, nb_images_to_display, figsize=(25, 4)):
        def imshow(img):
            """helper function to un-normalize and display an image"""
            img = img / 2 + 0.5
            plt.imshow(np.transpose(img, (1, 2, 0)))

        # obtain one batch of training images
        images, labels = next(iter(data_loader))
        images = images.numpy()

        # plot the images in the batch, along with the corresponding labels
        fig = plt.figure(figsize=figsize)
        for idx in np.arange(nb_images_to_display):
            ax = fig.add_subplot(2, nb_images_to_display // 2, idx + 1, xticks=[], yticks=[])
            imshow(images[idx])
            ax.set_title(self.classes[labels[idx]])

        plt.show()

    def load_data(self):
        train_sampler, valid_sampler = self._train_valid_split()
        train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, sampler=train_sampler)
        valid_loader = torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, sampler=valid_sampler)
        return train_loader, valid_loader

    def _train_valid_split(self):
        """ Function that split the dataset into train and validation
            given in parameter the training set and the % of sample for validation"""

        # obtain training indices that will be used for validation
        num_train = len(self.train_data)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        split = int(np.floor(self.valid_ratio * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        return train_sampler, valid_sampler


if __name__ == "__main__":
    ROOT_DIR = Path("/home/medhyvinceslas/Documents/programming/datasets")
    TRAIN_DIR = ROOT_DIR / "plant_disease_dataset/Train/Train"
    MODEL_PATH = Path("/home/medhyvinceslas/Documents/programming/helpers/classification/weights/model.pt")

    assert TRAIN_DIR.is_dir()
    assert MODEL_PATH.parent.is_dir()

    VALID_RATIO = .25
    BATCH_SIZE = 4
    LR = 0.001
    N_EPOCHS = 1

    transform = T.Compose([T.RandomRotation(30), T.RandomHorizontalFlip(),
                           T.Resize(255), T.CenterCrop(224),
                           T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    trainer = Trainer(transform, TRAIN_DIR, MODEL_PATH, VALID_RATIO, BATCH_SIZE, N_EPOCHS,LR)
    train_loader, valid_loader = trainer.load_data()
    trainer.visualise_data_loader(train_loader, 4)
    # trainer.train()
