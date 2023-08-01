import argparse
import logging
import os

import numpy as np
import torch
import torchvision.transforms as T
from torch import nn, optim
from torch.utils.data import SubsetRandomSampler
from torchvision import datasets, models
from torchvision.models import ResNet18_Weights

torch.manual_seed(17)


class Trainer:
    def __init__(self, train_dir, model_path, valid_ratio, batch_size, n_epochs, lr=0.001, momentum=0.9):
        self.model = None
        self.train_dir = train_dir
        self.model_path = model_path
        self.valid_ratio = valid_ratio
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_data = datasets.ImageFolder(self.train_dir, transform=self.transform)
        self.classes = self.train_data.classes

        self._load_pretrained_model()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.fc.parameters(), lr=self.lr, momentum=momentum)

        self.transform = T.Compose([
            T.RandomRotation(30), T.RandomHorizontalFlip(),
            T.Resize(255), T.CenterCrop(224),
            T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

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


def parser_fn():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_dir",
        type=str,
        help="Path to the training directory. On GCP, use /gcs/bucket/my_data_dir"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to where to save the model (.pt). On GCP, use /gcs/bucket/model-output/my_model.pt"
    )
    parser.add_argument(
        "--log_path",
        type=str,
        help="Path to the log file. On GCP, use /gcs/bucket/log/my_log.log"
    )
    parser.add_argument("--valid_ratio", type=float, help="Ratio of the validation set", default=0.25)
    parser.add_argument("--batch_size", type=int, help="Batch size", default=4)
    parser.add_argument("--n_epochs", type=int, help="Number of epochs", default=1)
    parser.add_argument("--lr", type=float, help="Learning rate", default=0.001)
    parser.add_argument("--momentum", type=float, help="Momentum", default=0.9)
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = parser_fn()

    logging.basicConfig(filename=args.log_path, level=logging.INFO)
    logging.info("Running pytorch version: {}".format(torch.__version__))
    logging.info("Starting training.")

    trainer = Trainer(
        args.train_dir,
        args.model_path,
        args.valid_ratio,
        args.batch_size,
        args.n_epochs,
        args.lr,
        args.momentum
    )
    train_loader, valid_loader = trainer.load_data()
    trainer.train()

    logging.info("Training complete.")
