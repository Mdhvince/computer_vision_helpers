from pathlib import Path

import albumentations as A
import torch
import torchvision
import numpy as np
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from custom_dataset import CustomDataset, build_loaders, get_data
from utils.gpu import cuda_setup


def load_model(model, model_path, optimizer, device):
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    return model, optimizer, epoch, loss


if __name__ == "__main__":
    dataset_path = Path("/home/medhyvinceslas/Documents/programming/DL/datasets/defect_detection")
    label_folder = dataset_path / "label/label"
    image_folder = dataset_path / "images/images"
    model_path = Path("/home/medhyvinceslas/Documents/programming/helpers/base_detection/weights/detect_defect.pt")
    batch_size = 8
    num_workers = 8
    valid_size = 0.25
    num_epochs = 1
    im_size = 800
    resume_training = True

    _, device = cuda_setup()

    transforms = [
        A.HorizontalFlip(p=.5),
        A.VerticalFlip(p=.5)
    ]

    data, num_classes = get_data(label_folder, image_folder)
    dataset = CustomDataset(data, transforms, im_size=im_size)
    train_loader, valid_loader = build_loaders(dataset, batch_size, valid_size, num_workers)

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    if resume_training:
        model, optimizer, last_epoch, valid_loss_min = load_model(model, model_path, optimizer, device)
        model.train()
        print(f"Resume training: Last epoch={last_epoch}\tValidation loss={valid_loss_min}")
    else:
        valid_loss_min = np.Inf
        last_epoch = 0

    model.to(device)

    for epoch in range(1, num_epochs + 1):
        epoch = epoch + last_epoch
        train_loss = 0.0
        valid_loss = 0.0

        for images, targets in train_loader:
            optimizer.zero_grad()

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            train_loss += losses.item()
            losses.backward()
            optimizer.step()

        for images, targets in valid_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with torch.no_grad():
                loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            valid_loss += losses.item()

        # calculate average losses
        train_loss = train_loss / len(train_loader)
        valid_loss = valid_loss / len(valid_loader)

        print(f"Epoch: {epoch} \tTraining Loss: {train_loss} \tValidation Loss: {valid_loss}")

        if valid_loss <= valid_loss_min:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': valid_loss,
            }, str(model_path))

        valid_loss_min = valid_loss
