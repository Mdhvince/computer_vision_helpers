from pathlib import Path

import albumentations as A
import torch
import torchvision
import numpy as np
# noinspection PyProtectedMember
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from custom_dataset import CustomDataset, build_loaders, get_data
from utils.gpu import cuda_setup


def save_model(model, model_path, epoch, optimizer):
    torch.save({
        'epoch': epoch+1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, str(model_path))


if __name__ == "__main__":
    dataset_path = Path("/home/medhyvinceslas/Documents/programming/DL/datasets/defect_detection")
    label_folder = dataset_path / "label/label"
    image_folder = dataset_path / "images/images"
    model_path = Path("/home/medhyvinceslas/Documents/programming/helpers/base_detection/weights/detect_defect.pt")
    batch_size = 8
    num_workers = 4
    valid_size = 0.25
    _, device = cuda_setup()

    transforms = [
        A.HorizontalFlip(p=.5),
        A.VerticalFlip(p=.5)
    ]

    data, num_classes = get_data(label_folder, image_folder)
    dataset = CustomDataset(data, transforms, im_size=800)
    train_loader, valid_loader = build_loaders(dataset, batch_size, valid_size, num_workers)

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    num_epochs = 2
    valid_loss_min = np.Inf

    for epoch in range(num_epochs):
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

        print(f"Epoch: {epoch+1} \tTraining Loss: {train_loss} \tValidation Loss: {valid_loss}")

        if valid_loss <= valid_loss_min:
            save_model(model, model_path, epoch, optimizer)



