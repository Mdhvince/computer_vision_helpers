from pathlib import Path

import albumentations as A
import torch
import torchvision
import numpy as np
from torchvision.models import MobileNet_V2_Weights
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN

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

def save_model(model, model_path, optimizer, epoch, loss):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, str(model_path))


if __name__ == "__main__":
    dataset_path = Path("/home/medhyvinceslas/Documents/programming/DL/datasets/defect_detection")
    label_folder = dataset_path / "label/label"
    image_folder = dataset_path / "images/images"
    model_path = Path("/home/medhyvinceslas/Documents/programming/helpers/detection/weights/detect_defect.pt")

    assert label_folder.is_dir()
    assert image_folder.is_dir()
    assert model_path.parent.is_dir()

    batch_size = 8
    num_workers = 4
    valid_size = 0.25
    num_epochs = 1
    im_size = 256  # 800
    resume_training = False

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
    # changing the backbone
    backbone = torchvision.models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1).features
    # freezing the parameters of the backbone
    for param in backbone.parameters():
        param.requires_grad = False

    backbone.out_channels = 1280
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
    model = FasterRCNN(backbone, num_classes=num_classes, rpn_anchor_generator=anchor_generator)

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
        total_train_loss = 0.0
        total_valid_loss = 0.0

        for images, targets in train_loader:
            optimizer.zero_grad()

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_train_loss += losses.item()

            losses.backward()
            optimizer.step()

        for images, targets in valid_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with torch.no_grad():
                loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_valid_loss += losses.item()

        # loss_dict = {
        #   'loss_classifier': tensor(2.5018, grad_fn=<NllLossBackward0>),
        #   'loss_box_reg': tensor(0.0273, grad_fn=<DivBackward0>),
        #   'loss_objectness': tensor(0.6984, grad_fn=<BinaryCrossEntropyWithLogitsBackward0>),
        #   'loss_rpn_box_reg': tensor(0.0445, grad_fn=<DivBackward0>)
        # }

        # calculate average losses
        total_train_loss = total_train_loss / len(train_loader)
        total_valid_loss = total_valid_loss / len(valid_loader)

        print(f"Epoch: {epoch} \tTraining Loss: {total_train_loss} \tValidation Loss: {total_valid_loss}")

        if total_valid_loss <= valid_loss_min:
            save_model(model, model_path, optimizer, epoch, total_valid_loss)
            valid_loss_min = total_valid_loss
