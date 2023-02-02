from pathlib import Path

import cv2
import torch
import torchvision
import numpy as np
from torchvision.models import MobileNet_V2_Weights
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN

from utils.opencv_helpers import OpencvHelper
from utils.gpu import cuda_setup
from utils.utils import timer


def normalize(image):
    image = image / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    return image


def to_tensor(image):
    image = image.transpose((2, 0, 1))
    image = torch.from_numpy(image).type(torch.FloatTensor)
    image = torch.unsqueeze(image, 0)
    return image


@timer
def predict(model, image, device):
    with torch.no_grad():
        outputs = model(image.to(device))
    return outputs


if __name__ == "__main__":
    # Load model
    model_path = Path("/home/medhyvinceslas/Documents/programming/helpers/detection/weights/detect_defect.pt")
    assert model_path.is_file()

    num_classes = 11
    _, device = cuda_setup()
    ocv_helper = OpencvHelper()

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

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Predict
    dataset_path = Path("/home/medhyvinceslas/Documents/programming/DL/datasets/defect_detection")
    image_folder = dataset_path / "images/images"
    img_path = image_folder / "inclusion/img_02_425502300_00338_resized.jpg"

    img = cv2.imread(str(img_path))
    orig_image = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = normalize(img)
    img = to_tensor(img)

    outputs = predict(model, img, device)
    print(outputs)

    # ocv_helper.imshow(orig_image, "out")
