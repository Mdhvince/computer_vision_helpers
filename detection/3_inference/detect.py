from pathlib import Path

import cv2
import torch
import torchvision
import numpy as np
from torchvision.models import MobileNet_V2_Weights
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN

from utils.detection import DetectionUtils
from utils.opencv_helpers import OpencvHelper
from utils.gpu import cuda_setup
from utils.utils import timer


@timer
def load_model_inference(model_path, num_classes, device):
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
    return model

@timer
def normalize(image):
    image = image / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    return image

@timer
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


def resize_boxes(bbox, actual_size=(None, None), target_size=(None, None)):
    boxes = []

    # actual size of image that contains the bboxes
    y, x = actual_size

    # target size of the image that will contain the resized bboxes
    ty, tx = target_size

    x_scale = tx / x
    y_scale = ty / y

    for box in bbox:
        (origLeft, origTop, origRight, origBottom) = box
        x = int(np.round(origLeft * x_scale))
        y = int(np.round(origTop * y_scale))
        xmax = int(np.round(origRight * x_scale))
        ymax = int(np.round(origBottom * y_scale))
        boxes.append([x, y, xmax, ymax])
    return np.array(boxes)

@timer
def reformat_detections(outputs, actual_size, target_size):
    outputs = outputs[0]
    boxes = outputs['boxes'].detach().numpy()
    labels = outputs['labels'].detach().numpy()
    scores = np.round(outputs['scores'].detach().numpy(), 2)

    ind_candidates = np.where(scores > min_confidence)[0]

    boxes = boxes[ind_candidates, :]
    scores = scores[ind_candidates]
    labels = labels[ind_candidates]

    # TODO: Add non-maximal suppression here

    boxes = resize_boxes(boxes, actual_size=actual_size, target_size=target_size)

    return boxes, scores, labels


if __name__ == "__main__":
    _, device = cuda_setup()
    ocv_helper = OpencvHelper()
    detection_utils = DetectionUtils()

    model_path = Path("/home/medhyvinceslas/Documents/programming/helpers/detection/weights/detect_defect.pt")
    dataset_path = Path("/home/medhyvinceslas/Documents/programming/DL/datasets/defect_detection")
    image_folder = dataset_path / "images/images"
    img_path = image_folder / "inclusion/img_03_425503100_00066.jpg"

    assert model_path.is_file()
    assert img_path.is_file()

    num_classes = 11
    min_confidence = 0.3

    model = load_model_inference(model_path, num_classes, device)

    img = cv2.imread(str(img_path))
    orig_image = img.copy()
    img_resized = cv2.resize(img, (256, 256))
    img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    actual_size, target_size = img_resized.shape[:2], orig_image.shape[:2]

    img = normalize(img)
    img = to_tensor(img)
    results = predict(model, img, device)
    boxes, _, _ = reformat_detections(results, actual_size, target_size)

    im = detection_utils.draw_boxes_opencv(orig_image, boxes, thickness=2)
    ocv_helper.imshow(im, "out")

