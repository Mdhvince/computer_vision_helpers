from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn
from torchvision import models, datasets
from torchvision.models import ResNet18_Weights

from utils.gpu import cuda_setup
from utils.opencv_helpers import OpencvHelper


# def process_image(image):
#     """ Scales, crops, and normalizes a PIL image for a PyTorch model, returns a Numpy array"""
#     img = Image.open(image)
#
#     # Resize image
#     if img.size[0] > img.size[1]:
#         img.thumbnail((10000, 256))
#     else:
#         img.thumbnail((256, 10000))
#
#     # Crop image
#     bottom_margin = (img.height - 224) / 2
#     top_margin = bottom_margin + 224
#     left_margin = (img.width - 224) / 2
#     right_margin = left_margin + 224
#
#     img = img.crop((left_margin, bottom_margin, right_margin, top_margin))
#
#     # Normalize image
#     img = np.array(img) / 255
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     img = (img - mean) / std
#
#     # move to first dimension --> PyTorch
#     img = img.transpose((2, 0, 1))
#
#     return img


def normalize(image):
    image = image / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    return image


def center_crop(image, size):
    center = image.shape
    x = center[1] / 2 - size / 2
    y = center[0] / 2 - size / 2
    img_crop = image[int(y):int(y + size), int(x):int(x + size)]
    return img_crop


def to_tensor(image):
    image = image.transpose((2, 0, 1))
    image = torch.from_numpy(image).type(torch.FloatTensor)
    image = torch.unsqueeze(image, 0)
    return image


def load_model(model_path, classes, device):
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    n_inputs = model.fc.in_features
    last_layer = nn.Linear(n_inputs, len(classes))
    model.fc = last_layer

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def get_classes(directory):
    data = datasets.ImageFolder(directory)
    classes = data.classes
    return data.class_to_idx.items(), classes


def classify(model, tensor_img, idx_class_map, top_k):
    logpas = model(tensor_img)
    proba = torch.exp(logpas)

    top_proba, top_label = proba.topk(top_k)
    top_proba = top_proba.detach().numpy().tolist()[0]
    top_label = top_label.detach().numpy().tolist()[0]

    # transform indices to classes name
    idx_to_class = {val: key for key, val in idx_class_map}
    top_labels = [idx_to_class[lab] for lab in top_label]

    return top_proba, top_labels


def plot_solution(img, probas, labels):
    plt.figure(figsize=(10, 10))

    ax = plt.subplot(2, 1, 1)
    ax.imshow(img)

    plt.subplot(2, 1, 2)
    sns.barplot(x=probas, y=labels, color=sns.color_palette()[0])

    plt.show()


if __name__ == "__main__":
    ROOT_DIR = Path("/home/medhyvinceslas/Documents/programming/datasets")
    TEST_DIR = ROOT_DIR / "plant_disease_dataset/Test/Test"
    MODEL_PATH = Path("/home/medhyvinceslas/Documents/programming/helpers/classification/weights/model.pt")
    ocv_helper = OpencvHelper()

    img_path = TEST_DIR / "Rust/85f0c2c0db4b4f4f.jpg"

    assert img_path.is_file()
    assert MODEL_PATH.is_file()

    _, device = cuda_setup()
    idx_class_map, classes = get_classes(TEST_DIR)
    model = load_model(MODEL_PATH, classes, device)

    img = cv2.imread(str(img_path))
    orig_image = img.copy()
    img_resized = cv2.resize(img, (256, 256))
    img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    cropped = center_crop(img, size=224)
    normalized_img = normalize(cropped)
    tensor_img = to_tensor(normalized_img)

    probas, labels = classify(model, tensor_img, idx_class_map, top_k=len(classes))
    plot_solution(img, probas, labels)
