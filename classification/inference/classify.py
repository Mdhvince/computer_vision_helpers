from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from torch import nn
import torch.nn.functional as F
from torchvision import models, datasets
from torchvision.models import ResNet18_Weights

from utils.gpu import cuda_setup
from utils.opencv_helpers import OpencvHelper

plt.style.use("ggplot")


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

def classify(model, tensor_img, idx_class_map, top_k, device):
    # we want to calculate gradient of highest score w.r.t. input, so we set requires_grad to True for input
    tensor_img.requires_grad = True
    logpas = model(tensor_img.to(device))
    probas = F.softmax(logpas, dim=1)

    score, indices = torch.max(logpas, 1)  # for the saliency map
    score.backward()  # backward pass to get gradients of score predicted class w.r.t. input image
    slc, _ = torch.max(torch.abs(tensor_img.grad[0]), dim=0)  # get max along channel axis
    slc = (slc - slc.min()) / (slc.max() - slc.min())  # normalize to [0..1]

    top_proba, top_label = probas.topk(top_k)
    top_proba = top_proba.detach().numpy().tolist()[0]
    top_label = top_label.detach().numpy().tolist()[0]

    # transform indices to classes name
    idx_to_class = {val: key for key, val in idx_class_map}
    top_labels = [idx_to_class[lab] for lab in top_label]

    return top_proba, top_labels, slc

def overlay(image, saliency):
    # Normalize to [0, 1] range
    image = (image - image.min()) / (image.max() - image.min())

    heatmap = cv2.applyColorMap(np.uint8(255 * saliency), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    combined = heatmap + np.float32(image)
    combined = combined / np.max(combined)
    combined = np.uint8(255 * combined)

    # heatmap = cm.hot(saliency)[..., :3]
    # combined = image * 0.4 + heatmap * 0.6
    return combined

def plot_solution(img, probas, labels, saliency):
    saliency = saliency.detach().numpy()

    plt.figure(figsize=(10, 10))

    ax = plt.subplot(2, 2, 1)
    plt.axis(False)
    plt.title("Input")
    ax.imshow(img)

    ax = plt.subplot(2, 2, 2)
    plt.axis(False)
    plt.title("Saliency map")
    ax.imshow(saliency, cmap=plt.cm.hot)

    plt.subplot(2, 2, 3)
    plt.title("Class probability")
    sns.barplot(x=probas, y=labels, color=sns.color_palette()[0])

    ax = plt.subplot(2, 2, 4)
    combined = overlay(img, saliency)
    ax.imshow(combined)
    plt.axis(False)
    plt.title("Combined")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    ROOT_DIR = Path("/home/medhyvinceslas/Documents/programming/datasets")
    TEST_DIR = ROOT_DIR / "plant_disease_dataset/Test/Test"
    MODEL_PATH = Path("/home/medhyvinceslas/Documents/programming/helpers/classification/weights/plant_disease.pt")
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

    probas, labels, saliency = classify(model, tensor_img, idx_class_map, top_k=len(classes), device=device)
    plot_solution(cropped, probas, labels, saliency)
