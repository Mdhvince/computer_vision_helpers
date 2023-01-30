from pathlib import Path
from lxml import etree

import cv2
import torch
import pandas as pd
import numpy as np
import torchvision.transforms as T
import albumentations as albu
from torch.utils.data import SubsetRandomSampler, DataLoader

from utils.camera import set_window_pos


class DefectDataset(torch.utils.data.Dataset):

    def __init__(self, label_folder, image_folder, transforms, im_size=800):
        self.transforms = transforms
        self.label_folder = label_folder
        self.image_folder = image_folder
        self.im_size = im_size

        self.data = self._get_data()

    def __getitem__(self, idx):
        data = self.data.iloc[idx]
        img_path = data.img_path
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        category = [Path(img_path).parent.name]

        # get bounding box coordinates for each objects
        num_objs = 1  # len(obj_ids)    here we have only one detection per image
        boxes = []
        xmin, ymin, xmax, ymax = int(data.xmin), int(data.ymin), int(data.xmax), int(data.ymax)
        boxes.append([xmin, ymin, xmax, ymax])

        # for i in range(num_objs):
        #     xmin = np.min(pos[1])
        #     xmax = np.max(pos[1])
        #     ymin = np.min(pos[0])
        #     ymax = np.max(pos[0])
        #     boxes.append([xmin, ymin, xmax, ymax])

        img, boxes = self._resize(img, np.array(boxes))

        if self.transforms is not None:
            img, boxes = self._augment_images_boxes(img, boxes, category, pOneOf=1, pCompose=1)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "image_id": image_id, "area": area, "iscrowd": iscrowd}

        final_transform = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        img = final_transform(img)

        return img, target

    def __len__(self):
        return len(self.data)

    def _resize(self, img, bbox):
        y = img.shape[0]
        x = img.shape[1]

        x_scale = self.im_size / x
        y_scale = self.im_size / y
        img = cv2.resize(img, (self.im_size, self.im_size))
        img = np.array(img)

        (origLeft, origTop, origRight, origBottom) = bbox.flatten()
        x = int(np.round(origLeft * x_scale))
        y = int(np.round(origTop * y_scale))
        xmax = int(np.round(origRight * x_scale))
        ymax = int(np.round(origBottom * y_scale))

        return img, [[x, y, xmax, ymax]]

    def _augment_images_boxes(self, image, bbox, category, pOneOf=1, pCompose=1):
        """
        """
        bbox = np.round(np.array(bbox), 1)
        bbox = [b.tolist() for b in bbox]
        annotations = {'image': image, 'bboxes': bbox, 'category_id': category}
        compose = albu.Compose(
            [albu.OneOf(self.transforms, p=pOneOf)],
            p=pCompose,
            bbox_params=albu.BboxParams(format='pascal_voc', label_fields=['category_id'])
        )
        transformed = compose(**annotations)
        im = transformed['image']
        bbox = [list(i) for i in transformed['bboxes']]
        return im, bbox

    def _get_data(self):
        all_data = []
        file_list = self.label_folder.glob('*.xml')
        for xml_file in file_list:
            df = self.__parse_xml(xml_file)
            all_data.append(df)

        data = pd.concat(all_data, ignore_index=True)
        return data

    def __parse_xml(self, xml_file_path):
        with open(xml_file_path, "rb") as f:
            xml = f.read()
        root = etree.fromstring(xml)
        annotation = root.find("object")
        bndbox = annotation.find("bndbox")

        filename = root.find("filename").text
        file = next((f for f in self.image_folder.rglob(filename) if f.is_file()), None)
        full_path = str(file.resolve())

        data = {"img_path": full_path,
                "width": root.find("size/width").text,
                "height": root.find("size/height").text,
                "xmin": bndbox.find("xmin").text,
                "ymin": bndbox.find("ymin").text,
                "xmax": bndbox.find("xmax").text,
                "ymax": bndbox.find("ymax").text}
        return pd.DataFrame(data, index=[0])


def collate_fn(batch):
    return tuple(zip(*batch))


def undo_transform(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp


def visualize_data_loader(data_loader):
    images, targets = next(iter(data_loader))
    im = images[0]
    im = undo_transform(im)
    im = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    bbox = targets[0]["boxes"]
    bbox = bbox[0].numpy()
    x1, y1, x2, y2 = np.array([int(b) for b in bbox])
    cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 255), 1, cv2.LINE_AA)

    return im


def train_valid_split(training_set, validation_size):
    """ Function that split our dataset into train and validation
        given in parameter the training set and the % of sample for validation"""

    # obtain training indices that will be used for validation
    num_train = len(training_set)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(validation_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    return train_sampler, valid_sampler


def build_loaders(dataset, batch_size, valid_size, num_workers):
    train_sampler, valid_sampler = train_valid_split(dataset, valid_size)
    train_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, collate_fn=collate_fn
    )
    valid_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers, collate_fn=collate_fn
    )
    return train_loader, valid_loader


if __name__ == "__main__":
    dataset_path = Path("/home/medhyvinceslas/Documents/programming/DL/datasets/defect_detection")
    label_folder = dataset_path / "label/label"
    image_folder = dataset_path / "images/images"
    batch_size = 8
    num_workers = 4
    valid_size = 0.25
    name_window = "img"

    transforms = [
        albu.HorizontalFlip(p=.5),
        albu.VerticalFlip(p=.5),
        albu.ShiftScaleRotate(),
    ]

    dataset = DefectDataset(label_folder, image_folder, transforms, im_size=800)
    # data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    train_loader, valid_loader = build_loaders(dataset, batch_size, valid_size, num_workers)

    im1 = visualize_data_loader(train_loader)
    im1 = cv2.cvtColor(im1, cv2.COLOR_RGB2BGR)

    im2 = visualize_data_loader(valid_loader)
    im2 = cv2.cvtColor(im2, cv2.COLOR_RGB2BGR)

    im = np.hstack((im1, im2))

    set_window_pos(name_window)
    cv2.imshow(name_window, im)
    cv2.waitKey()
