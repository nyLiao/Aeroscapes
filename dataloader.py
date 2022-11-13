from ctypes.wintypes import LONG
import torch
import torch.utils.data
import numpy as np
import os
import random
from PIL import Image, ImageFile
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize,RandomVerticalFlip,\
                                    RandomHorizontalFlip,RandomRotation,ColorJitter,RandomErasing,RandomAdjustSharpness,GaussianBlur,Pad
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def _convert_image_to_rgb(image):
    return image.convert("RGB")
# txt_file = open("ImageSets/trn.txt")
# train_filenames = txt_file.readlines()
# for train_filename in train_filenames:
#     print(train_filename)

def train_data_transform(p1,p2):
    return Compose([
        # Resize((h,w), interpolation=BICUBIC),
        Pad((0,8)),
        RandomHorizontalFlip(p=p1),
        # RandomVerticalFlip(p=p2),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def train_label_transform(p1,p2):
    return Compose([
        Pad((0,8)),
        RandomHorizontalFlip(p=p1),
        # RandomVerticalFlip(p=p2),
        # Resize((h,w), interpolation=BICUBIC),
        # ToTensor(),
    ])

class DatasetTrain(torch.utils.data.Dataset):
    def __init__(self, base_dir):


        self.base_dir = base_dir
        self.img_dir = base_dir + "JPEGImages/"
        self.label_dir = base_dir + "SegmentationClass/"

        self.new_img_h = 512
        self.new_img_w = 1024

        self.examples = []
        txt_path = self.base_dir + "ImageSets/trn.txt"
        txt_file = open(txt_path)
        train_filenames = txt_file.readlines()

        train_img_dir_path = self.img_dir
        label_img_dir_path = self.label_dir

        for train_filename in train_filenames:
            train_filename=train_filename.strip('\n')
            img_path = train_img_dir_path + train_filename + '.jpg'
            label_img_path = label_img_dir_path + train_filename + '.png'
            example = {}
            example["img_path"] = img_path
            example["label_img_path"] = label_img_path
            # Over-sampling on things
            label_img = Image.open(label_img_path).convert('L')
            label_img = np.array(label_img,dtype=LONG)
            thing_count =0
            for i in range(1,8):
                thing_count+=np.sum(label_img==i)
            # print(thing_count)
            if thing_count/(1280*720)>0.0151:
                self.examples.append(example)
            self.examples.append(example)

        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        img_path = example["img_path"]
        # print(img_path)
        img = Image.open(img_path)
        # img = cv2.resize(img, (self.new_img_w, self.new_img_h),
        #                  interpolation=cv2.INTER_NEAREST)
        label_img_path = example["label_img_path"]
        # print(label_img_path)
        label_img = Image.open(label_img_path).convert('L')
        # label_img = cv2.resize(label_img, (self.new_img_w, self.new_img_h),
                            #    interpolation=cv2.INTER_NEAREST)

        # normalize the img (with the mean and std for the pretrained ResNet):
        # img = img/255.0
        # img = img - np.array([0.485, 0.456, 0.406])
        # img = img/np.array([0.229, 0.224, 0.225])
        # img = np.transpose(img, (2, 0, 1))
        # img = img.astype(np.float32)

        # Random transforms
        if random.random()> 0.5:
            p1 = 1
        else:
            p1 = 0
        if random.random()> 0.5:
            p2 = 1
        else:
            p2 = 0
        img = train_data_transform(p1,p2)(img)
        label_img = train_label_transform(p1,p2)(torch.from_numpy(np.array(label_img,dtype=LONG)))
        # print(torch.where(label_img==2))
        return (img, label_img)

    def __len__(self):
        return self.num_examples

def val_data_transform(h,w):
    return Compose([
        Pad((0,8)),
        # Resize((h,w), interpolation=BICUBIC),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def val_label_transform():
    return Compose([
        Pad((0,8)),
        # Resize((h,w), interpolation=BICUBIC),
        # ToTensor(),
    ])

class DatasetVal(torch.utils.data.Dataset):
    def __init__(self, base_dir):


        self.base_dir = base_dir
        self.img_dir = base_dir + "JPEGImages/"
        self.label_dir = base_dir + "SegmentationClass/"

        self.new_img_h = 512
        self.new_img_w = 1024

        self.examples = []
        txt_path = self.base_dir + "ImageSets/val.txt"
        txt_file = open(txt_path)
        valid_filenames = txt_file.readlines()

        train_img_dir_path = self.img_dir
        label_img__dir_path = self.label_dir

        for valid_filename in valid_filenames:
            valid_filename=valid_filename.strip('\n')
            img_path = train_img_dir_path + valid_filename + '.jpg'
            label_img_path = label_img__dir_path + valid_filename + '.png'
            example = {}
            example["img_path"] = img_path
            example["label_img_path"] = label_img_path
            self.examples.append(example)

        self.num_examples = len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        img_path = example["img_path"]
        # print(img_path)
        img = Image.open(img_path)
        # img = cv2.resize(img, (self.new_img_w, self.new_img_h),
                        #  interpolation=cv2.INTER_NEAREST)
        label_img_path = example["label_img_path"]
        # print(label_img_path)
        label_img = Image.open(label_img_path)
        # label_img = cv2.resize(label_img, (self.new_img_w, self.new_img_h),
        #                        interpolation=cv2.INTER_NEAREST)

        # normalize the img (with the mean and std for the pretrained ResNet):
        # img = img/255.0
        # img = img - np.array([0.485, 0.456, 0.406])
        # img = img/np.array([0.229, 0.224, 0.225])
        # img = np.transpose(img, (2, 0, 1))
        # img = img.astype(np.float32)

        # convert numpy -> torch:
        # img = torch.from_numpy(img)
        # label_img = torch.from_numpy(label_img)
        img = val_data_transform(self.new_img_h,self.new_img_w)(img)
        label_img = val_label_transform()(torch.from_numpy(np.array(label_img,dtype=LONG)))
        return (img, label_img)

    def __len__(self):
        return self.num_examples


class DatasetClean(DatasetVal):
    def __getitem__(self, index):
        example = self.examples[index]

        img_path = example["img_path"]
        # print(img_path)
        img = Image.open(img_path)
        label_img_path = example["label_img_path"]
        # print(label_img_path)
        label_img = Image.open(label_img_path)
        img = np.array(img,dtype=LONG)
        label_img = np.array(label_img,dtype=LONG)
        return (img, label_img)
