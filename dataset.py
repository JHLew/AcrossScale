from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import default_loader, pil_loader, IMG_EXTENSIONS
import random
from random import randint
import torch
import numpy as np


# do not always need to be centered.
def zero_pad_img(img, target_size=224):
    w, h = img.size
    offset_w, offset_h = target_size - w, target_size - h
    left = randint(0, offset_w)
    top = randint(0, offset_h)
    padded_img = Image.new(img.mode, (target_size, target_size), (0, 0, 0))
    padded_img.paste(img, (left, top))
    return padded_img


def tensor2img(tensor, mean=np.array([0.485, 0.456, 0.406]), std=np.array([0.229, 0.224, 0.225])):
    # c h w to h w c
    tensor = tensor.permute(1, 2, 0)

    tensor = (tensor * std) + mean
    tensor = tensor * 255
    img = Image.fromarray(tensor.numpy().astype(np.uint8)).convert('RGB')

    return img


def save_batch(batch, tag='dummy'):
    b, c, h, w = batch.shape
    for i in range(b):
        img = tensor2img(batch[i].cpu())
        img.save('dummy/{}_{}.png'.format(i, tag))


class custom_data(DatasetFolder):
    def __init__(self, data_path, default_transform, target_size=224, is_train=True):
        super(custom_data, self).__init__(root=data_path, loader=default_loader, extensions=IMG_EXTENSIONS)
        self.def_transf = default_transform
        self.target_size = target_size
        self.is_train = is_train

        if self.is_train:
            grayscale = transforms.RandomGrayscale()
            colorjitter = transforms.ColorJitter()
            hflip = transforms.RandomHorizontalFlip()
            self.augmentation = transforms.Compose([
                colorjitter, grayscale, hflip
            ])
            self.rrc = transforms.RandomResizedCrop(target_size)

        else:
            self.augmentation = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(target_size),
            ])

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.is_train:
            # conduct augmentation
            sample = self.augmentation(sample)
            rotation = random.choice([0, 90, 180, 270])
            sample = sample.rotate(rotation)

            # random resized crop for base sample image
            i, j, h, w = self.rrc.get_params(sample, self.rrc.scale, self.rrc.ratio)
            _size = randint(1, self.target_size)
            _size = (_size, _size)
            sample = F.resized_crop(sample, i, j, h, w, _size, self.rrc.interpolation)
            smaller_sample = zero_pad_img(sample, self.target_size)

            # resize the base sample image
            new_size = self.target_size
            interpolation = random.choice([Image.NEAREST, Image.BILINEAR, Image.BICUBIC])
            larger_sample = sample.resize(size=(new_size, new_size), resample=interpolation)

        else:
            sample = self.augmentation(sample)
            smaller_sample = sample.resize(size=(self.target_size // 2, self.target_size // 2), resample=Image.BICUBIC)
            larger_sample = smaller_sample.resize(size=(self.target_size, self.target_size), resample=Image.BICUBIC)

        # to tensor
        smaller_sample = self.def_transf(smaller_sample)
        larger_sample = self.def_transf(larger_sample)

        return smaller_sample, larger_sample, target



