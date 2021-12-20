from torchsummary import summary
from torchvision.models.alexnet import alexnet
from torchvision.models.vgg import vgg11, vgg13, vgg16, vgg19
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
import torch
import random
from PIL import Image
import torchvision.transforms as transforms

# ckpt = torch.load('./model_best.pth.tar')
# print(ckpt['state_dict'])
resizedcrop = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
            ])

img = Image.open('examples/0.JPEG')
img = resizedcrop(img)
img.save('0_224.png')
w, h = img.size
img = img.resize((w // 4, h // 4), resample=Image.BICUBIC)

img.save('0_224_small_4.png')
print(img.size)
img = img.resize((w, h), resample=Image.BICUBIC)
print(img.size)
img.save('0_224_bicubic_4.png')

# ex1_path = 'C:/Users/JH/Datasets/ImageNet/val/n01871265/ILSVRC2012_val_00000067.JPEG'
# ex2_path = 'C:/Users/JH/Datasets/ImageNet/val/n01871265/ILSVRC2012_val_00012976.JPEG'  # other 1
# ex2_path = 'C:/Users/JH/Datasets/ImageNet/val/n01871265/ILSVRC2012_val_00006725.JPEG'  # other 2
# ex2_path = 'C:/Users/JH/Datasets/ImageNet/val/n01871265/ILSVRC2012_val_00038178.JPEG'  # other 3
# ex2_path = 'C:/Users/JH/Datasets/ImageNet/val/n01871265/ILSVRC2012_val_00046452.JPEG'  # other 4
# ex2_path = 'C:/Users/JH/Datasets/ImageNet/val/n01871265/ILSVRC2012_val_00047945.JPEG'  # other 5
# ex2_path = 'C:/Users/JH/Datasets/ImageNet/val/n01882714/ILSVRC2012_val_00003254.JPEG'  # other 6 - koala

