from torchvision.models.alexnet import alexnet
from torchvision.models.vgg import vgg11, vgg13, vgg16, vgg19
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152

from PIL import Image
from torchvision.transforms.functional import to_pil_image, to_tensor
import torch
import torch.nn as nn
from torch.nn.functional import mse_loss, l1_loss, cosine_similarity

from reorganize_features import AlexNet, VGG, ResNet


if __name__ == '__main__':
    # model = AlexNet(alexnet(pretrained=True)).features.cuda().eval()
    # model = VGG(vgg16(pretrained=True)).features.cuda().eval()
    model = ResNet(resnet18(pretrained=True)).features.cuda().eval()
    # model = nn.DataParallel(ResNet(resnet18()).features).cuda().eval()
    # model.load_state_dict(torch.load('checkpoint.pth.tar')['state_dict'])

    # ex1_path = 'examples/0.JPEG'  # ResNet18 (L1, L2, Cos.Sim) / VGG16 (L1, L2, Cos.Sim)
    # comparing with smaller examples
    ex1_path = '0_small_4.png'  # 0.8146949410438538 / 1.6619117259979248 / 0.7260609269142151 (orig.size)

    # ex2_path = '0_bicubic_2.png'  # 0.28164124488830566 / 0.19226132333278656 / 0.862919807434082 / 0.1395910680294037 / 0.07891430705785751 / 0.7044201493263245
    # ex2_path = '0_bicubic_4.png'  # 0.3377046585083008 / 0.2587481439113617 / 0.8025323152542114 / 0.1792815625667572 / 0.11959114670753479 / 0.5236344337463379
    # ex2_path = '0_small_2.png'  # 0.3664081394672394 / 0.2703591287136078 / 0.8545681834220886 / 0.20839443802833557 / 0.23043273389339447 / 0.6931169033050537
    # ex2_path = '0_small_4.png'  # 0.8237079977989197 / 1.596572995185852 / 0.6894648671150208 / 0.3588990569114685 / 0.7935216426849365 / 0.3945225179195404
    # ex2_path = 'examples/1.JPEG'  # 0.2607431411743164 / 0.13753172755241394 / 0.9295168519020081 / 0.11814342439174652 / 0.06385411322116852 / 0.8351526260375977
    # ex2_path = 'examples/2.JPEG'  # 0.31596672534942627 / 0.1941777765750885 / 0.8718470931053162 / 0.14820162951946259 / 0.09880480170249939 / 0.6887456774711609
    # ex2_path = 'examples/3.JPEG'  # 0.3552648425102234 / 0.2614906430244446 / 0.8005115985870361 / 0.17826944589614868 / 0.10921414196491241 / 0.5870909094810486
    # ex2_path = 'examples/4.JPEG'  # 0.35923922061920166 / 0.2426682710647583 / 0.8648840188980103 / 0.17575465142726898 / 0.12326934933662415 / 0.6567056179046631
    ex2_path = 'examples/5.JPEG'  # 0.3559305667877197 / 0.2539767026901245 / 0.8408498167991638 / 0.19091153144836426 / 0.14342008531093597 / 0.5058000683784485

    ex1 = Image.open(ex1_path).convert('RGB')
    ex1 = to_tensor(ex1).unsqueeze(0).cuda()

    ex2 = Image.open(ex2_path).convert('RGB')
    ex2 = to_tensor(ex2).unsqueeze(0).cuda()

    with torch.no_grad():
        feats_ex1 = model(ex1)
        feats_ex2 = model(ex2)

    print('Difference: {} / {} / {}'
          .format(l1_loss(feats_ex1, feats_ex2), mse_loss(feats_ex1, feats_ex2),
                  cosine_similarity(feats_ex1, feats_ex2).item()))

# smaller2 vs bicubic2
# Difference: 0.165213942527771 / 0.06672841310501099 / 0.9227539896965027

# Original vs bicubic2
# Difference: 0.13744446635246277 / 0.04521540552377701 / 0.8635921478271484

# Original vs smaller2
# Difference: 0.18768639862537384 / 0.08458086848258972 / 0.8467699885368347


#####################################################################################


# smaller4 vs bicubic4
# Difference: 0.5082474946975708 / 0.7599782943725586 / 0.716705322265625

# Original vs bicubic4
# Difference: 0.16246119141578674 / 0.071178138256073 / 0.7739406228065491

# Original vs smaller4
# Difference: 0.5073211789131165 / 0.737619161605835 / 0.6713756918907166


#####################################################################################

# 0 vs 1
# Difference: 0.11459161341190338 / 0.03190845251083374 / 0.9286258220672607

# 0 vs 2
# Difference: 0.16591233015060425 / 0.07279087603092194 / 0.7986683249473572

# 0 vs 3
# Difference: 0.18255460262298584 / 0.08843141049146652 / 0.7117946147918701

# 0 vs 4
# Difference: 0.20152181386947632 / 0.0922960638999939 / 0.8164383769035339

# 0 vs 5
# Difference: 0.1778615415096283 / 0.06924337148666382 / 0.8272113800048828



