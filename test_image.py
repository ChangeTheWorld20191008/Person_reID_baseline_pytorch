from model import PCB, PCB_test
import torch
from torchvision import transforms
import cv2
from PIL import Image
import numpy as np
from torch.autograd import Variable


def load_network(network, model_path):
    network.load_state_dict(torch.load(model_path))
    return network


def preprocess(img_path) -> torch.Tensor:
    """
    Image preprocessing
    Args:
        device (torch.device): deploy data to 'gpu' or 'cpu'
    Returns:
        torch.Tensor: preprocessed image
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    trans = transforms.Compose([
        transforms.Resize((384, 192), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = trans(img).unsqueeze(0)

    inv_idx = torch.arange(img.size(3)-1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)

    input_image = Variable(img_flip.cuda())

    return input_image


def fliplr(img):
    inv_idx = torch.arange(img.size(3)-1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip


model_path = "/home/zhuhao/myModel/human_reid/model/PCB/net_last.pth"
model_structure = PCB(751)
model = load_network(model_structure, model_path)
model = PCB_test(model)
model = model.eval()
model = model.cuda()

img_path = "/home/zhuhao/dataset/human_reid/attributes/cropped_data/0001.jpg"
input_img = preprocess(img_path)

outputs = model(input_img)
# ---- L2-norm Feature ------
ff = outputs.data.cpu()

# for PCB
fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6)
ff = ff.div(fnorm.expand_as(ff))
ff = ff.view(ff.size(0), -1)

feature = ff.detach().numpy().tolist()
print(f"[TMP]: feature type is {type(feature[0])}")
print(f"[TMP]: length is {len(feature[0])}")
