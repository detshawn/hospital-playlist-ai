from __future__ import print_function

import torch
import torch.nn as nn
import os
import torchvision.transforms as transforms
import numpy as np

import re
import cv2

from model.model import TransformerNet, VGG16
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

style_image_location = "./dataset/style/summeringiiwebsite.jpg"

style_image_sample = Image.open(style_image_location, 'r')
display(style_image_sample)

# config
batch_size = 8
random_seed = 10
num_epochs = 64
initial_lr = 1e-3
checkpoint_dir = "/content/gdrive/My Drive/Colab_Notebooks/data/" #FIXME

content_weight = 1e5
style_weight = 1e10
log_interval = 50
checkpoint_interval = 500

np.random.seed(random_seed)
torch.manual_seed(random_seed)

# set the pre-processing functions: transform(), style_transform()
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255))
])

style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
])

# model and optimizer construction
transformer = TransformerNet()
vgg = VGG16(requires_grad=False).to(device)

optimizer = torch.optim.Adam(transformer.parameters(), initial_lr)
mse_loss = nn.MSELoss()

# style image importing and pre-processing
style = load_image(filename=style_image_location, size=None, scale=None)
style = style_transform(style)
style = style.repeat(batch_size, 1, 1, 1).to(device)

features_style = vgg(normalize_batch(style))
gram_style = [gram_matrix(y) for y in features_style]

# inference data importing
if option == "image":
    content_image = load_image("./dataset/inference/hospital-playlist-test.jpg", scale=2)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device)

    with torch.no_grad():
        style_model = TransformerNet()

        ckpt_model_path = os.path.join(checkpoint_dir, "ckpt_epoch_63_batch_id_500.pth")  # FIXME
        checkpoint = torch.load(ckpt_model_path, map_location=device)

        # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
        for k in list(checkpoint.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del checkpoint[k]

        style_model.load_state_dict(checkpoint['model_state_dict'])
        style_model.to(device)

        output = style_model(content_image).cpu()

    save_image("/content/gdrive/My Drive/Colab_Notebooks/data/sanok_result.png", output[0])  # FIXME
elif option == "video":
    with torch.no_grad():
        style_model = TransformerNet()

        ckpt_model_path = os.path.join(checkpoint_dir, "ckpt_epoch_63_batch_id_500.pth")  # FIXME
        checkpoint = torch.load(ckpt_model_path, map_location=device)

        # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
        for k in list(checkpoint.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del checkpoint[k]

        style_model.load_state_dict(checkpoint['model_state_dict'])
        style_model.to(device)

        cap = cv2.VideoCapture("/content/gdrive/My Drive/Colab_Notebooks/data/mirama_demo.mp4")  # FIXME

        frame_cnt = 0

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('/content/gdrive/My Drive/Colab_Notebooks/data/mirama_demo_result.avi', fourcc, 60.0,
                              (1920, 1080))  # FIXME

        while (cap.isOpened()):
            ret, frame = cap.read()

            try:
                frame = frame[:, :, ::-1] - np.zeros_like(frame)
            except:
                break

            print(frame_cnt, "th frame is loaded!")

            content_image = frame
            content_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.mul(255))
            ])
            content_image = content_transform(content_image)
            content_image = content_image.unsqueeze(0).to(device)

            output = style_model(content_image).cpu()
            # save_image("/content/gdrive/My Drive/Colab_Notebooks/data/vikendi_video_result/" + str(frame_cnt) +".png", output[0]) #FIXME
            out.write(post_process_image(output[0]))
            frame_cnt += 1

        cap.release()
        out.release()
        cv2.destroyAllWindows()
