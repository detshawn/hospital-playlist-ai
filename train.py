from __future__ import print_function

import torch
import torch.nn as nn
import os
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np

import time
import glob

from model.model import TransformerNet, VGG16
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

style_image_location = "/content/gdrive/My Drive/Colab_Notebooks/data/vikendi.jpg" #FIXME

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

# training data importing and pre-processing
print(glob.glob("/content/gdrive/My Drive/Colab_Notebooks/data/COCO/val2017/*"))
train_dataset = datasets.ImageFolder("/content/gdrive/My Drive/Colab_Notebooks/data/COCO", transform) #FIXME
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

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

# transfer learning setting (is it first-time or continued learning?)
transfer_learning = False # inference or training first --> False / Transfer learning --> True
ckpt_model_path = os.path.join(checkpoint_dir, "ckpt_epoch_63_batch_id_500.pth") #FIXME

if transfer_learning:
    checkpoint = torch.load(ckpt_model_path, map_location=device)
    transformer.load_state_dict(checkpoint['model_state_dict'])
    transformer.to(device)
    transfer_learning_epoch = checkpoint['epoch']
else:
    transfer_learning_epoch = 0

# training
for epoch in range(transfer_learning_epoch, num_epochs):
    transformer.train()
    agg_content_loss = 0.
    agg_style_loss = 0.
    count = 0

    for batch_id, (x, _) in enumerate(train_loader):
        n_batch = len(x)
        count += n_batch
        optimizer.zero_grad()

        # forward
        x = x.to(device)
        y = transformer(x)

        y = normalize_batch(y)
        x = normalize_batch(x)

        features_y = vgg(y)
        features_x = vgg(x)

        # losses
        content_loss = content_weight * mse_loss(features_y.relu2_2, features_x.relu2_2)

        style_loss = 0.
        for ft_y, gm_s in zip(features_y, gram_style):
            gm_y = gram_matrix(ft_y)
            style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])
        style_loss *= style_weight

        total_loss = content_loss + style_loss

        # backward
        total_loss.backward()
        optimizer.step()

        agg_content_loss += content_loss.item()
        agg_style_loss += style_loss.item()

        if (batch_id + 1) % log_interval == 0:
            mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                time.ctime(), epoch + 1, count, len(train_dataset),
                              agg_content_loss / (batch_id + 1),
                              agg_style_loss / (batch_id + 1),
                              (agg_content_loss + agg_style_loss) / (batch_id + 1)
            )
            print(mesg)

        if checkpoint_dir is not None and (batch_id + 1) % checkpoint_interval == 0:
            transformer.eval().cpu()
            ckpt_model_filename = "ckpt_epoch_" + str(epoch) + "_batch_id_" + str(batch_id + 1) + ".pth"
            print(str(epoch), "th checkpoint is saved!")
            ckpt_model_path = os.path.join(checkpoint_dir, ckpt_model_filename)
            torch.save({
            'epoch': epoch,
            'model_state_dict': transformer.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss
            }, ckpt_model_path)

            transformer.to(device).train()
