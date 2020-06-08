from __future__ import print_function

from argparse import ArgumentParser

import os
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np

import time
import glob

from model.learnable_loss import *
from model.transformer_net import TransformerNet, VGG16
from utils import *

from gdrive import upload


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device: {device}')

    # config
    style_image_path = args.style_image_path

    tag = args.tag
    train_dataset_dir = args.train_dataset_dir
    train_dataset_subdir = args.train_dataset_subdir

    batch_size = args.batch_size
    random_seed = 10
    num_epochs = args.num_epochs
    initial_lr = args.initial_lr

    log_interval = args.log_interval
    log_dir = args.log_dir
    logger = Logger(log_dir)
    checkpoint_interval = args.checkpoint_interval
    checkpoint_dir = args.checkpoint_dir
    upload_by_epoch = args.upload_by_epoch
    defined_ckpt_filename = args.ckpt_filename
    def get_saved_ckpt_filename(_epoch, _batch_id):
        return "ckpt_epoch_" + str(_epoch) + "_batch_id_" + str(_batch_id + 1) + ".ckpt"

    transfer_learning = False  # inference or training first --> False / Transfer learning --> True

    # style_image_sample = Image.open(style_image_path, 'r')
    # display(style_image_sample)

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # model and optimizer construction
    transformer = TransformerNet()
    trainer = LearnableLoss(model=transformer,
                            loss_names=['content', 'style', 'total_variation'],
                            device=device)
    vgg = VGG16(requires_grad=False).to(device)

    # transfer learning set-up and existing model loading (is it first-time or continued learning?)
    ckpt_model_path = os.path.join(checkpoint_dir, defined_ckpt_filename)
    if transfer_learning:
        checkpoint = torch.load(ckpt_model_path, map_location=device)
        trainer.load_state_dict(checkpoint['model_state_dict'])
        transfer_learning_epoch = checkpoint['epoch']
    else:
        transfer_learning_epoch = 0

    trainer.to(device)
    optimizer = torch.optim.Adam(trainer.parameters(), initial_lr)

    # desired size of the output image
    imsize = 256 if torch.cuda.is_available() else 128  # use small size if no gpu

    # training data importing and pre-processing
    transform = transforms.Compose([
        transforms.Resize(imsize),
        transforms.CenterCrop(imsize),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    print(f'train dataset list: {glob.glob("/".join([train_dataset_dir, train_dataset_subdir]) + "/*")}')
    train_dataset = datasets.ImageFolder(train_dataset_dir, transform)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=8,
                                               pin_memory=True)

    # style image importing and pre-processing
    styles = []
    if os.path.isdir(style_image_path):
        paths = glob.glob(os.path.join(style_image_path, f'*'))
        for path in paths:
            styles.append(load_image(filename=path, size=None, scale=None))
    else:
        styles.append(load_image(filename=style_image_path, size=None, scale=None))
    style_transform = transforms.Compose([
        transforms.Resize(imsize*4),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    styles = [style_transform(style) for style in styles]
    if not os.path.exists('./output'):
        os.mkdir('./output')
    for i, style in enumerate(styles):
        save_image(f'./output/style_transformed[{i}].png', style)
    styles = [style.repeat(batch_size, 1, 1, 1).to(device) for style in styles]

    # check the size
    # print(f'train_dataset[0][0].size(): {train_dataset[0][0].size()}')
    # print(f'style[0].size(): {style[0].size()}')

    # pre-calculating gram_style
    features_styles = [vgg(normalize_batch(style)) for style in styles]
    gram_styles = [[gram_matrix(y) for y in features_style] for features_style in features_styles]

    # training
    for epoch in range(transfer_learning_epoch, num_epochs):
        trainer.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        agg_total_variation_loss = 0.
        count = 0

        gram_style_itr = iter(gram_styles)

        for batch_id, (x, _) in enumerate(train_loader):
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()

            # forward
            x = x.to(device)
            y = trainer(x)

            y = normalize_batch(y)
            x = normalize_batch(x)

            features_y = vgg(y)
            features_x = vgg(x)

            # losses
            try:
                gram_style = next(gram_style_itr)
            except StopIteration:
                gram_style_itr = iter(gram_styles)
                gram_style = next(gram_style_itr)

            content_loss = get_content_loss(features_y.relu2_2, features_x.relu2_2)
            style_loss = get_style_loss(features_y, gram_style, n_batch)
            total_variation_loss = get_total_variation_loss(y)

            total_loss, meta = trainer.get_total_loss([content_loss, style_loss, total_variation_loss])

            # backward
            total_loss.backward()
            optimizer.step()

            agg_content_loss += meta['loss']['content']
            agg_style_loss += meta['loss']['style']
            agg_total_variation_loss += meta['loss']['total_variation']

            if (batch_id + 1) % log_interval == 0 or batch_id + 1 == len(train_loader.dataset):
                logger.scalars_summary(f'{tag}/train', meta['loss'], epoch * len(train_loader.dataset) + count + 1)
                logger.scalars_summary(f'{tag}/train_eta', meta['eta'], epoch * len(train_loader.dataset) + count + 1)

                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal_variation: {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(), epoch + 1, count, len(train_dataset),
                    agg_content_loss / (batch_id + 1),
                    agg_style_loss / (batch_id + 1),
                    agg_total_variation_loss / (batch_id + 1),
                    (agg_content_loss + agg_style_loss + agg_total_variation_loss) / (batch_id + 1)
                )
                print(mesg)

            if checkpoint_dir is not None and (batch_id + 1) % checkpoint_interval == 0:
                trainer.eval().cpu()
                saved_ckpt_filename = get_saved_ckpt_filename(epoch, batch_id)
                if not os.path.exists(checkpoint_dir):
                    os.mkdir(checkpoint_dir)
                ckpt_model_path = os.path.join(checkpoint_dir, saved_ckpt_filename)
                torch.save({
                'epoch': epoch,
                'model_state_dict': trainer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss
                }, ckpt_model_path)
                print(str(epoch), "th checkpoint is saved!")

                if epoch + 1 % upload_by_epoch == 0:
                    try:
                        upload(ckpt_model_path)
                    except:
                        1

                trainer.to(device).train()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-tag', '-t', required=True)
    # train_dataset_dir = "~/Documents/proj/ML/Deep Learning/cocodataset"
    # train_dataset_subdir = "val2017"
    parser.add_argument('-train_dataset_dir', required=True)
    parser.add_argument('-train_dataset_subdir', required=True)

    parser.add_argument('-style_image_path', default="./dataset/style")

    parser.add_argument('-batch_size', default=8, type=int)
    parser.add_argument('-num_epochs', default=64, type=int)
    parser.add_argument('-initial_lr', default=1e-3, type=float)

    parser.add_argument('-log_interval', default=50, type=int)
    parser.add_argument('-log_dir', default='./log')
    parser.add_argument('-checkpoint_interval', default=500, type=int)

    parser.add_argument('-checkpoint_dir', default='./ckpts/')
    parser.add_argument('-ckpt_filename', default=None)
    parser.add_argument('-upload_by_epoch', default=10, type=int)

    parser.add_argument('--transfer_learning', action='store_true')

    args = parser.parse_args()

    if args.ckpt_filename is None:
        args.ckpt_filename = f"ckpt_epoch_{args.num_epochs-1}_batch_id_{args.checkpoint_interval}.ckpt"

    main()
