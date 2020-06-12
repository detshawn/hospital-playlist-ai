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


def get_saved_ckpt_filename(_epoch, _batch_id):
    return "ckpt_epoch_" + str(_epoch) + "_batch_id_" + str(_batch_id + 1) + ".ckpt"


def get_data_loader():
    transform = transforms.Compose([
        transforms.Resize(args.imsize),
        transforms.CenterCrop(args.imsize),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    print(f'train dataset list: {glob.glob("/".join([args.train_dataset_dir, args.train_dataset_subdir]) + "/*")}')
    train_dataset   = datasets.ImageFolder(args.train_dataset_dir, transform)
    val_dataset     = None
    if args.val_dataset_dir is not None:
        print(f'val dataset list: {glob.glob("/".join([args.val_dataset_dir, args.val_dataset_subdir]) + "/*")}')
        val_dataset = datasets.ImageFolder(args.val_dataset_dir, transform)
    else:
        l = len(train_dataset)
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset,
                                                                   [l - int(l * args.val_set_ratio),
                                                                    int(l * args.val_set_ratio)])

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=8,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=8,
                                             pin_memory=True)

    return train_loader, val_loader


def get_gram_styles(vgg):
    if not os.path.exists('./output'):
        os.mkdir('./output')
    style_transform = transforms.Compose([
        transforms.Resize(args.imsize*4),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    gram_styles = []
    if os.path.isdir(args.style_image_path):
        paths = glob.glob(os.path.join(args.style_image_path, f'*'))
    else:
        paths = [args.style_image_path]
    for i, path in enumerate(paths):
        # import the image
        style = load_image(filename=path, size=None, scale=None)

        # transform the image into a tensor
        style_t = style_transform(style)
        save_image(f'./output/style_transformed[{i}].png', style_t)
        style_t = style_t.repeat(args.batch_size, 1, 1, 1).to(args.device)

        # check the size
        # print(f'train_dataset[0][0].size(): {train_dataset[0][0].size()}')
        # print(f'style_t[0].size(): {style_t[0].size()}')

        # forward propagation of pre-trained net and derivation of a Gram matrix
        features_style = vgg(normalize_batch(style_t))
        gram_styles.append([gram_matrix(y) for y in features_style])

        del style_t, features_style

    return gram_styles


def permuted_iterator(x):
    ps = np.random.permutation(len(x))
    return iter([x[p] for p in ps])


def build_step_fn(trainer, vgg, optimizer):
    def _step(x, train=True):
        if train:
            optimizer.zero_grad()

        # forward prop
        x = x.to(args.device)
        y = trainer(x)

        samples = {}
        samples['x'] = x[:3].clone().detach().div_(255.)
        samples['y'] = y[:3].clone().detach().div_(255.)

        y = normalize_batch(y)
        x = normalize_batch(x)

        features_y = vgg(y)
        features_x = vgg(x)

        # losses
        try:
            gram_style = next(args.gram_style_itr)
        except StopIteration:
            args.gram_style_itr = permuted_iterator(args.gram_styles)
            gram_style = next(args.gram_style_itr)

        content_loss = get_content_loss(features_y.relu2_2, features_x.relu2_2)
        style_loss = get_style_loss(features_y, gram_style, len(x))
        total_variation_loss = get_total_variation_loss(y)

        total_loss, meta = trainer.get_total_loss([content_loss, style_loss, total_variation_loss])

        # backward prop and update parameters
        if train:
            total_loss.backward()
            optimizer.step()
        else:
            meta['val_loss'] = {}
            for k, v in meta['loss'].items():
                meta['val_loss']['val_' + k] = v
            del meta['loss']
            meta['val_eta'] = {}
            for k, v in meta['eta'].items():
                meta['val_eta']['val_' + k] = v
            del meta['eta']

        del x, y, features_x, features_y
        del content_loss, style_loss, total_variation_loss, total_loss

        return meta, samples
    return _step


def train(trainer, vgg, optimizer, transfer_learning_epoch,
          train_loader, val_loader):
    logger = Logger(args.log_dir)
    step = build_step_fn(trainer, vgg, optimizer)

    # training
    for epoch in range(transfer_learning_epoch, args.num_epochs):
        trainer.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        agg_total_variation_loss = 0.
        agg_val_content_loss = 0.
        agg_val_style_loss = 0.
        agg_val_total_variation_loss = 0.
        count = 0
        count_val = 0

        args.gram_style_itr = permuted_iterator(args.gram_styles)
        val_iter = iter(val_loader)
        for batch_id, (x, _) in enumerate(train_loader):
            count += len(x)

            meta, _ = step(x)

            agg_content_loss += meta['loss']['content']
            agg_style_loss += meta['loss']['style']
            agg_total_variation_loss += meta['loss']['total_variation']

            # logging
            if (batch_id + 1) % args.log_interval == 0 or batch_id + 1 == len(train_loader.dataset):
                logger.scalars_summary(f'{args.tag}/train', meta['loss'], epoch * len(train_loader.dataset) + count + 1)
                logger.scalars_summary(f'{args.tag}/train_eta', meta['eta'], epoch * len(train_loader.dataset) + count + 1)

                mesg = "{}\tEpoch {}:\t[{}/{}]\tbatch_id: {}\t" \
                       "content: {:.6f}\tstyle: {:.6f}\ttotal_variation: {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(), epoch + 1, count, len(train_loader.dataset), batch_id,
                    agg_content_loss / (batch_id + 1),
                    agg_style_loss / (batch_id + 1),
                    agg_total_variation_loss / (batch_id + 1),
                    (agg_content_loss + agg_style_loss + agg_total_variation_loss) / (batch_id + 1)
                )
                print(mesg)

            # validation
            if (batch_id + 1) % int(1 / args.val_set_ratio) == 0:
                batch_id_val = int((batch_id + 1) / int(1 / args.val_set_ratio))
                try:
                    x_val, _ = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_loader)
                    x_val, _ = next(val_iter)
                count_val += len(x_val)
                trainer.eval()
                with torch.no_grad():
                    meta_val, samples = step(x_val, train=False)
                trainer.train()

                agg_val_content_loss += meta_val['val_loss']['val_content']
                agg_val_style_loss += meta_val['val_loss']['val_style']
                agg_val_total_variation_loss += meta_val['val_loss']['val_total_variation']

                if (batch_id + 1) % args.log_interval == 0 or batch_id + 1 == len(train_loader.dataset):
                    logger.scalars_summary(f'{args.tag}/train', meta_val['val_loss'],
                                           epoch * len(train_loader.dataset) + count + 1)
                    logger.scalars_summary(f'{args.tag}/train_eta', meta_val['val_eta'],
                                           epoch * len(train_loader.dataset) + count + 1)

                    mesg = "{}\tEpoch {}:\t[{}/{}]\tbatch_id_val: {}\t" \
                           "val_content: {:.6f}\tval_style: {:.6f}\tval_total_variation: {:.6f}\tval_total: {:.6f}".format(
                        time.ctime(), epoch + 1, count_val, len(val_loader.dataset), batch_id_val,
                                      agg_val_content_loss / (batch_id_val + 1),
                                      agg_val_style_loss / (batch_id_val + 1),
                                      agg_val_total_variation_loss / (batch_id_val + 1),
                                      (agg_val_content_loss + agg_val_style_loss + agg_val_total_variation_loss) / (batch_id_val + 1)
                    )
                    print(mesg)

                if args.checkpoint_dir is not None and (batch_id + 1) % args.checkpoint_interval == 0:
                    # sampling and saving the result
                    for i in range(samples['x'].size()[0]):
                        logger.image_summary(f'{args.tag}/img{i}_x', samples['x'][i], epoch)
                        logger.image_summary(f'{args.tag}/img{i}_y', samples['y'][i], epoch)

            # checkpoint saving
            if args.checkpoint_dir is not None and (batch_id + 1) % args.checkpoint_interval == 0:
                trainer.eval().cpu()
                saved_ckpt_filename = get_saved_ckpt_filename(epoch, batch_id)
                if not os.path.exists(args.checkpoint_dir):
                    os.mkdir(args.checkpoint_dir)
                ckpt_model_path = os.path.join(args.checkpoint_dir, saved_ckpt_filename)
                torch.save({
                'epoch': epoch,
                'model_state_dict': trainer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': meta['loss']['total']
                }, ckpt_model_path)
                print(str(epoch), "th checkpoint is saved!")

                # uploading the saved ckpt file to Google Drive
                if args.gdrive and epoch + 1 % args.upload_by_epoch == 0:
                    try:
                        upload(ckpt_model_path)
                    except:
                        1

                trainer.to(args.device).train()


def main():
    # config
    random_seed = 10
    initial_lr = args.initial_lr
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # training data importing and pre-processing
    train_loader, val_loader = get_data_loader()

    # model construction
    transformer = TransformerNet()
    trainer = LearnableLoss(model=transformer,
                            loss_names=['content', 'style', 'total_variation'],
                            device=args.device)
    vgg = VGG16(requires_grad=False).to(args.device)

    # transfer learning set-up and model parameters loading
    ckpt_model_path = os.path.join(args.checkpoint_dir, args.ckpt_filename)
    if args.transfer_learning:
        checkpoint = torch.load(ckpt_model_path, map_location=args.device)
        trainer.load_state_dict(checkpoint['model_state_dict'])
        transfer_learning_epoch = checkpoint['epoch']
    else:
        transfer_learning_epoch = 0
    trainer.to(args.device)

    # optimizer construction
    optimizer = torch.optim.Adam(trainer.parameters(), initial_lr)

    # style image importing, pre-processing and gram_style pre-calculating
    args.gram_styles = get_gram_styles(vgg)

    train(trainer, vgg, optimizer, transfer_learning_epoch,
          train_loader, val_loader)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-tag', '-t', required=True)
    # train_dataset_dir = "~/Documents/proj/ML/Deep Learning/cocodataset"
    # train_dataset_subdir = "val2017"
    parser.add_argument('-train_dataset_dir', required=True)
    parser.add_argument('-train_dataset_subdir', required=True)
    parser.add_argument('-val_dataset_dir', default=None)
    parser.add_argument('-val_dataset_subdir', default=None)
    parser.add_argument('-val_set_ratio', default=0.1, type=int)

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

    parser.add_argument('--gdrive', action='store_true')

    args = parser.parse_args()

    if args.ckpt_filename is None:
        args.ckpt_filename = f"ckpt_epoch_{args.num_epochs-1}_batch_id_{args.checkpoint_interval}.ckpt"

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device: {args.device}')

    # desired size of the output image
    args.imsize = 256 if torch.cuda.is_available() else 128  # use small size if no gpu
    print(f'imsize: {args.imsize}')

    main()
