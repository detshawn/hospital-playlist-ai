from __future__ import print_function

from argparse import ArgumentParser

import torch
import os
import torchvision.transforms as transforms
import numpy as np

import matplotlib.pyplot as plt
import re
import cv2

from model.transformer_net import TransformerNet, VGG16
from utils import *


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device: {device}')

    # config
    style_image_path = args.style_image_path

    checkpoint_dir = args.checkpoint_dir
    ckpt_filename = args.ckpt_filename
    
    content_image_path = args.content_image_path
    content_video_path = args.content_video_path
    option = args.option

    output_dir = args.output_dir

    style_image_sample = Image.open(style_image_path, 'r')
    display(style_image_sample)

    # inference data importing
    if option == "image":
        content_image = load_image(content_image_path)
        content_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])
        content_image = content_transform(content_image)
        print(f'content_image shape: {content_image.shape}')
        content_image = content_image.unsqueeze(0).to(device)

        with torch.no_grad():
            style_model = TransformerNet()

            ckpt_model_path = os.path.join(checkpoint_dir, ckpt_filename)
            checkpoint = torch.load(ckpt_model_path, map_location=device)

            # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
            for k in list(checkpoint.keys()):
                if re.search(r'in\d+\.running_(mean|var)$', k):
                    del checkpoint[k]

            style_model.load_state_dict(checkpoint['model_state_dict'])
            style_model.to(device)

            output = style_model(content_image).cpu()

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        output_filename = content_image_path.split("/")[-1].split(".")[0] + "_result.png"
        save_image(output_dir + "/" + output_filename, output[0])

    elif option == "video":
        # Video Capture Sanity Check
        cap = cv2.VideoCapture(content_video_path)

        while (cap.isOpened()):
            ret, frame = cap.read()
            frame = frame[:, :, ::-1]

            print(frame.shape)

            plt.imshow(frame)

            break
        cap.release()
        cv2.destroyAllWindows()

        with torch.no_grad():
            style_model = TransformerNet()

            ckpt_model_path = os.path.join(checkpoint_dir, ckpt_filename) 
            checkpoint = torch.load(ckpt_model_path, map_location=device)

            # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
            for k in list(checkpoint.keys()):
                if re.search(r'in\d+\.running_(mean|var)$', k):
                    del checkpoint[k]

            style_model.load_state_dict(checkpoint['model_state_dict'])
            style_model.to(device)

            cap = cv2.VideoCapture(content_video_path)

            frame_cnt = 0

            fourcc = cv2.VideoWriter_fourcc(*'XVID')

            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            output_filename = content_video_path.split("/")[-1].split(".")[0] + "_result.avi"
            out = cv2.VideoWriter(output_dir + '/' + output_filename, fourcc, 60.0,
                                  (1920, 1080)) 

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

    else:
        print(f' Error: The option \'{option}\' is not a valid key!')
        exit(1)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('-option', default="image")
    parser.add_argument('-content_image_path', default="./dataset/inference/hospital-playlist-test.jpg")
    # parser.add_argument('-content_image_path', required=True)
    parser.add_argument('-content_video_path', default="./dataset/inference/test.mp4")
    parser.add_argument('-output_dir', '-o', default="./output")    

    parser.add_argument('-style_image_path', default="./dataset/style/andy_dixon_summering.png")

    parser.add_argument('-checkpoint_dir', default='./ckpts/')
    parser.add_argument('-ckpt_filename', default=None)

    args = parser.parse_args()

    if args.ckpt_filename is None:
        args.ckpt_filename = f"ckpt_epoch_63_batch_id_500.pth"

    main()
