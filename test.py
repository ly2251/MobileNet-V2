from __future__ import print_function, division

import torch
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable
import os
from models.MobileNetV2 import *
import argparse
from readdata.read_TestData import TestImageNetData

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch implementation of SENet")
    parser.add_argument('--data-dir', type=str,
                        default=os.path.join(os.path.split(os.path.realpath(__file__))[0], 'DogImage'))
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-class', type=int, default=1000)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.045)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--gpus', type=str, default=0)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--save-epoch-freq', type=int, default=1)
    parser.add_argument('--save-path', type=str, default="output")
    parser.add_argument('--pig', type=str, default="")
    parser.add_argument('--resume', type=str, default="", help="For training from one checkpoint")
    parser.add_argument('--start-epoch', type=int, default=0, help="Corresponding to the epoch of resume ")
    args = parser.parse_args()

    # use gpu or not
    use_gpu = torch.cuda.is_available()
    print("use_gpu:{}".format(use_gpu))

    # get model
    model = mobilenetv2_19(num_classes=args.num_class)

    print(("=> loading checkpoint '{}'".format('epoch_99.pth.tar')))
    checkpoint = torch.load('output/epoch_99.pth.tar')
    base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.state_dict().items())}
    model.load_state_dict(base_dict)

    labelMap = []
    labelPath = os.path.join(args.data_dir, 'Images')
    img_path = os.listdir(labelPath)
    for img_folder in img_path:
        if not labelMap.count(img_folder):
            labelMap.append(img_folder)

    if use_gpu:
        model = model.cuda()
        model = torch.nn.DataParallel(model, device_ids=[int(i) for i in args.gpus.strip().split(',')])

    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # imagePath = os.path.join(args.data_dir, 'Samoyed.jpg')
    # image = Image.open(imagePath).convert('RGB')
    # imgblob = data_transforms(image)

    dataloders, dataset_sizes = TestImageNetData(args)
    model.train(False)
    for i, (inputs, labels) in enumerate(dataloders['test']):
        if use_gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        preds = preds.int()
        num = preds[0]
        print(labelMap[num])
        print("\n------------------------------")

