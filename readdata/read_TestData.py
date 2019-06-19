from torchvision import transforms, datasets
import os
import torch
import math
from PIL import Image
import scipy.io as scio
import logging

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def TestImageNetData(args):
    image_datasets = {'test': TestImageNetSet(
        os.path.join(args.data_dir, "future_img", args.pig))}

    # wrap your data and label into Tensor
    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                 batch_size=args.batch_size,
                                                 shuffle=True,
                                                 num_workers=args.num_workers) for x in ['test']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['test']}
    return dataloders, dataset_sizes


class TestImageNetSet(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        self.img_path = root_dir
        print(root_dir)
        self.data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.imgs = []
        self.imgs.append(root_dir)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        img = Image.open(self.img_path).convert('RGB')
        if self.data_transforms is not None:
            try:
                img = self.data_transforms(img)
            except:
                print("Cannot transform image: {}".format(self.img_path[item]))
        return img, 0
