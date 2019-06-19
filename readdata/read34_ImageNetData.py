from torchvision import transforms, datasets
import os
import torch
import math
from PIL import Image
import scipy.io as scio
import logging

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def ImageNetData(args):
    # data_transform, pay attention that the input of Normalize() is Tensor and the input of RandomResizedCrop() or RandomHorizontalFlip() is PIL Image
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(168),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(168),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    image_datasets = {}
    # image_datasets['train'] = datasets.ImageFolder(os.path.join(args.data_dir, 'ILSVRC2012_img_train'), data_transforms['train'])
    # image_datasets['train'] = ImageNetTrainDataSet(
    #     os.path.join(args.data_dir, 'ILSVRC2012_img_train'),
    #     os.path.join(args.data_dir, 'ILSVRC2012_devkit_t12', 'data',
    #                  'meta.mat'),
    #     data_transforms['train'])

    image_datasets['train'] = ImageNetDataSet(
        os.path.join(args.data_dir, 'Images'),
        data_transforms['train'],
        True)

    image_datasets['val'] = ImageNetDataSet(
        os.path.join(args.data_dir, 'Images'),
        data_transforms['val'],
        False)

    # wrap your data and label into Tensor
    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                 batch_size=args.batch_size,
                                                 shuffle=True,
                                                 num_workers=args.num_workers) for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    return dataloders, dataset_sizes


class ImageNetDataSet(torch.utils.data.Dataset):
    def __init__(self, root_dir, data_transforms, is_train):
        self.is_train = is_train
        self.img_path = os.listdir(root_dir)
        self.data_transforms = data_transforms
        self.root_dir = root_dir
        self.label_map = self._make_label()
        self.imgs = self._make_dataset()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        data = self.imgs[item]
        img = Image.open(data).convert('RGB')
        if self.data_transforms is not None:
            try:
                img = self.data_transforms(img)
            except:
                print("Cannot transform image: {}".format(self.img_path[item]))
        str = os.path.abspath(os.path.dirname(data))
        start_index = str.rfind('/') + 1
        label = self.label_map.index(str[start_index:])
        return img, label

    def _make_label(self):
        label = []
        for img_folder in self.img_path:
            if not label.count(img_folder):
                label.append(img_folder)
        return label

    def _make_dataset(self):
        images = []
        for img_folder in self.img_path:
            for root, dirs, files in sorted(os.walk(os.path.join(self.root_dir, img_folder))):
                filterNum = round(len(files) * 0.9)
                if self.is_train:
                    for index in range(len(files)):
                        if index >= filterNum:
                            break
                        if self._is_image_file(files[index]):
                            path = os.path.join(root, files[index])
                            images.append(path)
                else:
                    last_index = len(files) - filterNum
                    for index in range(len(files)):
                        if index >= last_index:
                            break
                        if self._is_image_file(files[-(index + 1)]):
                            path = os.path.join(root, files[index])
                            images.append(path)
                        # for target in sorted(os.listdir(dir)):
        #     d = os.path.join(dir, target)
        #     if not os.path.isdir(d):
        #         continue
        #
        #     for root, _, fnames in sorted(os.walk(d)):
        #         for fname in sorted(fnames):
        #             if self._is_image_file(fname):
        #                 path = os.path.join(root, fname)

        return images

    def _is_image_file(self, filename):
        """Checks if a file is an image.

        Args:
            filename (string): path to a file

        Returns:
            bool: True if the filename ends with a known image extension
        """
        filename_lower = filename.lower()
        return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)

# class ImageNetValDataSet(torch.utils.data.Dataset):
#     def __init__(self, img_path, img_label, data_transforms):
#         self.data_transforms = data_transforms
#         img_names = os.listdir(img_path)
#         img_names.sort()
#         self.img_path = [os.path.join(img_path, img_name) for img_name in img_names]
#         with open(img_label, "r") as input_file:
#             lines = input_file.readlines()
#             self.img_label = [(int(line) - 1) for line in lines]
#
#     def __len__(self):
#         return len(self.img_path)
#
#     def __getitem__(self, item):
#         img = Image.open(self.img_path[item]).convert('RGB')
#         label = self.img_label[item]
#         if self.data_transforms is not None:
#             try:
#                 img = self.data_transforms(img)
#             except:
#                 print("Cannot transform image: {}".format(self.img_path[item]))
#         return img, label
