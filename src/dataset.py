import os
from PIL import Image

import glob
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class CustomDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        class_dirs = sorted(filter(os.path.isdir, glob.glob(f'{dataset_path}/*')))
        assert len(class_dirs) > 0, "No class directory exists."

        img_path_per_class = {i: sorted(glob.glob(f'{class_dirs[i]}/*.jpg') +
                                        glob.glob(f'{class_dirs[i]}/*.JPG') +
                                        glob.glob(f'{class_dirs[i]}/*.jpeg') +
                                        glob.glob(f'{class_dirs[i]}/*.JPEG') +
                                        glob.glob(f'{class_dirs[i]}/*.bmp') +
                                        glob.glob(f'{class_dirs[i]}/*.BMP') +
                                        glob.glob(f'{class_dirs[i]}/*.png') +
                                        glob.glob(f'{class_dirs[i]}/*.PNG'))
                              for i in range(len(class_dirs))}
        self.class_names = [os.path.basename(class_dir) for class_dir in class_dirs]
        self.img_path_per_class = [(img_path_per_class[i][j], i)
                                   for i in img_path_per_class.keys()
                                   for j in range(len(img_path_per_class[i]))]
        self.length = len(self.img_path_per_class)
        assert self.length > 0, "No image exists for all class directories."

        self.transform = transform if transform else transforms.ToTensor()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img_path = self.img_path_per_class[idx][0]
        x = self.transform(Image.open(img_path).convert('RGB'))
        y = self.img_path_per_class[idx][1]
        return img_path, x, y
