import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from PIL import Image
import numpy as np
import math

# Wrapper to create Multi-View datasets starting from 1 view and augmentation
class AugmentedDataset(Dataset):
    def __init__(self, dataset, augmentation, transform=None, target_transform=None, apply_same=False):
        assert hasattr(augmentation, '__call__')

        self.dataset = dataset
        self.augmentation = augmentation
        self.transform = transform
        self.target_transform = target_transform
        self.to_tensor = transforms.ToTensor()
        self.apply_same = apply_same

    def __getitem__(self, index):
        x, y = self.dataset[index]

        v_1 = self.augmentation(x)
      
        if self.apply_same:
            v_2 = v_1
        else:
            v_2 = self.augmentation(x)

        if self.transform is not None:
            v_1 = self.transform(v_1)
            v_2 = self.transform(v_2)

        if self.target_transform is not None:
            y = self.target_transform(y)

        return v_1, v_2, y

    def __len__(self):
        return len(self.dataset)


    
# Transform which randomly corrupts pixels with a given probabiliy
class PixelCorruption(object):
    MODALITIES = ['flip', 'drop']

    def __init__(self, p, min=0, max=1, mode='drop'):
        super(PixelCorruption, self).__init__()

        assert mode in self.MODALITIES

        self.p = p
        self.min = min
        self.max = max
        self.mode = mode

    def __call__(self, im):
        if isinstance(im, Image.Image) or isinstance(im, np.ndarray):
            im = F.to_tensor(im)

        if self.p < 1.0:
            mask = torch.bernoulli(torch.zeros(im.size(1),im.size(2)) + 1. - self.p).byte()
        else:
            mask = torch.zeros(im.size(1),im.size(2)).byte()

        if len(im.size())>2:
            mask = mask.unsqueeze(0).repeat(im.size(0),1,1)

        if self.mode == 'flip':
            im[mask] = self.max - im[mask]
        elif self.mode == 'drop':
            im[mask] = self.min

        return im

# Schedulers for beta
class Scheduler:
    def __call__(self, **kwargs):
        raise NotImplemented()

class LinearScheduler(Scheduler):
    def __init__(self, start_value, end_value, n_iterations, start_iteration=0):
        self.start_value = start_value
        self.end_value = end_value
        self.n_iterations = n_iterations
        self.start_iteration = start_iteration
        self.m = (end_value - start_value) / n_iterations

    def __call__(self, iteration):
        if iteration > self.start_iteration + self.n_iterations:
            return self.end_value
        elif iteration <= self.start_iteration:
            return self.start_value
        else:
            return (iteration - self.start_iteration) * self.m + self.start_value

class ExponentialScheduler(LinearScheduler):
    def __init__(self, start_value, end_value, n_iterations, start_iteration=0, base=10):
        self.base = base

        super(ExponentialScheduler, self).__init__(start_value=math.log(start_value, base),
                                                   end_value=math.log(end_value, base),
                                                   n_iterations=n_iterations,
                                                   start_iteration=start_iteration)

    def __call__(self, iteration):
        linear_value = super(ExponentialScheduler, self).__call__(iteration)
        return self.base ** linear_value