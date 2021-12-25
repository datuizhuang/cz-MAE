from torchvision import transforms
import torchvision.transforms.functional as F
from PIL import ImageFilter
import random
import math


class HideAndSeek(object):
    def __init__(self, hide_prob=0.2, mean=[0.485, 0.456, 0.406]):
        self.hide_prob = hide_prob  # hiding probability
        self.mean = mean

    def __call__(self, img):
        # get width and height of the image
        s = img.shape
        wd = s[1]
        ht = s[2]

        # possible grid size, 0 means no hiding
        grid_sizes = [16, 32, 44, 56]

        # randomly choose one grid size
        grid_size = grid_sizes[random.randint(0, len(grid_sizes) - 1)]

        # hide the patches
        if grid_size != 0:
            for x in range(0, wd, grid_size):
                for y in range(0, ht, grid_size):
                    x_end = min(wd, x + grid_size)
                    y_end = min(ht, y + grid_size)
                    if random.random() <= self.hide_prob:
                        if img.size()[0] == 3:
                            img[0, x:x_end, y:y_end] = self.mean[0]
                            img[1, x:x_end, y:y_end] = self.mean[1]
                            img[2, x:x_end, y:y_end] = self.mean[2]
                        else:
                            img[0, x:x_end, y:y_end] = self.mean[0]

        return img


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img


class RandomPad(transforms.Pad):
    def __call__(self, img):
        x, y = img.size
        x = int((random.random() * 0.2) * x)
        y = int((random.random() * 0.2) * y)
        self.padding = (x, y, x, y)
        self.fill = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.padding_mode = random.choice(['constant', 'constant', 'constant', 'edge', 'reflect', 'symmetric'])
        return F.pad(img, self.padding, self.fill, self.padding_mode)


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def getTrans(args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet/v1
    args.image_size = tuple(args.image_size)
    if args.aug == 'NULL':
        train_transform = transforms.Compose([transforms.RandomResizedCrop(args.image_size, scale=(args.crop, 1.)),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              normalize])
    elif args.aug == 'CJ':
        train_transform = transforms.Compose([transforms.RandomResizedCrop(args.image_size, scale=(args.crop, 1.)),
                                              transforms.RandomGrayscale(p=0.2),
                                              transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                                              transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406]),
                                              normalize])
    elif args.aug == 'v1':
        train_transform = transforms.Compose([transforms.RandomApply([transforms.RandomAffine(10, shear=10, resample=3)], p=0.5),
                                              transforms.RandomApply([RandomPad(20)], p=0.75),
                                              transforms.RandomResizedCrop(args.image_size, scale=(0.3, 1.), ratio=(0.5, 2.0),
                                                                           interpolation=3),
                                              transforms.RandomGrayscale(p=0.2),
                                              transforms.RandomApply([transforms.ColorJitter(0.4, 0.3, 0.3, 0.05)], p=0.75),
                                              transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.35),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406]),
                                              normalize])

    else:
        raise NotImplementedError('augmentation not supported: {}'.format(args.aug))

    return train_transform
