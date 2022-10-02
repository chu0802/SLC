from pathlib import Path
import torch
from torch.utils.data import DataLoader, Sampler, BatchSampler, RandomSampler

from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from randaugment import RandAugmentMC
import numpy as np
import random
from PIL import Image

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

TRANSFORM = {
    'train': transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,std=std)
    ]),
    'test': transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,std=std)
    ]),
    'strong': transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224),
        RandAugmentMC(n=2, m=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,std=std)
    ])
}

def get_all_loaders(args):
    root, s_name, t_name = Path(args.dataset['path']), args.dataset['domains'][args.source], args.dataset['domains'][args.target]

    s_list_file = s_name + args.dataset['list']['all']
    s_train_set = ImageList(root, s_list_file, transform=TRANSFORM['train'])
    s_train_loader = get_loader(s_train_set, args.seed, args.bsize, train=True)

    s_test_set = ImageList(root, s_list_file, transform=TRANSFORM['test'])
    s_test_loader = get_loader(s_test_set, args.seed, args.bsize*2, train=False)

    if args.mode == 'uda':
        t_list_file = t_name + args.dataset['list']['all']

        t_unlabeled_train_set = ImageList(root, t_list_file, transform=TRANSFORM['train'])
        t_unlabeled_train_loader = get_loader(t_unlabeled_train_set, args.seed, args.bsize, train=True)
        
        t_unlabeled_test_set = ImageList(root, t_list_file, transform=TRANSFORM['test'])
        t_unlabeled_test_loader = get_loader(_unlabeled_test_set, args.seed, args.bsize*2, train=False)

        return s_train_loader, s_test_loader, t_unlabeled_train_loader, t_unlabeled_test_loader
    elif args.mode == 'ssda':
        t_train_list_file = t_name + args.dataset['list']['3shot_train']
        t_test_list_file = t_name + args.dataset['list']['3shot_test']

        t_labeled_train_set = ImageList(root, t_train_list_file, transform=TRANSFORM['train'])
        t_labeled_train_loader = get_loader(t_labeled_train_set, args.seed, args.bsize, train=True)

        t_labeled_test_set = ImageList(root, t_train_list_file, transform=TRANSFORM['test'])
        t_labeled_test_loader = get_loader(t_labeled_test_set, args.seed, args.bsize, train=False)

        t_unlabeled_train_set = ImageList(root, t_test_list_file, transform=TRANSFORM['train'], strong_transform=TRANSFORM['strong'] if 'CDAC' in args.method else None)
        t_unlabeled_train_loader = get_loader(t_unlabeled_train_set, args.seed, args.bsize, train=True)
        
        t_unlabeled_test_set = ImageList(root, t_test_list_file, transform=TRANSFORM['test'])
        t_unlabeled_test_loader = get_loader(t_unlabeled_test_set, args.seed, args.bsize*2, train=False)

        return s_train_loader, s_test_loader, t_labeled_train_loader, t_labeled_test_loader, t_unlabeled_train_loader, t_unlabeled_test_loader

def pil_loader(path: str):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_generator(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g

def get_loader(dset, seed=None, bsize=24, train=True):
    g = get_generator(seed)
    if train:
        dloader = InfiniteDataLoader(dset,
            batch_size = bsize,
            worker_init_fn=seed_worker, generator=g,
            drop_last=True, num_workers=8)
    else:
        dloader = DataLoader(dset, 
            batch_size = bsize,
            worker_init_fn=seed_worker, generator=g, 
            shuffle=False, drop_last=False, num_workers=8, pin_memory=True)
    return dloader

class ImageList(Dataset):
    def __init__(self, root, list_file, transform, strong_transform=None):
        with (root / list_file).open('r') as f:
            paths = [p[:-1].split() for p in f.readlines()]
        self.imgs = [(root / p, int(l)) for p, l in paths]
        self.transform = transform
        self.strong_transform = strong_transform
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, idx):
        path, label = self.imgs[idx]
        img = pil_loader(path)
        if self.strong_transform:
            img1 = self.strong_transform(img)
            img2 = self.strong_transform(img)
            return self.transform(img), label, img1, img2, idx
        return self.transform(img), label, idx

class _InfiniteSampler(Sampler):
    """Wraps another Sampler to yield an infinite stream."""
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch

class InfiniteDataLoader:
    def __init__(self, dataset, batch_size, 
                 worker_init_fn=None, 
                 generator=None, drop_last=True, 
                 num_workers=4):
        
        sampler = RandomSampler(dataset, replacement=False, generator=generator)
        batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=drop_last)

        self._infinite_iterator = iter(DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler),
            worker_init_fn=worker_init_fn
        ))

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        return 0