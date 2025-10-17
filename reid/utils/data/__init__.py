from __future__ import absolute_import

from .base_dataset import BaseDataset, BaseImageDataset
from .preprocessor import Preprocessor

class IterLoader:
    def __init__(self, loader, length=None, max_iter=400):
        self.loader = loader
        self.length = length
        self.iter = None
        self.max_iter = max_iter

    def __len__(self):
        if self.length is not None:
            return self.length
        return self.max_iter

    def new_epoch(self):
        self.iter = iter(self.loader)

    def next(self):
        try:
            return next(self.iter)
        except:
            self.iter = iter(self.loader)
            return next(self.iter)
