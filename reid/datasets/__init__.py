from __future__ import absolute_import

from .market1501 import Market1501
from .msmt17 import MSMT17
from .cuhk03 import CUHK03
from .cuhk02 import CUHK02
from PIL import Image
from torch.utils.data import Dataset

__factory = {
    'market1501': Market1501, 'msmt17': MSMT17,
    'cuhk03': CUHK03, 'cuhk02': CUHK02, 
}

def names():
    set_names = sorted(__factory.keys())
    return set_names


def create(name, root, *args, **kwargs):
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](root, *args, **kwargs)


class BaseDataset(Dataset):
    def __init__(self, dataset, trans):
        self.dataset = dataset
        self.trans = trans

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        fname, label, cam = self.dataset[index]
        image = Image.open(fname).convert('RGB')
        return self.trans(image), fname, label, index, cam
