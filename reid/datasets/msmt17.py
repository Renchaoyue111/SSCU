from __future__ import print_function, absolute_import
import os.path as osp
import re

from ..utils.osutils import mkdir_if_missing


def _pluck_msmt(list_file, subdir, pattern=re.compile(r'([-\d]+)_c([-\d]+)_([-\d]+)'), root=None):
    with open(list_file, 'r') as f:
        lines = f.readlines()
    ret = []
    pids = [] 
    for line in lines:
        # line = line.strip()
        img_path, pid = line.split(' ')
        pid = int(pid)
        if pid not in pids:
            pids.append(pid)
        camid = int(img_path.split('_')[2]) - 1  
        img_path = osp.join(root, subdir, img_path)
        ret.append((img_path, pid, camid))

    return ret, pids


class Dataset_MSMT(object):
    def __init__(self, root):
        self.root = root
        self.train, self.val, self.trainval = [], [], []
        self.query, self.gallery = [], []
        self.num_train_ids, self.num_val_ids, self.num_trainval_ids = 0, 0, 0

    @property
    def images_dir(self):
        return osp.join(self.root, 'MSMT17_V1')

    def load(self, verbose=True):
        exdir = osp.join(self.root, 'MSMT17_V1')
        self.train, train_pids = _pluck_msmt(osp.join(exdir, 'list_train.txt'), 'train', root=exdir)
        self.val, val_pids = _pluck_msmt(osp.join(exdir, 'list_val.txt'), 'train', root=exdir)
        self.train = self.train + self.val
        self.query, query_pids = _pluck_msmt(osp.join(exdir, 'list_query.txt'), 'test', root=exdir)
        self.num_query_pids = len(list(set(query_pids)))
        self.gallery, gallery_pids = _pluck_msmt(osp.join(exdir, 'list_gallery.txt'), 'test', root=exdir)
        self.num_train_pids = len(list(set(train_pids).union(set(val_pids))))
        self.train_dir = osp.join(exdir, 'train')
        # self.train_dir = osp.join(exdir, 'bounding_box_train', 'images')
        if verbose:
            print(self.__class__.__name__, "dataset loaded")
            print("  subset   | # ids | # images")
            print("  ---------------------------")
            print("  train    | {:5d} | {:8d}"
                  .format(self.num_train_pids, len(self.train)))
            print("  query    | {:5d} | {:8d}"
                  .format(len(query_pids), len(self.query)))
            print("  gallery  | {:5d} | {:8d}"
                  .format(len(gallery_pids), len(self.gallery)))
            
    def split_clients(self, splits, sp_pid=0):
        print("split clients: MSMT17")
        sp_count = self.num_train_pids // splits
        start = sp_count*sp_pid
        rear = min(start+sp_count, self.num_train_pids) if sp_pid!=splits-1 else self.num_train_pids
        self.train = [(val[0], val[1]-start, val[2]) for val in self.train if start <= val[1] < rear]
        self.num_train_pids = len(set([val[1] for val in self.train]))


class MSMT17(Dataset_MSMT):

    def __init__(self, root, split_id=0, download=True):
        super(MSMT17, self).__init__(root)

        if download:
            self.download()

        self.load()

    def download(self):

        raw_dir = osp.join(self.root)
        mkdir_if_missing(raw_dir)

        # Download the raw zip file
        fpath = osp.join(raw_dir, 'MSMT17_V1')
        if osp.isdir(fpath):
            print("Using downloaded file: " + fpath)
        else:
            raise RuntimeError("Please download the dataset manually to {}".format(fpath))
