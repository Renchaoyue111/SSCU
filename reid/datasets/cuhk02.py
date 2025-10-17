from __future__ import division, print_function, absolute_import
import glob
import os.path as osp

class CUHK02(object):
    """CUHK02.

    Reference:
        Li and Wang. Locally Aligned Feature Transforms across Views. CVPR 2013.

    URL: `<http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html>`_
    
    Dataset statistics:
        - 5 camera view pairs each with two cameras
        - 971, 306, 107, 193 and 239 identities from P1 - P5
        - totally 1,816 identities
        - image format is png

    Protocol: Use P1 - P4 for training and P5 for evaluation.

    Note: CUHK01 and CUHK02 overlap.
    """
    cam_pairs = ['P1', 'P2', 'P3', 'P4', 'P5']
    test_cam_pair = 'P5'
    
    @property
    def images_dir(self):
        return osp.join(self.root, 'images')
    
    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, 'Dataset')

        train, query, gallery = self.get_data_list()
        self.train, self.val, self.trainval = train, [], []
        self.query, self.gallery = query, gallery
        q_pids = len(set([val[1] for val in query]))
        g_pids = len(set([val[1] for val in gallery]))
        
        self.num_train_pids = len(set([val[1] for val in train]))
        self.train_dir = osp.join(self.root, 'images')

        print(self.__class__.__name__, "dataset loaded")
        print("  subset   | # ids | # images")
        print("  ---------------------------")
        print("  train    | {:5d} | {:8d}"
                .format(self.num_train_pids, len(self.train)))
        print("  trainval | {:5d} | {:8d}"
                .format(self.num_train_pids, len(self.trainval)))
        print("  query    | {:5d} | {:8d}"
                .format(q_pids, len(self.query)))
        print("  gallery  | {:5d} | {:8d}"
                .format(g_pids, len(self.gallery)))

    def get_data_list(self):
        num_train_pids, camid = 0, 0
        train, query, gallery = [], [], []

        for cam_pair in self.cam_pairs:
            cam_pair_dir = osp.join(self.dataset_dir, cam_pair)

            cam1_dir = osp.join(cam_pair_dir, 'cam1')
            cam2_dir = osp.join(cam_pair_dir, 'cam2')

            impaths1 = glob.glob(osp.join(cam1_dir, '*.png'))
            impaths2 = glob.glob(osp.join(cam2_dir, '*.png'))

            if cam_pair == self.test_cam_pair:
                # add images to query
                for impath in impaths1:
                    pid = osp.basename(impath).split('_')[0]
                    pid = int(pid)
                    query.append((impath, pid, camid))
                camid += 1

                # add images to gallery
                for impath in impaths2:
                    pid = osp.basename(impath).split('_')[0]
                    pid = int(pid)
                    gallery.append((impath, pid, camid))
                camid += 1

            else:
                pids1 = [
                    osp.basename(impath).split('_')[0] for impath in impaths1
                ]
                pids2 = [
                    osp.basename(impath).split('_')[0] for impath in impaths2
                ]
                pids = set(pids1 + pids2)
                pid2label = {
                    pid: label + num_train_pids
                    for label, pid in enumerate(pids)
                }

                # add images to train from cam1
                for impath in impaths1:
                    pid = osp.basename(impath).split('_')[0]
                    pid = pid2label[pid]
                    train.append((impath, pid, camid))
                camid += 1

                # add images to train from cam2
                for impath in impaths2:
                    pid = osp.basename(impath).split('_')[0]
                    pid = pid2label[pid]
                    train.append((impath, pid, camid))
                camid += 1
                num_train_pids += len(pids)

        return train, query, gallery

    def split_clients(self, splits, sp_pid=0):
        sp_count = self.num_train_pids // splits
        start = sp_count*sp_pid
        rear = min(start+sp_count, self.num_train_pids) if sp_pid!=splits-1 else self.num_train_pids
        self.train = [(val[0], val[1]-start, val[2]) for val in self.train if start <= val[1] < rear]
        self.num_train_pids = len(set([val[1] for val in self.train]))
        self.num_train_imgs = len(self.train)
        