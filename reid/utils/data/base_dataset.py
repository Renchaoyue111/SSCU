# encoding: utf-8
import numpy as np
import os


class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        if data is None:
            return 0, 0, 0
        pids, cams = [], []
        for _, pid, camid in data:
            pids += [pid]
            cams += [camid]
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        return num_pids, num_imgs, num_cams

    def print_dataset_statistics(self):
        raise NotImplementedError

    @property
    def images_dir(self):
        return None


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams = self.get_imagedata_info(gallery)
        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")

    def merge_dataset(self, image_sets):
        self.train, self.query, self.gallery, offset, offset_cam = [], [], [], 0, 0
        for name_set in image_sets:
            temp_set, image_dir = name_set.train, name_set.train_dir
            pid_count = len(set([val[1] for val in temp_set]))
            cam_count = len(set([val[2] for val in temp_set]))
            if temp_set[0][0].count('/') == 0:
                temp_set = [(os.path.join(image_dir, val[0]), val[1] + offset, val[2] + offset_cam) for val in temp_set]
            else:
                temp_set = [(val[0], val[1] + offset, val[2] + offset_cam) for val in temp_set]
            offset += pid_count
            offset_cam += cam_count
            self.train += temp_set
        self.print_dataset_statistics(self.train, self.query, self.gallery)
        self.num_classes = offset
