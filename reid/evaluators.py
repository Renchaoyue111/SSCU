from __future__ import print_function, absolute_import
import time
from collections import OrderedDict
import torch

from .evaluation_metrics import cmc, mean_ap
from .utils.meters import AverageMeter
from .utils.rerank import re_ranking
from .utils import to_torch
import numpy as np
from sklearn.metrics import accuracy_score


def compute_accuracy(predictions, labels):
    if np.ndim(labels) == 2:
        y_true = np.argmax(labels, axis=-1)
    else:
        y_true = labels
    accuracy = accuracy_score(
        y_true=y_true, y_pred=np.argmax(predictions, axis=-1))
    return accuracy


def extract_cnn_feature(model, inputs):
    inputs = to_torch(inputs).cuda()
    outputs = model(inputs)  # pool5 for eva
    if isinstance(outputs, list) and len(outputs) > 1:
        outputs = outputs[-1][0]
    outputs = outputs.data.cpu()
    return outputs


def extract_features(model, data_loader, print_freq=50):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    with torch.no_grad():
        for i, (imgs, fnames, pids, _, _) in enumerate(data_loader):
            data_time.update(time.time() - end)

            outputs = extract_cnn_feature(model, imgs)
            for fname, output, pid in zip(fnames, outputs, pids):
                features[fname] = output
                labels[fname] = pid

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Extract Features: [{}/{}]\t'
                    'Time {:.3f} ({:.3f})\t'
                    'Data {:.3f} ({:.3f})\t'
                    .format(i + 1, len(data_loader),
                            batch_time.val, batch_time.avg,
                            data_time.val, data_time.avg))

    return features, labels


def pairwise_distance(features, query=None, gallery=None):
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
        dist_m = dist_m.expand(n, n) - 2 * torch.mm(x, x.t())
        return dist_m

    x = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0)
    y = torch.cat([features[f].unsqueeze(0) for f, _, _ in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_m.addmm_(1, -2, x, y.t())
    return dist_m, x.numpy(), y.numpy()


def evaluate_all(distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10), cmc_flag=False):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # Compute mean AP
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))

    if not cmc_flag:
        return mAP

    cmc_configs = {
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True), }
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}

    print('CMC Scores:')
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}'.format(k, cmc_scores['market1501'][k - 1]))
    return mAP, cmc_scores['market1501']


class Evaluator(object):
    def __init__(self, model=None):
        super(Evaluator, self).__init__()
        self.model = model

    def evaluate(self, data_loader, query, gallery, cmc_flag=False, rerank=False):
        features, _ = extract_features(self.model, data_loader)
        distmat, _, _ = pairwise_distance(features, query, gallery)
        results = evaluate_all(distmat, query=query, gallery=gallery,
                               cmc_flag=cmc_flag)
        return results

    def evaluate_pacs(self, test_loader, set_name):
        self.model.eval()
        y_pred, y_labels = [], []
        for (images, pids) in test_loader:
            images, pids = images.cuda(), pids.cuda()
            with torch.no_grad():
                y_pred.append(self.model(images))
                y_labels.append(pids)
        y_pred, y_labels = torch.cat(y_pred, 0).cpu().numpy(), torch.cat(y_labels).cpu().numpy()
        
        accuracy = compute_accuracy(predictions=y_pred, labels=y_labels)
        print(f'----------accuracy test on {set_name.upper()}----------: {100*accuracy:4.2f} %')
        return accuracy
    