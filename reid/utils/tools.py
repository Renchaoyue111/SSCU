import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns
import pylab as pl
from .data import transforms as T
from .data.preprocessor import Preprocessor
from torch.utils.data import DataLoader
import torch
from reid import datasets
import os.path as osp
from .data import IterLoader, Preprocessor
from .data.sampler import RandomMultipleGallerySampler
import matplotlib as mpl
from torchvision.models.inception import inception_v3
import torch.nn.functional as F
from scipy.stats import entropy
mpl.use("Agg")
sns.set()


def inception_score(imgs, cuda=True, resize=False):
    N = len(imgs)
    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor
    # Load inception model
    inception_model = inception_v3(
        pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    up = torch.nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)

    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    imgs = imgs.type(dtype)
    preds = get_pred(imgs)
    py = np.mean(preds, axis=0)
    scores = []
    for i in range(preds.shape[0]):
        pyx = preds[i, :]
        scores.append(entropy(pyx, py))

    return np.exp(np.mean(scores))


class ScaffoldOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(ScaffoldOptimizer, self).__init__(params, defaults)

    def step(self, server_controls, client_controls, global_epoch, closure=None):
        # update params based on prev grads
        for counter, group in enumerate(self.param_groups):
            if global_epoch == 0:
                # vanilla train
                for p, c, ci in zip(group['params'], server_controls.values(), client_controls.values()):
                    if p.grad is None:
                        continue
                    p.data = p.data - p.grad.data * group['lr']  # vanilla SGD
            else:
                if counter == 0:
                    # non fc
                    for p, c, ci in zip(group['params'], server_controls.values(), client_controls.values()):
                        if p.grad is None:
                            continue
                        dp = p.grad.data + c.data - ci.data  # guidence on grad, server - client
                        p.data = p.data - dp.data * group['lr']  # vanilla SGD
                else:
                    # fc, vanilla train
                    for p in group['params']:
                        if p.grad is None:
                            continue
                        p.data = p.data - p.grad.data * group['lr']


def freeze_model(cur_model):
    cur_model.eval()
    for param in cur_model.parameters():
        param.requires_grad = False
    return cur_model


def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(), normalizer
    ])
    root_dir = dataset.images_dir if dataset.__class__.__name__ == 'CUHK03' else None

    if isinstance(dataset.query[0], list):
        testset = dataset.query + dataset.gallery
    elif testset is None:
        testset = list(set(dataset.query) | set(dataset.gallery))
    test_loader = DataLoader(
        Preprocessor(testset, root=root_dir, transform=test_transformer),
        batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)
    return test_loader


def get_train_loaders(dataset_lists, args, is_shuffle=False):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    transformer = T.Compose([
        T.Resize((args.height, args.width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10), T.RandomCrop((args.height, args.width)),
        T.ToTensor(), normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])
    train_loaders = []
    for dataset in dataset_lists:
        if is_shuffle:
            temp_loader = DataLoader(
                Preprocessor(dataset.train, root=None, transform=transformer),
                batch_size=args.batch_size, num_workers=args.workers, shuffle=True, pin_memory=True
            )
        else:
            temp_loader = IterLoader(DataLoader(
                Preprocessor(dataset.train, transform=transformer, root=None),
                batch_size=args.batch_size, shuffle=False, drop_last=True,
                sampler=RandomMultipleGallerySampler(
                    dataset.train, args.num_instances),
                pin_memory=True, num_workers=args.workers
            ), length=None)
        train_loaders.append(temp_loader)

    return train_loaders


def get_entropy(p_softmax):
    mask = p_softmax.ge(1e-6)
    mask_out = torch.masked_select(p_softmax, mask)
    entropy = -(torch.sum(mask_out * torch.log(mask_out)))
    return (entropy / float(p_softmax.size(0)))


def get_auth_loss(ent_aug_global, ent_ori_global, ent_aug_local):
    # HG(x'), HG(x), HL(x'); should be HG(x) < HG(x') < HL(x')
    ranking_loss = torch.nn.SoftMarginLoss()
    y = torch.ones_like(ent_aug_global)
    # HG(x) < HG(x'), HG(x') < HL(x')
    return ranking_loss(ent_aug_global - ent_ori_global, y) +\
        ranking_loss(ent_aug_local - ent_aug_global, y)


def get_data(args, set_names=None, sub_split=1):
    data_dir = args.data_dir
    if set_names is not None:
        dataset = []
        for name in set_names:
            root = osp.join(data_dir, name)
            if sub_split==1:
                dataset.append(datasets.create(name, root))
            else:
                for idx in range(sub_split):
                    cur_set = datasets.create(name, root)
                    cur_set.split_clients(sub_split, idx)
                    dataset.append(cur_set)
    else:
        name = args.test_dataset
        root = osp.join(data_dir, name)
        dataset = datasets.create(name, root)
    return dataset

def get_aug_data(args):
    set_names = 'unreal_v1.1,unreal_v2.1,unreal_v3.1,unreal_v4.1,unreal_v1.2,unreal_v2.2,unreal_v3.2,unreal_v4.2,unreal_v1.3,unreal_v2.3,unreal_v3.3,unreal_v4.3'
    return datasets.create(
        name='unreal', root=args.data_dir,
        dataset=set_names.split(','), data=args.data_dir
    )

def plotTSNE(features, domains, save_path, epoch):
    func = TSNE()

    def map_label(val):
        if val == 0:
            return 'C2'
        elif val == 1:
            return 'Novel'
        elif val == 2:
            return 'C3'
        else:
            return 'MS'

    embFeat = func.fit_transform(features)
    embFeat = pd.DataFrame(embFeat, columns=["x", "y"])
    embFeat["Domain"] = pd.Series(domains).apply(map_label)

    pl.figure()
    fig = sns.scatterplot(x=embFeat["x"], y=embFeat["y"],
                          hue=embFeat["Domain"], palette="tab10")
    fig.xaxis.set_ticklabels([])
    fig.yaxis.set_ticklabels([])
    fig.xaxis.set_label_text(None)
    fig.yaxis.set_label_text(None)
    pl.savefig(save_path)
    torch.save(embFeat, osp.join(osp.dirname(save_path), f"tsne_{epoch}.pth"))
    pl.close()


# instance norm mix, ref: https://github.com/amazon-science/crossnorm-selfnorm
def calc_ins_mean_std(x, eps=1e-5):
    """extract feature map statistics"""
    size = x.size()
    assert (len(size) == 4)
    N, C = size[:2]
    var = x.contiguous().view(N, C, -1).var(dim=2) + eps
    std = var.sqrt().view(N, C, 1, 1)
    mean = x.contiguous().view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return mean, std


def instance_norm_mix(content_feat, style_feat):
    """replace content statistics with style statistics"""
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_ins_mean_std(style_feat)
    content_mean, content_std = calc_ins_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def cn_rand_bbox(size, beta, bbx_thres):
    """sample a bounding box for cropping."""
    W = size[2]
    H = size[3]
    while True:
        ratio = np.random.beta(beta, beta)
        cut_rat = np.sqrt(ratio)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        ratio = float(bbx2 - bbx1) * (bby2 - bby1) / (W * H)
        if ratio > bbx_thres:
            break

    return bbx1, bby1, bbx2, bby2


def cn_op_2ins_space_chan(x, crop='neither', beta=1, bbx_thres=0.1, lam=None, chan=False):
    """2-instance crossnorm with cropping."""
    assert crop in ['neither', 'style', 'content', 'both']
    ins_idxs = torch.randperm(x.size()[0]).to(x.device)

    if crop in ['style', 'both']:
        bbx3, bby3, bbx4, bby4 = cn_rand_bbox(
            x.size(), beta=beta, bbx_thres=bbx_thres)
        x2 = x[ins_idxs, :, bbx3:bbx4, bby3:bby4]
    else:
        x2 = x[ins_idxs]

    if chan:
        chan_idxs = torch.randperm(x.size()[1]).to(x.device)
        x2 = x2[:, chan_idxs, :, :]

    if crop in ['content', 'both']:
        x_aug = torch.zeros_like(x)
        bbx1, bby1, bbx2, bby2 = cn_rand_bbox(
            x.size(), beta=beta, bbx_thres=bbx_thres)
        x_aug[:, :, bbx1:bbx2, bby1:bby2] = instance_norm_mix(content_feat=x[:, :, bbx1:bbx2, bby1:bby2],
                                                              style_feat=x2)
        mask = torch.ones_like(x, requires_grad=False)
        mask[:, :, bbx1:bbx2, bby1:bby2] = 0.
        x_aug = x * mask + x_aug
    else:
        x_aug = instance_norm_mix(content_feat=x, style_feat=x2)

    if lam is not None:
        x = x * lam + x_aug * (1-lam)
    else:
        x = x_aug
    return x
