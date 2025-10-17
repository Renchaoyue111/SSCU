from torch.utils.data import DataLoader
from .utils.data import IterLoader, Preprocessor
import torch
from .utils.data.sampler import RandomMultipleGallerySampler
from .utils.tools import get_entropy, get_auth_loss, ScaffoldOptimizer, cn_op_2ins_space_chan, freeze_model, inception_score
from .loss import TripletLoss
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
import os
import copy
from pytorch_msssim import ssim
import numpy as np


# trainers in user side
class DomainLocalUpdate(object):
    def __init__(self, args, dataset=None, trans=None, memory = None):
        self.args = args
        self.trans = trans
        self.memory = memory
        # only for non-qaconv algos
        if dataset is not None:
            if not isinstance(dataset, list):
                self.local_train = IterLoader(DataLoader(
                    Preprocessor(dataset.train, transform=trans, root=None),
                    batch_size=self.args.batch_size, shuffle=False, drop_last=True,
                    sampler=RandomMultipleGallerySampler(
                        dataset.train, args.num_instances),
                    pin_memory=False, num_workers=self.args.num_workers
                ), length=None)
                self.set_name = dataset.__class__.__name__
            else:
                self.local_train = [IterLoader(DataLoader(
                    Preprocessor(cur_set.train, transform=trans, root=None),
                    batch_size=self.args.batch_size, shuffle=False, drop_last=True,
                    sampler=RandomMultipleGallerySampler(
                        cur_set.train, args.num_instances),
                    pin_memory=False, num_workers=self.args.num_workers
                ), length=None) for cur_set in dataset]
                pid_list = [user.num_train_pids for user in dataset]
                self.padding = np.cumsum([0, ]+pid_list)
        self.max_iter = args.max_iter
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tri_loss = TripletLoss(margin=0.5, is_avg=True)
        self.ce_loss = torch.nn.CrossEntropyLoss(reduce='mean')
        self.dataset = dataset

    def handle_set(self, dataset):
        cur_loader = IterLoader(DataLoader(
            Preprocessor(dataset.train, transform=self.trans, root=None),
            batch_size=self.args.batch_size, shuffle=False, drop_last=True,
            sampler=RandomMultipleGallerySampler(
                dataset.train, self.args.num_instances),
            pin_memory=True, num_workers=self.args.num_workers
        ), length=None)
        return cur_loader

    def get_optimizer(self, nets, epoch, optimizer_type='sgd'):
        if optimizer_type.lower() == 'sgd':
            optimizer = torch.optim.SGD(
                [{'params': sub_net.parameters()} for sub_net in nets],
                lr=self.args.lr, weight_decay=self.args.weight_decay,
                momentum=self.args.momentum
            )
            lr_scheduler = MultiStepLR(
                optimizer, milestones=self.args.milestones, gamma=0.5)
        elif optimizer_type.lower() == 'scaffold':
            optimizer = ScaffoldOptimizer(
                [{'params': sub_net.parameters()} for sub_net in nets],
                lr=self.args.lr, weight_decay=self.args.weight_decay
            )
            lr_scheduler = MultiStepLR(optimizer,
                                       milestones=self.args.milestones, gamma=0.5)
        lr_scheduler.step(epoch)
        return optimizer

    def train_resnet(self, net, avg_net, aug_mod,
                        global_epoch, client_id,
                        cls_layer, Features_norm, Labels, op_type='sgd'):
        net.train(True)
        avg_net.train(True)

        # Build temporary memory (Only used for training in the current epoch)
        memory = copy.deepcopy(self.memory)

        self.local_train.new_epoch()
        # avg optimizer
        optimizer = self.get_optimizer(
            nets=[avg_net, cls_layer, aug_mod],
            epoch=global_epoch, optimizer_type=op_type
        )
        # local optimizer
        optimizer_local = self.get_optimizer(
            nets=[net, ], epoch=global_epoch,
            optimizer_type=op_type
        )
        # local train, each contains local_ep epochs
        for batch_idx in range(self.max_iter):
            (images, _, labels, _, _) = self.local_train.next()
            images, labels = images.cuda(), labels.cuda()
            # generate data stats to normalize
            cur_mean, cur_var = images.mean(0), images.var(0)
            norm_images = (images-cur_mean).div(cur_var.sqrt()+1e-8)

            # PSCU Client-L model
            _, local_features_norm = net(images)
            loss_id_local = memory[client_id](local_features_norm, labels).mean()
            optimizer_local.zero_grad()
            loss_id_local.backward()
            optimizer_local.step()

            # PSCU Client-G model
            _, feature_avg_norm = avg_net(images)
            loss_id_avg = memory[client_id](feature_avg_norm, labels).mean() 

            # Generate transformed images
            aug_images = aug_mod(norm_images)

            # Maintain temporary memory
            with torch.no_grad():
                f_new = avg_net(aug_images)[1]
                Features_norm[client_id].append(f_new)
                Labels[client_id].append(labels)
                memory[client_id].module.MomentumUpdate(f_new, labels)
            loss_aug, loss_aux_avg, loss_wd= 0, 0, 0
            if global_epoch > 0:
                # Train STM
                freeze_avg = freeze_model(copy.deepcopy(avg_net))
                aug_feature_avg_freeze = freeze_avg(aug_images)
                feature_avg_freeze = freeze_avg(images)
                aug_score_avg_freeze = cls_layer(aug_feature_avg_freeze)
                score_avg_freeze = cls_layer(feature_avg_freeze)
                aug_feature_local,_ = net(aug_images)
                aug_score_local = cls_layer(aug_feature_local)
                # loss to disentangle, fL(I) < fA(I) < fA(I') < fL(I')
                loss_aux_avg = get_auth_loss(
                    get_entropy(F.softmax(aug_score_avg_freeze)),
                    get_entropy(F.softmax(score_avg_freeze)),
                    get_entropy(F.softmax(aug_score_local))
                )
                # div loss 
                shift_mean, shift_var = aug_mod.get_mean_var()
                loss_wd = -F.mse_loss(shift_mean, cur_mean) - \
                    F.mse_loss(cur_var, shift_var)

                # NSA
                aug_feature_avg, _ = avg_net(aug_images)
                aug_score_avg = cls_layer(aug_feature_avg)
                loss_aug = self.ce_loss(aug_score_avg, labels) + self.tri_loss(aug_feature_avg, labels) 

            loss = loss_id_avg + loss_aug + self.args.lam * loss_aux_avg + loss_wd
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print(f'Update Epoch / Total Epoch: [{global_epoch}/{self.args.epochs}]. Net Client: {client_id}. '
                    f'Iter / Total Iter: [{batch_idx + 1}/{self.max_iter}] (LossID_local: {loss_id_local.item():.2f}, '
                    f'LossID: {loss_id_avg.item():.2f}, LossAux:{float(loss_aux_avg):.2f})')

        return avg_net.state_dict()
    def train_resnetsnr(self, net, avg_net, aug_mod,
                        global_epoch, client_id,
                        fc, fc1, fc2, fc3, Features_norm, Labels, op_type='sgd'):
        net.train(True)
        avg_net.train(True)

        memory = copy.deepcopy(self.memory)

        self.local_train.new_epoch()
        # avg optimizer
        optimizer = self.get_optimizer(
            nets=[avg_net, fc, fc1, fc2, fc3, aug_mod],
            epoch=global_epoch, optimizer_type=op_type
        )
        # local optimizer
        optimizer_local = self.get_optimizer(
            nets=[net, ], epoch=global_epoch,
            optimizer_type=op_type
        )
        # local train, each contains local_ep epochs
        for batch_idx in range(self.max_iter):
            (images, _, labels, _, _) = self.local_train.next()
            images, labels = images.cuda(), labels.cuda()
            # generate data stats to normalize
            cur_mean, cur_var = images.mean(0), images.var(0)
            norm_images = (images-cur_mean).div(cur_var.sqrt()+1e-8)

            # PSCU Client-L model
            _, local_features_norm, x_IN_1_pool, x_1_useful_pool, x_1_useless_pool, \
                x_IN_2_pool, x_2_useful_pool, x_2_useless_pool, \
                x_IN_3_pool, x_3_useful_pool, x_3_useless_pool = net(images)
            
            x_IN_1_prob = F.softmax(fc1(x_IN_1_pool))
            x_1_useful_prob = F.softmax(fc1(x_1_useful_pool))
            x_1_useless_prob = F.softmax(fc1(x_1_useless_pool))
            x_IN_2_prob = F.softmax(fc2(x_IN_2_pool))
            x_2_useful_prob = F.softmax(fc2(x_2_useful_pool))
            x_2_useless_prob = F.softmax(fc2(x_2_useless_pool))
            x_IN_3_prob = F.softmax(fc3(x_IN_3_pool))
            x_3_useful_prob = F.softmax(fc3(x_3_useful_pool))
            x_3_useless_prob = F.softmax(fc3(x_3_useless_pool))
            loss_causality = 0.01 * get_auth_loss(get_entropy(x_IN_1_prob), get_entropy(x_1_useful_prob), get_entropy(x_1_useless_prob)) + \
                0.01 * get_auth_loss(get_entropy(x_IN_2_prob), get_entropy(x_2_useful_prob), get_entropy(x_2_useless_prob)) + \
                0.01 * get_auth_loss(get_entropy(x_IN_3_prob), get_entropy(
                    x_3_useful_prob), get_entropy(x_3_useless_prob))
            loss_id_local = memory[client_id](local_features_norm, labels).mean()
            loss = loss_causality + loss_id_local
            optimizer_local.zero_grad()
            loss.backward()
            optimizer_local.step()

            # PSCU Client-G model
            feature_avg, feature_avg_norm, x_IN_1_pool, x_1_useful_pool, x_1_useless_pool, \
                x_IN_2_pool, x_2_useful_pool, x_2_useless_pool, \
                x_IN_3_pool, x_3_useful_pool, x_3_useless_pool = avg_net(
                    images)
            
            x_IN_1_prob = F.softmax(fc1(x_IN_1_pool))
            x_1_useful_prob = F.softmax(fc1(x_1_useful_pool))
            x_1_useless_prob = F.softmax(fc1(x_1_useless_pool))
            x_IN_2_prob = F.softmax(fc2(x_IN_2_pool))
            x_2_useful_prob = F.softmax(fc2(x_2_useful_pool))
            x_2_useless_prob = F.softmax(fc2(x_2_useless_pool))
            x_IN_3_prob = F.softmax(fc3(x_IN_3_pool))
            x_3_useful_prob = F.softmax(fc3(x_3_useful_pool))
            x_3_useless_prob = F.softmax(fc3(x_3_useless_pool))
            score_avg = fc(feature_avg)
            loss_causality_avg = 0.01 * get_auth_loss(get_entropy(x_IN_1_prob), get_entropy(x_1_useful_prob), get_entropy(x_1_useless_prob)) + \
                0.01 * get_auth_loss(get_entropy(x_IN_2_prob), get_entropy(x_2_useful_prob), get_entropy(x_2_useless_prob)) + \
                0.01 * get_auth_loss(get_entropy(x_IN_3_prob), get_entropy(
                    x_3_useful_prob), get_entropy(x_3_useless_prob))
            loss_id_avg = memory[client_id](feature_avg_norm, labels).mean() 

            # Generate transformed images
            aug_images = aug_mod(norm_images)

            # Maintain temporary memory
            with torch.no_grad():
                features_norm = avg_net(aug_images)[1]
                Features_norm[client_id].append(features_norm)
                Labels[client_id].append(labels)
                memory[client_id].module.MomentumUpdate(features_norm, labels)

            loss_aug, loss_aux_avg, loss_wd= 0, 0, 0
            if global_epoch > 0:
                # Train STM
                aug_feature_avg, _, x_IN_1_pool_aug, x_1_useful_pool_aug, x_1_useless_pool_aug, \
                    x_IN_2_pool_aug, x_2_useful_pool_aug, x_2_useless_pool_aug, \
                    x_IN_3_pool_aug, x_3_useful_pool_aug, x_3_useless_pool_aug = avg_net(
                        aug_images)
                x_IN_1_prob_aug = F.softmax(fc1(x_IN_1_pool_aug))
                x_1_useful_prob_aug = F.softmax(fc1(x_1_useful_pool_aug))
                x_1_useless_prob_aug = F.softmax(fc1(x_1_useless_pool_aug))
                x_IN_2_prob_aug = F.softmax(fc2(x_IN_2_pool_aug))
                x_2_useful_prob_aug = F.softmax(fc2(x_2_useful_pool_aug))
                x_2_useless_prob_aug = F.softmax(fc2(x_2_useless_pool_aug))
                x_IN_3_prob_aug = F.softmax(fc3(x_IN_3_pool_aug))
                x_3_useful_prob_aug = F.softmax(fc3(x_3_useful_pool_aug))
                x_3_useless_prob_aug = F.softmax(fc3(x_3_useless_pool_aug))
                loss_aug_causality = 0.01 * get_auth_loss(get_entropy(x_IN_1_prob_aug), get_entropy(x_1_useful_prob_aug), get_entropy(x_1_useless_prob_aug)) + \
                    0.01 * get_auth_loss(get_entropy(x_IN_2_prob_aug), get_entropy(x_2_useful_prob_aug), get_entropy(x_2_useless_prob_aug)) + \
                    0.01 * get_auth_loss(get_entropy(x_IN_3_prob_aug), get_entropy(
                        x_3_useful_prob_aug), get_entropy(x_3_useless_prob_aug))

                aug_feature_local = net(aug_images)[0]
                aug_score_avg, aug_score_local = fc(
                    aug_feature_avg), fc(aug_feature_local)
                # loss to disentangle, fL(I) < fA(I) < fA(I') < fL(I')
                loss_aux_avg = get_auth_loss(
                    get_entropy(F.softmax(aug_score_avg)),
                    get_entropy(F.softmax(score_avg)),
                    get_entropy(F.softmax(aug_score_local))
                )
                loss_aug = self.ce_loss(aug_score_avg, labels) + self.tri_loss(aug_feature_avg, labels) + loss_aug_causality

                # div loss 
                shift_mean, shift_var = aug_mod.get_mean_var()
                loss_wd = -F.mse_loss(shift_mean, cur_mean) - \
                    F.mse_loss(cur_var, shift_var)

            # optimize avg model, sahre across domains
            loss = loss_id_avg + loss_causality_avg + loss_aug + self.args.lam * loss_aux_avg + loss_wd

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print(f'Update Epoch / Total Epoch: [{global_epoch}/{self.args.epochs}]. Net Client: {client_id}. '
                    f'Iter / Total Iter: [{batch_idx + 1}/{self.max_iter}] (LossID_local: {loss_id_local.item():.2f}, '
                    f'LossID: {loss_id_avg.item():.2f}, LossAux:{float(loss_aux_avg):.2f})')

        return avg_net.state_dict()