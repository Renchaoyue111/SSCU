import argparse
import os
import os.path as osp
import random
import numpy as np
# import sys
import time
from datetime import timedelta
import collections

import torch.nn.functional as F
import torch
from torch import nn
from torch.backends import cudnn

from reid.models.memory import MemoryClassifier
from reid import models
from reid.server import FedDomainMemoTrainer
from reid.evaluators import Evaluator, extract_features
# from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint
from reid.utils.tools import get_test_loader, get_data
from reid import datasets
import copy


start_epoch = mAP_Best = R1_best = R1_last = 0


def create_model(args, num_cls=0):
    # resnet50
    # model = models.create(
    #     args.arch, num_features=args.features, norm=True,
    #     dropout=args.dropout, num_classes=num_cls
    # )
    # resnet50-snr
    model = models.create( #resnet50_snr
        args.arch, norm=True,num_classes=num_cls
    )
    # use CUDA
    model = model.cuda()
    model = nn.DataParallel(model) if args.is_parallel else model
    return model

def main():
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    main_worker(args)

def main_worker(args):
    global start_epoch, mAP_Best, R1_best, R1_last
    start_time = time.monotonic()

    cudnn.benchmark = True
    all_datasets = datasets.names()
    test_set_name = args.test_dataset
    all_datasets.remove(test_set_name)
    
    if args.exclude_dataset is not '':
        exclude_set_name = args.exclude_dataset.split(',')
        [all_datasets.remove(name) for name in exclude_set_name]
    train_sets_name = sorted(all_datasets)
    
    print("==========\nArgs:{}\n==========".format(args))
    # Create datasets
    print("==> Building Datasets")
    test_set = get_data(args) 
    test_loader = get_test_loader(test_set, args.height, args.width,  
                                  args.batch_size, args.workers)
    
    train_sets = get_data(args, train_sets_name) 
    num_classes1 = train_sets[0].num_train_pids
    num_classes2 = train_sets[1].num_train_pids
    num_classes3 = train_sets[2].num_train_pids
    num_classes = [num_classes1, num_classes2, num_classes3]
    # num_classes = [num_classes1, num_classes2]
    print('number classes = ', num_classes)
    num_users = len(train_sets) 

    # Create the server and client models respectively
    model = create_model(args)  
    sub_models = [create_model(args) for key in range(num_users)]
    aug_mods = [
        models.create('aug', num_features=3, width=args.width, height=args.height).cuda() 
        for idx in range(num_users)]
    
    #Initialize GGDSM
    print("==> Initialize Generalization Gain-Guided Dynamic Style Memory (GGDSM)")
    memories = []
    for dataset_i in range(len(train_sets)):
        dataset_source = train_sets[dataset_i] 
        sour_cluster_loader = get_test_loader(dataset_source, args.height, args.width,
                                              args.batch_size, args.workers,testset=sorted(dataset_source.train)) 
        source_features, _ = extract_features(model, sour_cluster_loader, print_freq=50)
        sour_fea_dict = collections.defaultdict(list) 
        
        for f, pid, _ in sorted(dataset_source.train):
            sour_fea_dict[pid].append(source_features[f].unsqueeze(0)) 

        source_centers = [torch.cat(sour_fea_dict[pid], 0).mean(0) for pid in sorted(sour_fea_dict.keys())] 
        source_centers = torch.stack(source_centers, 0)  
        print(source_centers.shape)

        source_centers = F.normalize(source_centers, dim=1).cuda() 
        print('The number of identity centers is:',source_centers.shape[0])
        curMemo = MemoryClassifier(2048, source_centers.shape[0], temp=args.temp, momentum=args.momentumm).cuda()
        curMemo.features = source_centers
        curMemo.labels = torch.arange(num_classes[dataset_i]).cuda()
        curMemo = nn.DataParallel(curMemo) 
        
        memories.append(curMemo) 

        del source_centers, sour_cluster_loader, sour_fea_dict

    evaluator = Evaluator(model)

    # snr=False: resnet / snr=True: resnet50-snr
    # trainer = FedDomainMemoTrainer(args, train_sets, model, memory=memories, snr=False)
    trainer = FedDomainMemoTrainer(args, train_sets, model, memory=memories, snr=True)

    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        for idx in range(num_users):
            sub_models[idx].load_state_dict(checkpoint['sub_models'][idx])
            # trainer.classifier[idx].load_state_dict(checkpoint['cls_params'][idx]) #only resnet50
        start_epoch = checkpoint['epoch'] - 1
        model.load_state_dict(checkpoint['state_dict'])
    
    # onlu evaluate
    if args.evaluate:
        evaluator.evaluate(test_loader, test_set.query, test_set.gallery, cmc_flag=True)
        return
        
    # start training
    for epoch in range(start_epoch, args.epochs):  # number of epochs
        w_locals = []
        torch.cuda.empty_cache()
        Features_norm = [[],[],[]]
        Labels = [[],[],[]]
        for index in range(num_users):  
            # avg model boosted 
            # snr=False: resnet / snr=True: resnet50-snr
            w = trainer.train_SSCU(
                memories, sub_models[index], model, aug_mods[index], epoch, index, 
                op_type='sgd', Features_norm=Features_norm, Labels=Labels, snr=True
            )
            w_locals.append(w)
        # update server-side global model
        w_global = trainer.fed_avg(w_locals)
        model.load_state_dict(w_global)
        # evaluate
        mAP_cur, cmc = evaluator.evaluate(test_loader, test_set.query,test_set.gallery, cmc_flag=True)
        if cmc[0] >= R1_last:
            print('sever Performance better with this epoch of style images,update the memory!')
            for index in range(num_users):  
                for lenth in range(len(Labels[index])):
                    memories[index].module.MomentumUpdate(Features_norm[index][lenth], Labels[index][lenth])
        R1_last = cmc[0]
        if mAP_cur >= mAP_Best :
            print('best model saved!')
            save_checkpoint({
                'state_dict': w_global,
                # 'cls_params': [cls_layer.state_dict() for cls_layer in trainer.classifier],
                'sub_models': [mod.state_dict() for mod in sub_models],
                'epoch': epoch + 1, 'mAP_Best': mAP_Best,
            }, 1, fpath=osp.join(args.logs_dir, f'checkpoint_{epoch}.pth.tar'))
            mAP_Best = mAP_cur

        # save some transformed samples
        # if epoch % args.eval_step == 0:
        #     trainer.save_images(aug_mods[train_sets_name.index('msmt17')], epoch)
        #     # save tsne
        #     name_list = [cur_set.__class__.__name__ for cur_set in train_sets]
        #     for index in range(num_users):  
        #         file_path = os.path.join(args.logs_dir, f"{name_list[index]}.pth")
        #         trainer.get_pth(index,file_path = file_path)

        #     trainer.save_tsne(
        #         [os.path.join(args.logs_dir, f"{pth_name}.pth") for pth_name in name_list], model, 
        #         aug_mods[train_sets_name.index('msmt17')], epoch
        #     )
        #     return

    end_time = time.monotonic()
    print('bese mAP:',mAP_Best)
    print('best rank1:',R1_best)
    print('Total running time: ', timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Domain-level Fed Learning")
    # data
    parser.add_argument('-td', '--test-dataset', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-ed', '--exclude-dataset', type=str, default='')
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    # model
    # backbone: resnet50/resnet50-snr
    # parser.add_argument('-a', '--arch', type=str, default='resnet50',
    #                     choices=models.names())
    parser.add_argument('-a', '--arch', type=str, default='resnet50_snr',
                        choices=models.names())

    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--lam', type=float, default=1)
    # optimizer
    parser.add_argument('--temp', type=float, default=0.05, help="temperature")
    parser.add_argument('--rho', type=float, default=0.05, help="rho")
    parser.add_argument('--momentum', type=float, default=0.9,
                        help="momentum to update model")
    parser.add_argument('--momentumm', type=float, default=0.9,
                        help="momentumm to update memory")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--warmup-step', type=int, default=10)
    
    parser.add_argument('--milestones', nargs='+', type=int, 
                        default=[20, 40], help='milestones for the learning rate decay')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="learning rate")
    
    parser.add_argument('--epochs', type=int, default=70)
    parser.add_argument('--max-iter', type=int, default=200)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 4")
    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--eval-step', type=int, default=10)
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--is_parallel', type=int, default=1)
    main()
