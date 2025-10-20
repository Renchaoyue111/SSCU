# Positive Style Accumulation: A Style Screening and Continuous Utilization Framework for Federated DG-ReID (MM'25)

## Start Guideline

Official PyTorch implementation of the paper Positive Style Accumulation: A Style Screening and Continuous Utilization Framework for Federated DG-ReID. (MM 2025)[arXiv](https://arxiv.org/abs/2507.16238)

![](./figures/Method diagram.pdf)

## Introduction
The Federated Domain Generalization for Person re-identification (FedDG-ReID) aims to learn a global server model that can be effectively generalized to source and target domains through distributed source domain data. Existing methods mainly improve the diversity of samples through style transformation, which to some extent enhances the generalization performance of the model. However, we discover that not all styles contribute to the generalization performance. Therefore, we define styles that are beneficial/harmful to the model's generalization performance as positive/negative styles. Based on this, new issues arise: How to effectively screen and continuously utilize the positive styles. To solve these problems, we propose a Style Screening and ContinuousUtilization (SSCU) framework. Firstly, we design a Generalization Gain-guided Dynamic Style Memory (GGDSM) for each client model to screen and accumulate generated positive styles. Specifically, the memory maintains a prototype initialized from raw data for each category, then screens positive styles that enhance the global model during training, and updates these positive styles into the memory using a momentum-based approach. Meanwhile, we propose a style memory recognition loss to fully leverage the positive styles memorized by GGDSM. Furthermore, we propose a Collaborative Style Training (CST) strategy to make full use of positive styles. Unlike traditional learning strategies, our approach leverages both newly generated styles and the accumulated positive styles stored in memory to train client models on two distinct branches. This training strategy is designed to effectively promote the rapid acquisition of new styles by the client models, ensuring that they can quickly adapt to and integrate novel stylistic variations. Simultaneously, this strategy guarantees the continuous and thorough utilization of positive styles, which is highly beneficial for the model's generalization performance. Extensive experimental results demonstrate that our method outperforms existing methods in both the source domain and the target domain.

![](figures/flowchart.pdf)

## News

**2025/10/17** We have released the official codes.

**2025/07/06** Accepted by 33rd ACM International Conference on Multimedia (ACM MM25)


### 1. Setup

- CUDA>=11.7
- At least two Nvidia GeForce RTX-2080Ti GPUs
- Other necessary packages listed in [requirements.txt](requirements.txt)
```bash
pip install -r requirements.txt
```
- Download [ViT pre-trained model](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth) and put it under "./checkpoints"
- Training Data
  
  Download Market1501, MSMT17 and CUHK03 datasets from http://www.liangzheng.org/Project/project_reid.html 

   Unzip all datasets and ensure the file structure is as follow:
   
   ```
   SSCU/data    
   │
   └───market1501 OR msmt17
        │   
        └───Market-1501-v15.09.15 OR MSMT17_V1
            │   
            └───bounding_box_train
            │   
            └───bounding_box_test
            | 
            └───query
            │   
            └───list_train.txt (only for MSMT-17)
            | 
            └───list_query.txt (only for MSMT-17)
            | 
            └───list_gallery.txt (only for MSMT-17)
            | 
            └───list_val.txt (only for MSMT-17)
   ```

### 2. Train

- Train
```bash
CUDA_VISIBLE_DEVICES=1 python -W ignore FedDG_SSCU.py -td market1501 --logs-dir ./logs/mar --data-dir ./data
```

### 3.Citation

Please cite our work in your publications if it helps your research:

```
@misc{xu2025styleaccumulation,
      title={Positive Style Accumulation: A Style Screening and Continuous Utilization Framework for Federated DG-ReID}, 
      author={Xin Xu and Chaoyue Ren and Wei Liu and Wenke Huang and Bin Yang and Zhixi Yu and Kui Jiang},
      year={2025},
      eprint={2507.16238},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.16238}, 
}
```

### 4.Contact Us

Email: renchaoyue@wust.edu.cn
