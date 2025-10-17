# Positive Style Accumulation: A Style Screening and Continuous Utilization Framework for Federated DG-ReID (MM'25)

## Start Guideline

Official PyTorch implementation of the paper Positive Style Accumulation: A Style Screening and Continuous Utilization Framework for Federated DG-ReID. (MM 2025)[arXiv](https://arxiv.org/abs/2507.16238)


![](figures/flowchart.pdf)

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
