# One for All: Unified Workload Prediction for Dynamic Multi-tenant Edge Cloud Platforms (KDD'23 Research Track)
![](https://img.shields.io/badge/python-3.7-brightgreen.svg)
![](https://img.shields.io/badge/Pytorch-red.svg)
![](https://img.shields.io/badge/license-brightgreen.svg)

This is the origin Pytorch implementation of DynEformer in the following paper: [One for All: UnifiedWorkload Prediction for Dynamic Multi-tenant Edge Cloud Platforms.](https://arxiv.org/abs/2306.01507)

🚩News(May 31, 2023): We will soon release an updated mechanism for global pooling.

<div align="center">
  <img src="https://github.com/hsy23/KDD23_DynEformer/assets/45703329/ea33c6dc-973c-4175-b5fb-bef5da843802">
  <p>Figure 1.Framework overview of DynEformer</p>
</div>

## Global Pooling
Before predicting with DynEformer, first create a Global Pool for your data using [vade_main](models/GlobalPooing/vade_search_cluster/vade_main.py). This identifies and stores patterns in your time series data. In our work, the Global Pool is built on the seasonal component of edge cloud server load. Ensure [decomposed](models/GlobalPooing/series_decomp.py) data is generated before using vade_main when replicating the experiment.

<div align="center">
  <img src="https://github.com/hsy23/KDD23_DynEformer/assets/45703329/abadfc62-7fd4-4082-93c9-7921e3c8d9d9">
  <p>Figure 2.The process of building the global pool.</p>
</div>

## Requirements
- Python 3.7
- matplotlib == 3.5.3
- numpy == 1.21.6
- pandas == 1.3.5
- scikit_learn == 1.0.2
- torch == 1.13.0

Dependencies can be installed using the following command:

```pip install -r requirements.txt```

## Data
The ECW dataset used in the paper can be downloaded in the repo [ECWDataset](https://github.com/hsy23/ECWDataset). The required data files should be put into data folder. A demo slice of the ECW data is illustrated in the following figure. Note that the input of each dataset is Min-Max normalization in this implementation.

<div align="center">
  <img src="https://github.com/hsy23/KDD23_DynEformer/assets/45703329/b523ec9f-0e2f-49d6-9780-687a903790fd">
  <p>Figure 2.The process of building the global pool.</p>
</div>

## Pipline
To reproduce the experiments or to adapt the whole process to your own dataset, follow these steps:

**Reproduce:**

1. **run** 'models/GlobalPooling/vade_pooling/global_pooling_main.py' to **generate the global pool**
2. **run** 'models/DynEformer/dyneformer_main.py' to **train and save the DynEformer**
3. **run** 'models/DynEformer/dynet_app_switch.py' or 'models/DynEformer/dynet_newadded.py' to **inference**

**Adapt to your own dataset**

1. **run** 'models/GlobalPooling/vade_search_cluster.py' to **find the best K for the global pool**
2. **run** 'models/GlobalPooling/vade_pooling/global_pooling_main.py' to **generate the global pool**
3. **run** 'models/DynEformer/dyneformer_main.py' to **train and save the DynEformer**
4.  **run** 'models/DynEformer/dynet_app_switch.py' or 'models/DynEformer/dynet_newadded.py' to **inference**
