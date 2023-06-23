# 3D Graph Contrastive Learning for Molecular Property Prediction
Official Code Repository for the paper "3D Graph Contrastive Learning for Molecular Property Prediction":  https://www.biorxiv.org/content/10.1101/2022.12.11.520009v2
[This is an external link to genome.gov](https://www.genome.gov/)

## Abstract
Self-supervised learning (SSL) is a method that learns the data representation by utilizing supervision inherent in the data. This learning method is in the spotlight in the drug field, lacking annotated data due to time-consuming and expensive experiments. SSL using enormous unlabeled data has shown excellent performance for molecular property prediction, but a few issues exist. (1) Existing SSL models are large-scale; there is a limitation to implementing SSL where the computing resource is insufficient. (2) In most cases, they do not utilize 3D structural information for molecular representation learning. The activity of a drug is closely related to the structure of the drug molecule. Nevertheless, most current models do not use 3D information or use it partially. (3) Previous models that apply contrastive learning to molecules use the augmentation of permuting atoms and bonds. Therefore, molecules having different characteristics can be in the same positive samples. We propose a novel contrastive learning framework, small-scale 3D Graph Contrastive Learning (3DGCL) for molecular property prediction, to solve the above problems. 3DGCL learns the molecular representation by reflecting the molecule’s structure through the pre-training process that does not change the semantics of the drug. Using only 1,128 samples for pre-train data and 0.5 million model parameters, we achieved state-of-the-art or comparable performance in six benchmark datasets. Extensive experiments demonstrate that 3D structural information based on chemical knowledge is essential to molecular representation learning for property prediction.


## Overview
<p align="center">
<img src=figures/3DGCL.png width=900px>
<img src=figures/methods_3D.png width=700px>
</p>

### Contribution
- We develop a compact self-supervised learning approach that can be run even in environments with low computational resources, using the small-scale pre-train samples and parameters. We also achieve the state-of-the-art or comparable performance in four regression benchmarks.
- To the best of our knowledge, we propose 3D-3D view contrastive learning that can take full advantage of 3D information for the first time. We actively utilize 3D positional information inherent in molecules through the pre-train scheme using the conformer pool.
- Extensive experiments demonstrate that our method, which can utilize structural information abundantly while maintaining semantics, is more suitable for molecular property prediction than conventional methods that can significantly change the structure or properties of molecules.

## Dependencies
- Python 3.6.9
- Pytorch 1.7.1
- Pytorch Geometric 2.0.3
- RDKit 2021.3.4

## Run
### 1. Pre-train
```shell script
run examples/sslgraph/pretrain.ipynb
```

### 2. Fine-tune
```shell script
run examples/sslgraph/finetune.ipynb
```

### 3. Supervised learning (No pre-train)
```shell script
run examples/sslgraph/downstream.ipynb
```


## Acknowledgment
Our implementation is mainly based on the following work.

[DIG: A Turnkey Library for Diving into Graph Deep Learning Research](https://github.com/divelab/DIG)

We are thankful for the great work.
