# MMFL

The code has been uploaded.

## Dataset

The repository for the dataset is [here](https://github.com/yxcai-alt/MultimodalDataset).

## RUN

### TADPOLE

```python
python main.py --dataset TADPOLE --task SMCI_PMCI 
python main.py --dataset TADPOLE --task AD_CN_SMCI
python main.py --dataset TADPOLE --task AD_CN_SMCI_PMCI
```

### ADNI3

```python
python main.py --dataset TADPOLE --task AD_CN
python main.py --dataset TADPOLE --task SMC_EMCI_LMCI
python main.py --dataset TADPOLE --task AD_CN_EMCI_LMCI
```

### ABIDE

```python
python main.py --dataset ABIDE --task NC_ASD
```

### ABIDE-5

```python
python main.py --dataset ABIDE-5 --task NC_ASD
```

## Requirements

It is recommended to install the [mamba wheels](https://github.com/yxcai-alt/Mamba-ssm) we built.

```
python                    3.10
pytorch                   2.1.1
cuda                      11.8
scikit-learn              1.5.2
pandas                    2.2.3
scipy                     1.14.1
tqdm                      4.66.5
```
