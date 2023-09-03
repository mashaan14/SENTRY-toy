# SENTRY PyTorch implementation with 2D toy example under label distribution shift (LDS)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.13-brightgreen.svg)](https://pytorch.org/get-started/previous-versions/)
[![torchvision](https://img.shields.io/badge/torchvision-0.14-brightgreen.svg)](https://pypi.org/project/torchvision/)

SENTRY standsfor **S**elective **ENTR**op**Y** Optimization via Committee Consistency for Unsupervised Domain Adaptation. It attempts to achieve domain adaptation under label distribution shift (LDS). SENTRY was introduced in this paper:

```bibtex
@InProceedings{Prabhu_2021_ICCV,
    author    = {Prabhu, Viraj and Khare, Shivam and Kartik, Deeksha and Hoffman, Judy},
    title     = {SENTRY: Selective Entropy Optimization via Committee Consistency for Unsupervised Domain Adaptation},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    year      = {2021},
    pages     = {8558-8567}
}
```

This SENTRY implementation uses a 2D toy dataset under label distribution shift (LDS). We used built-in plots that help to visualize how the SENTRY algorithm is learning the new features.

## 2D dataset under label distribution shift (LDS)
The code starts by retrieving `source dataset` from data folder. Then it performs a rotation (domain shift) on a copy of the dataset. The rotated dataset is the `target dataset`, with unbalanced label distribution. Here is a visualization of source and target datasets:
<p align="center">
  <img width="1200" src=dataset.png>
</p>

## Domain adaptation
The domain adaptation takes place in `core.train_tgt` function. SENTRY attempt to minimize three loss functions:

- `loss_CE` cross-entropy loss with respect to source ground-truth labels.
- `loss_IE` information-entropy loss computed over classes predicted by the model for the last-Q target instances
- `loss_SENTRY` selective entropy optimization:
	- minimizing predictive entropy with respect to the current target sample and one of its consistent versions.
	- maxiimizing predictive entropy with respect to the current target sample and one of its inconsistent versions.

## Consistent and inconsistent versions	of the target sample
The original paper used RandAugment to compute augmented versions of the current target sample. Since we're using 2D points, we cannot use image transformations in RandAugment. Therefore, we used k-nearest neighbor to find the nearest k points to the current target sample. We set k=7.

The goal is to train the `feature_extractor` to learn features for both `source` and `target` smaples. Here is the `feature_extractor` performance on `source` samples:

```
Avg Loss = 0.39374, Avg Accuracy = 90.500000%, ARI = 0.65438
```

<p align="center">
  <img width="1200" src="Testing source data using source feature extractor.png">
</p>

Now, we used the same `feature_extractor` to classify `target` samples. Note that we still did not perform domain adaptation:

```
Avg Loss = 0.50390, Avg Accuracy = 83.333333%, ARI = 0.40878
```

<p align="center">
  <img width="1200" src="Testing target data using source feature extractor.png">
</p>


After performing domain adaptation in `core.train_tgt` function, we can use the `feature_extractor` to classify `target` samples:


```
Avg Loss = 0.20647, Avg Accuracy = 90.833333%, ARI = 0.60555
```
<p align="center">
  <img width="1200" src="Testing target data using target feature extractor.png">
</p>


## Code acknowledgement
I reused some code from the original [repository](https://github.com/virajprabhu/SENTRY) provided by the authors.
