# circleloss.pytorch
Examples of playing with Circle Loss from the paper "[Circle Loss: A Unified Perspective of Pair Similarity Optimization](https://arxiv.org/abs/2002.10857)", CVPR 2020.

The implementation of Circle Loss is from [TinyZeaMays/CircleLoss](https://github.com/TinyZeaMays/CircleLoss).

## Example 1: Visualization (learned features)
```
>>> python example_vis.py
```
<p align="center">
<img src="https://github.com/zhjohnchan/circleloss.pytorch/blob/master/figures/tsne.png" width = "50%" />
</p>

## Example 2: Classification
### Training w/o circle loss
```
>>> python example_cls_wo_circleloss.py
[1/40] Training classifier.
Test set: Accuracy: 5348/10000 (53%)
...
[40/40] Training classifier.
Test set: Accuracy: 9863/10000 (99%)
```
This will train a simple neural network under the cross entropy loss.
### Training w/ circle loss
```
>>> python example_cls.py
[1/20] Training with Circle Loss.
...
[20/20] Training with Circle Loss.
[1/20] Training classifier. Test set: Accuracy: 9682/10000 (97%)
...
[20/20] Training classifier. Test set: Accuracy: 9888/10000 (99%)
```
This will train a simple neural network under the circle loss firstly, and then train a classifier under the cross entropy loss using the extracted features.

## Example 3: Comparison
```
>>> python example_compare.py
```

<p align="center">
<img src="https://github.com/zhjohnchan/circleloss.pytorch/blob/master/figures/compare.png" width = "50%" />
</p>

## Acknowledgements
Thanks the implementation [TinyZeaMays/CircleLoss](https://github.com/TinyZeaMays/CircleLoss) and the authors of the paper.
