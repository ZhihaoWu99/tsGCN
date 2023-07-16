Topological and Semantic Regularized Graph Convolutional Network
====
This is the implementation of tsGCN proposed in our paper:

Shiping Wang, Zhihao Wu, Yuhong Chen, and Yong Chen*, [Beyond Graph Convolutional Network: An Interpretable Regularizer-Centered Optimization Framework](https://ojs.aaai.org/index.php/AAAI/article/view/25593), AAAI 2023.

Full paper can be found [HERE](https://arxiv.org/abs/2301.04318).

## Requirement

  * Python == 3.9.12
  * PyTorch == 1.11.0
  * Numpy == 1.21.5
  * Scikit-learn == 1.1.0
  * Scipy == 1.8.0
  * Texttable == 1.6.4
  * Tensorly == 0.7.0
  * Tqdm == 4.64.0

## Usage

```
python main.py
```

  * --device: number of gpus or 'cpu'.
  * --path: path of datasets.
  * --dataset: name of datasets.
  * --seed: random seed.
  * --fix_seed: fix the seed or not.
  * --n_repeated: number of repeated times.
  * --model: choose the model, GCN or tsGCN.
  * --bias: enable bias.
  * --lr: learning rate.
  * --weight_decay: weight decay.
  * --num_pc: number of labeled samples per class.
  * --num_epoch: number of training epochs.

All the configs are set as default, so you only need to set --dataset and --model. 
For example:

 ```
 python main.py --dataset Cora --model tsGCN
 ```

## Datasets

  * ACM
  * BlogCatalog
  * Citeseer
  * Cora
  * CoraFull
  * Flickr
  * Pubmed
  * UAI

Please unzip the datasets folders first.

Saved in ./datasets/datasets.7z

## Reference 
```
@inproceedings{
    wu2023tsGCN,
    title={Beyond Graph Convolutional Network: An Interpretable Regularizer-Centered Optimization Framework},
    author={Shiping Wang, Zhihao Wu, Yuhong Chen, Yong Chen},
    booktitle={Proceedings of the 37th AAAI Conference on Artificial Intelligence},
    year={2023},
}
```
