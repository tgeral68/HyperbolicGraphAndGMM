# From Node Embedding To Community Embedding : A Hyperbolic Approach

## Introduction 
The provided code implements **RComE**  (https://arxiv.org/abs/1907.01662) the hyperbolic counterpart of **ComE**  algorithm (*S. Cavallari et al*). 
The repository performs graph embedding in hyperbolic space by preserving *first*, *second*-order proximities and
community awareness.
Tools and datasets used for developing and validating the methodology are provided in this package.

![gmm_clustering_flat](https://github.com/tgeral68/EM_Hyperbolic/raw/HyperbolicGraphAndGMM/ressources/LFR_community_clustering.png "title-1") 

## Dependencies
- python (version > 3.7)
- pytorch (version > 1.3)
- tqdm (pip only)
- matplotlib
- scikit-learn
- numpy
- argparse
- nltk (Hierarchical example script only)

**Install dependencies with pip**:

```
pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install tqdm matplotlib numpy argparse nltk scikit-learn
```

## Datasets
Small dataset are provided in the DATA folder for large one we provide a script to download them:
- Small datasets: Karate, Polblogs, Adjnoun, Polbooks, Football, lfr (generated dataset)
- Large datasets: DBLP, BlogCatalog and Wikipedia

Known true labels are also included. All datasets are described in the associated paper.
To download large datasets on linux we provide the script 'config_and_download.sh':
```
cd [path to]/EM_Hyperbolic
sh config_and_download.sh
```
Notice that you will need the tools *unzip* provided in most Linux distributions.
## Use our code
To train embedding we provide a script in "experiment_script/rcome_embedding.py" which run with the following parameters.

### Hyper-Parameters

|       Parameter name        | Description                      |
|----------------             |-------------------------------   |
| **dataset parameters**|
|--dataset                    | dataset to perform the learning stage (lower case) | 
| **Embedding parameters**|
|--dim      | dimenssion of embeddings |
| **Learning Parameters**|
|--lr |  Learning rate |
|--alpha         | O1 (*first-order proximity*) loss weight   |
|--beta        | O2 (*second-order proximity*) loss weight   |
|--gamma | O3 (*community-order*) loss weight |
|--walk-length       | size of path used in the random walk   |
|--context-size      | size of context (window on the path) |
|--n-negative | number of negative examples to use in the loss function |
|--batch-size      | how many sampled paths/nodes we consider for an update of weight|
| **others**|
|--verbose  | verbose mode|
|--plot     | plot representation and gmm at each step|
|--cuda     | Training using GPU acceleration (Nvidia GPU only)|
### Launching experiments
To make the script works you may need to add rcome package to python path (Linux):

```
PYTHONPATH=$PYTHONPATH:[folder to local repository]/rcome
export PYTHONPATH
```
And to run your experiments 
```
python experiment_script/rcome_embedding.py --dataset LFR1 --verbose
```

### Make you own grid-search
To make your own grid search we provide the script "experiment_script/grid_search.py" that generates a shell script from a JSON file. To get an idea of how it works you can try the following command :

```
python experiment_script/grid_search.py --cmd "python rcome_embedding" --file example/grid/grid_search_example.json --out example/grid/example_grid.sh
```
Which generates:
```
python rcome_embedding --context-size 10 --n-negative 5 --alpha 1 --beta 1  --dataset LFR1 --epoch 2 
python rcome_embedding --context-size 5 --n-negative 5 --alpha 1 --beta 1  --dataset LFR1 --epoch 2 
python rcome_embedding --context-size 10 --n-negative 5 --alpha 0.1 --beta 1  --dataset LFR1 --epoch 2 
python rcome_embedding --context-size 5 --n-negative 5 --alpha 0.1 --beta 1  --dataset LFR1 --epoch 2 
python rcome_embedding --context-size 10 --n-negative 5 --alpha 1 --beta 0.1  --dataset LFR1 --epoch 2 
python rcome_embedding --context-size 5 --n-negative 5 --alpha 1 --beta 0.1  --dataset LFR1 --epoch 2 
```
_______

### EM algorithm

To reproduce the EM algorithm without using your implementation we develop here some technical details to make it work.

- **Weighted Frechet Means:**  We set the variables ![equation](http://latex.codecogs.com/gif.latex?\lambda) (learning rate) and ![equation](http://latex.codecogs.com/gif.latex?\epsilon) (convergence rate) in Algorithm 1 of the paper to ![equation](http://latex.codecogs.com/gif.latex?\lambda=5e-2) and ![equation](http://latex.codecogs.com/gif.latex?\epsilon=1e-4).
- **Normalisation coefficient:** To evaluate the normalisation coefficient we pre-compute values of normalisation factor ![equation](http://latex.codecogs.com/gif.latex?\sigma) from a range of [1e-3,2] spaced of 1e-3. 
This parameter is of high importance: if the minimum value of sigma is too high then the precision in the unsupervised case will be higher on datasets that cannot be separated in small dimensions (wikipedia, blogCatalog mainly) because of considering the most represented community (for wikipedia most represented community labeled ![equation](http://latex.codecogs.com/gif.latex?\approx47\%) of nodes and ![equation](http://latex.codecogs.com/gif.latex?\approx17\%) for blog Catalog).
- **EM Convergence:** We consider that the EM algorithm converges when the values of ![equation](http://latex.codecogs.com/gif.latex?w_{ik}) change less than 1e-4  before and after an iteration or more formally when:


<p align="center">
  <img src="http://latex.codecogs.com/gif.latex?\frac{1}{N}\sum\limits_{i=0}^N\frac{1}{K}\sum\limits_{k=0}^K(|w_{ik}^t-w_{ik}^{t+1}|)<1e-4">
</p>

with ![equation](http://latex.codecogs.com/gif.latex?N) the number of nodes in the dataset and ![equation](http://latex.codecogs.com/gif.latex?K) the number of communities.

### Learning Embeddings

- **Optimisation :** In some cases due to the sum of gradient or a large distance, the update based on ![equation](http://latex.codecogs.com/gif.latex?exp_u(\lambda\nabla_f(u,v))) can leads to a norm equals to 1. In this special case we do not consider the gradient update. If it happens too frequently, we recommend lowering the learning rate.

