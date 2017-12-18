# Protein Family Classification

This repository aims at reproducing results from the [paper](https://cs224d.stanford.edu/reports/LeeNguyen.pdf). 
The project uses tensorflow, scikit*, numpy, pandas and nltk,
The model achieved f1 score of 0.83 on cv dataset.
Dataset used is swissprot-kB. Families with < 200 examples and sequences with length > 1000 were removed at the time of preprocessing.
Glove model was used to create embeddings.
*used only for getting f1 score, can be imlplemented separately.


## Getting Started

Download the dataset from [here](http://www.uniprot.org/) .
The file is SwissProt-kB.
Using utils.py the data can be pre-processed and run the model.py finally.

### Prerequisites

1. Tensorflow
2. Scikit-learn
3. Numpy
4. Pandas
5. NLTK

All the libraries can be installed using pip3.


## Train time

Each epoch using Tesla-K80 took approx ~ 4 secs.

## Authors

* [Me](https://github.com/s1998) and [Uday](https://github.com/Udayraj123). 


## Similar repo

* If interested, check out the repo for [Protein Secondary Structure Prediction](https://github.com/Udayraj123/protein-secondary-structure-prediction). 

<!-- ## Acknowledgments

* Hat tip to anyone who's code was used
* Inspiration
* etc
 -->
