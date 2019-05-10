capsule-attention-networks
==============================

Capsule networks to identify crop type of satellite imagery with 88% accuracy at 30x30m resolution.
Preprint available on [arxiv](https://arxiv.org/pdf/1904.10130.pdf).

## Data

The data set was provided in Rubworm and Korner (2017), comprising 26 Sentinel 2A images taken during 2016 of a 102 x 42 km area in Munich, Germany. The 10 m bands (2 blue, 3 green, 4 red, 8 near-infrared) and 20 m bands (11 short-wave-infrared-1, 12 short-wave- infrared-2) down-sampled to 10 m resolution were extracted from 406,000 points of interest were within the data set. Input data was formatted to 3 x 3 px neighborhoods for each of the 26 time steps with an approximate 75, 5, and 20 percent train, validation, and test ratio.

Ground truth labels were provided by the Bavarian Ministry of Agriculture, totalling 19 classes with at least 400 occur- rences (corn, corn, meadow, asparagus, rape, hops, summer oats, winter spelt, fallow, winter wheat, winter barley, winter rye, beans, winter triticale, summer barley, peas, potatoes, soybeans, and sugar beets). Sample points were additionally classified into cloud, water, snow, and cloud shadow, as well as an other category for non-agricultural pixels.

## Architecture overview

![Overview of approach](https://github.com/JohnMBrandt/capsule-attention-networks/blob/master/reports/figures/figure2.png?raw=true)

## Results

![Confusion matrix](https://github.com/JohnMBrandt/capsule-attention-networks/blob/master/reports/figures/figure1.png?raw=true)

| Model    | F-Score |
|----------|---------|
| SVM      | 31.1    |
| CNN      | 56.7    |
| LSTM     | 74.5    |
| CapsAttn | 85.6    |


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks.
    │    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox


--------
