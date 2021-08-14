# Facebook Friend Recommender

This is a friend recommendation system used on social media platforms (e.g. Facebook, Instagram, Twitter) to suggest friends/new connections based on common interests, workplace, common friends etc. using Graph Mining techniques. Here, we are given a social graph, i.e. a graph structure where nodes are individuals on social media platforms and a directed edges (or 'links') indicates that one person 'follows' the other, or are 'friends' on social media. Now, the task is to predict newer edges to be offered as 'friend suggestions'.

First we will load our dataset from Kaggle and perform exploratory data analysis on our given data set such as number of followers and followees of each person. Then we will generate some datapoints which were not present in our given data-set, since we have only class label 1 data. Then we will do some feature engineering on dataset like finding shortest path, kartz centrality, jaccard distances, page rank, preferential attachements etc. After performing exploratory data analysis and feature engineering, we will split whole dataset into train and test and perform random forest and xgboost taking f1-score as our metric. At the end we will plot confusion matrix and pretty-table for both algorithm and finf best hyperparameters.

**Problem statement** - Given a directed social graph, have to predict missing links to recommend users (Link Prediction in graph)

## Project structure
```
.
├── code
│   └── nbs
│       ├── reco-tut-ffr-01-ingestion.py
│       ├── reco-tut-ffr-02-ETL.py
│       ├── reco-tut-ffr-03-preprocessing.py
│       ├── reco-tut-ffr-04-feature-engineering.py
│       └── reco-tut-ffr-05-modeling.py
├── data
│   ├── bronze
│   │   ├── test.parquet.gzip
│   │   └── train.parquet.gzip
│   ├── gold
│   │   ├── katz.p
│   │   ├── page_rank.p
│   │   ├── storage_sample_stage1.h5
│   │   ├── storage_sample_stage2.h5
│   │   ├── storage_sample_stage3.h5
│   │   └── storage_sample_stage4.h5
│   └── silver
│       ├── X_test_neg.parquet.gzip
│       ├── X_test_pos.parquet.gzip
│       ├── X_train_neg.parquet.gzip
│       ├── X_train_pos.parquet.gzip
│       ├── y_test.parquet.gzip
│       └── y_train.parquet.gzip
├── graph_sample.pdf
├── LICENSE
├── model
├── notebooks
│   ├── reco-tut-ffr-01-ingestion.ipynb
│   ├── reco-tut-ffr-02-ETL.ipynb
│   ├── reco-tut-ffr-03-preprocessing.ipynb
│   ├── reco-tut-ffr-04-feature-engineering.ipynb
│   └── reco-tut-ffr-05-modeling.ipynb
└── README.md  
```