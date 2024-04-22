#!/bin/bash

kaggle competitions download -c house-prices-advanced-regression-techniques

unzip house-prices-advanced-regression-techniques.zip -d data/

rm house-prices-advanced-regression-techniques.zip

mv data/test.csv data/predict.csv
