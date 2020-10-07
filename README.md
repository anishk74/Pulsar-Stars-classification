# Pulsar-Stars-classification
Identifying Pulsar Stars in a dataset of astronomical objects with Anomaly Detection algorithm using Gaussian Distribution.

## Description
The dataset contains only a few entries of Pulsar-Stars(1639:Positive, 16259:Negative). Considering those as anomalies, built a model to fit the non-anomalous data using 
Gaussian-Distribution. The training set only consists of non-anamolous examples while the Cross-Validation and Test sets consists of both the anomalous and non-anomalous
data for evaluation. Trained a model on the non-anamolous training data achieving the F1-Score of 0.8917 on test set.

## Technologies
* Python3
  * NumPy
  * Pandas
  * Scikit-learn
  * Seaborn
  * Matplotlib

## Initial Pairplot

[](img/init_pairplot.png?raw=true "Title")

Observing the pairplot,

The features
  * x0, x2
  * x0, x3
  * x1, x3
  * x2, x3
  * x4, x5
  * x4, x6
  * x4, x7
  * x5, x6
  * x5, x7
  * x6, x7

are dependent(near to linearly dependent for most examples).

## Adding new features

Adding new features to cover the examples.

  * x8 = x0/x2
  * x9 = x0/x3
  * x10 = x1/x3
  * x11 = x2/x3
  * x12 = x4/x5
  * x13 = x4/x6
  * x14 = x4/x7
  * x15 = x5/x6
  * x16 = x5/x7
  * x17 = x6/x7




  