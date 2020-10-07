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

Observing the pairplot of training data,

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

Adding new features to training data(as well as cross validation and test data)

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

which leads to a histogram of training data
![](img/init_hist.png?raw=true "Title")

Notice the variance and standard deviation of the added features, those are very less which makes it easier for the model to differentiate anomalies.
Since the range of the features are 

## Transforming the features to Gaussian-like

Using log, squareroot and other transformation techniques, we can convert a feature to Gaussian-like. 
The transformed features of training data have a histogram
![](img/transformed_hist.png?raw=true "Title")

# Performance on Cross-Validation and Test data

Using the mean and std of training data,
We evaluate the examples of CV data


