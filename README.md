# Pulsar-Stars-classification
Identifying Pulsar Stars in a dataset of astronomical objects with Anomaly Detection algorithm using Gaussian Distribution.

## Description
The dataset contains only a few entries of Pulsar-Stars(1639:Positive, 16259:Negative). Considering those as anomalies, built a model to fit the non-anomalous data using 
Gaussian-Distribution. The training set only consists of non-anamolous examples while the Cross-Validation and Test sets consists of both the anomalous and non-anomalous
data for evaluation. Trained a model on the non-anamolous training data achieving the F1-Score of 0.8804 on Cross-Validation set and 0.8767 on test set.

## Technologies
* Python3
  * NumPy
  * Pandas
  * Scikit-learn
  * Seaborn
  * Matplotlib

## Initial Pairplot

![](img/init_pairplot.png?raw=true "Title")

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
  * x13 = x6/x4
  * x14 = x4/x7
  * x15 = x6/x5
  * x16 = x7/x5
  * x17 = x6/x7

which leads to a histogram of training data
![](img/init_hist.png?raw=true "Title")

Notice the variance and standard deviation of the added features, those are very less which makes it easier for the model to differentiate anomalies.


## Feature-Scaling
Since the range of the features varies heavily, the features are scaled.

   x<sup>(i)</sup><sub>j</sub> = (x<sup>(i)</sup><sub>j</sub> - &mu;<sub>j</sub>) / &sigma;<sub>j</sub>

## Transforming the features to Gaussian-like

Using log, squareroot and other transformation techniques, a feature can be transformed to Gaussian-like. 
The transformed features of training data have a histogram
![](img/transformed_hist.png?raw=true "Title")

# Performance on Cross-Validation and Test data

Considering the features of training data as Gaussian, we build a Gaussian Distribution model and fit the parameters Mean and Standerd Deviation.

Using the mean and std of the Gaussian Distribution model,
We calculte the density estimation of the examples of CV data.


Density Estimation of the example x<sup>(i)</sup> = p(x<sup>(i)</sup><sub>0</sub>; &mu;<sub>0</sub>, &sigma;<sub>0</sub><sup>2</sup>) * p(x<sup>(i)</sup><sub>1</sub>; &mu;<sub>1</sub>, &sigma;<sub>1</sub><sup>2</sup>) * p(x<sup>(i)</sup><sub>2</sub>; &mu;<sub>2</sub>, &sigma;<sub>2</sub><sup>2</sup>) * ...... * p(x<sup>(i)</sup><sub>17</sub>; &mu;<sub>17</sub>, &sigma;<sub>17</sub><sup>2</sup>)

where

![](https://latex.codecogs.com/svg.latex?\Large&space;p({x^{(i)}_j};\mu_j,\sigma_j^2)=\frac{1}{\sqrt{2\pi}\sigma_j}e^{-(\frac{x^{(i)}_j-\mu_j}{\sigma_j})^2}) 

<!-- p(x<sup>(i)</sup><sub>j</sub>; &mu;<sub>j</sub>, &sigma;<sub>j</sub>)= e<sup>{-(x<sup>{(i)}</sup><sub>j</sub>- &mu;<sub>j</sub>)/&sigma;<sub>j</sub>)<sup>2</sup>}</sup> -->

Note: Density Estimation is different from probabilities. Probabilities range from 0 to 1. While Density Estimation of the mean of a Gaussian distribution with &sigma; < 1 will be greater than 1.
  
After calculating Density estimation for all the examples of Cross-Validation data, the Density Estimation which performs best classifying the CV examples is selected as the optimal threshold.

![](img/history.png?raw=true "Title")