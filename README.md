# Pulsar-Stars-classification
Identifying Pulsar Stars in a dataset of astronomical objects with Anomaly Detection algorithm using Gaussian Distribution.

## Description
The dataset contains only a few entries of Pulsar-Stars(1639:Positive, 16259:Negative). Considering those as anomalies, built a model to fit the non-anomalous data using 
Gaussian-Distribution. The training set only consists of non-anamolous examples while the Cross-Validation and Test sets consists of both the anomalous and non-anomalous
data for evaluation. Trained a model on the non-anamolous training data achieving the F1-Score of 0.8564 on Cross-Validation set and 0.8551 on test set.

## Technologies
* Python3
  * NumPy
  * Pandas
  * Scikit-learn
  * Seaborn
  * Matplotlib

## Initial Histogram of features(Non-anomalous Examples)

![](img/init_hist.png?raw=true "Title")

## Histogram of transformed features(Non-anomalous Examples)

Using log, power and other transformation techniques, a feature can be transformed to Gaussian-like. 
The transformed features of Non-anomalous examples have a histogram
![](img/transformed_hist.png?raw=true "Title")

## Pairplot of the transformed features
![](img/pairplot.png?raw=true "Title")

# Performance on Cross-Validation and Test data

![](img/output_measures.png?raw=true "Title")

The plot shows the Training accuracy and Cross-Validation accuracy as well as Cross-Validation F1-score throughout the training epochs. The model chooses those parameters for which the F1-score is maximum on Cross-Validation data, achieving the F1-Score of 0.8564 on Cross-Validation data and 0.8551 on test data.
.
