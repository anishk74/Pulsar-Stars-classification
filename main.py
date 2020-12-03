import pandas as pd
import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt
import seaborn as sns

from model.model import Gaussian


def splitData(dataset):

	frames = [frame for _,frame in dataset.groupby('y')]
	negds=frames[0]
	posds=frames[1]

	train, test = np.split(negds, [int(0.8*len(negds))])

	cv, test = np.split(test, [int(0.5*len(test))])

	cv_pos, test_pos = np.split(posds, [int(0.5*len(posds))])

	cv=cv.append(cv_pos)
	test=test.append(test_pos)

	return train, cv, test


#Transform the features to an approximate Gaussian distribution
def transformFeats(ds):
	ds['x0']=(ds['x0']+1)**2
	ds['x1']=(np.log((ds['x1']+0.1)))
	ds['x2']=np.log(ds['x2']+0.18)
	ds['x3']=np.log(ds['x3']+0.004)
	ds['x4']=np.log(ds['x4']+0.0001)
	ds['x5']=np.log(ds['x5']+0.012)
	ds['x6']=np.log(ds['x6']+0.23)
	ds['x7']=(ds['x7']+10**-4)**(0.26)
	return ds


def featScale(ds,minima,maxima):
	ds=(ds-minima)/(maxima-minima)
	return ds


sns.set()

dataset=pd.read_csv('pulsar_stars.csv')

columns=['x'+str(i) for i in range(len(dataset.columns)-1)]
columns.append('y')
featureCount=len(columns)-1

dataset.columns=columns


labels=dataset.pop('y')

#Feature-scaling
orig_minima=dataset.min()
orig_maxima=dataset.max()

dataset=featScale(dataset, orig_minima, orig_maxima)

dataset['y']=labels

#Initial Histogram
dataset[dataset['y']==0].hist(bins=50,figsize=(13,13))
plt.tight_layout()
plt.savefig('img/init_hist.png')
plt.close()

dataset=transformFeats(dataset)

#Histogram of transformed features
dataset[dataset['y']==0].hist(bins=50,figsize=(13,13))
plt.tight_layout()
plt.savefig('img/transformed_hist.png')
plt.close()



#Pairplot

sns.pairplot(dataset,hue='y',diag_kind='kde',plot_kws={'s':7})
plt.tight_layout()
plt.savefig('img/pairplot.png')
plt.close()



dataset = dataset.sample(frac=1)

train, cv, test=splitData(dataset)

train_labels=train.pop('y')
cv_labels=cv.pop('y')
test_labels=test.pop('y')



model=Gaussian(train.mean(),train.std(),train.columns)

history=model.fit(train,train_labels,cv,cv_labels)

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])

plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.legend(['Training','Cross-Validation'])

plt.subplot(1,2,2)
plt.plot(history['f1'],'g')

plt.xlabel('Iteration')
plt.ylabel('F1-score')
plt.legend(['Cross-Validation'])
plt.tight_layout()
plt.savefig('img/output_measures.png')

testAcc,testF1=model.evaluate(test, test_labels)
print()
print('Accuracy of the model on Test data: ',testAcc)
print('F1-score of the model on Test data: ',testF1)

