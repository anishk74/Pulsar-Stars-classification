import pandas as pd
import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

import seaborn as sns


def splitData(dataset):
	frames = [frame for _,frame in dataset.groupby('y')]
	negds=frames[0]
	posds=frames[1]

	train, cv, test = np.split(negds, [int(0.7*len(negds)), int(0.85*len(negds))])

	cv_pos, test_pos=  np.split(posds, [int(0.5*len(posds))])

	cv=cv.append(cv_pos)
	test=test.append(test_pos)

	return train, cv, test


#Transform the features to an approximate Gaussian distribution
def transformFeats(ds):
	ds['x1']=(ds['x1']+4)**(1/5)
	ds['x2']=np.sqrt(ds['x2']+7)
	ds['x3']=np.log(ds['x3']+2.2)
	ds['x4']=np.log(ds['x4']+0.355)
	ds['x5']=np.log(ds['x5']+1.1)
	ds['x6']=np.sqrt(ds['x6']+4)
	ds['x7']=(ds['x7']+1.1)**(1/5)

	ds['x8']=np.log(ds['x8']+50)
	ds['x9']=np.log(ds['x9']+150)
	ds['x10']=np.log(ds['x10']+250)

	ds['x11']=((ds['x11']+100)**3)/1000000
	
	ds['x12']=np.log(ds['x12']+0.53)
	ds['x13']=np.log((ds['x13'] + 0.68))
	ds['x14']=((ds['x14']+80)**(1.5))
	ds['x15']=np.log((ds['x15'] + 1.8))
	ds['x16']=np.log((ds['x16'] + 0.74))

	ds['x17']=((ds['x17']+100)**(1/2))

	return ds


def addNewFeats(ds):
	ds['x8']=ds['x0']/ds['x2']
	ds['x9']=ds['x0']/ds['x3']
	ds['x10']=ds['x1']/ds['x3']
	ds['x11']=ds['x2']/ds['x3']
	ds['x12']=ds['x4']/ds['x5']
	ds['x13']=ds['x6']/ds['x4']
	ds['x14']=ds['x4']/ds['x7']
	ds['x15']=ds['x6']/ds['x5']
	ds['x16']=ds['x7']/ds['x5']
	ds['x17']=ds['x6']/ds['x7']
	return ds




def featScale(ds,mean,std):
	ds=(ds-mean)/std
	return ds


sns.set()

dataset=pd.read_csv('pulsar_stars.csv')

columns=['x'+str(i) for i in range(len(dataset.columns)-1)]
columns.append('y')
featureCount=len(columns)-1

dataset.columns=columns

#splitting the dataset
train, cv, test=splitData(dataset)

train_labels=train.pop('y')
cv_labels=cv.pop('y')
test_labels=test.pop('y')


#initial pairplot to find linearly dependent features

sns.pairplot(train,diag_kind='kde',plot_kws={'s':7})
plt.savefig('init_pairplot.png')
plt.close()


#adding new features
train=addNewFeats(train)
cv=addNewFeats(cv)
test=addNewFeats(test)

#Initial histograms of the features
train.hist(bins=50,figsize=(13,13))
plt.savefig('init_hist.png')
plt.close()

orig_mean=train.mean()
orig_std=train.std()


#feature-scaling
train=featScale(train, orig_mean, orig_std)
cv=featScale(cv, orig_mean, orig_std)
test=featScale(test, orig_mean, orig_std)


#Final histogram of the transformed gaussian-like features
train=transformFeats(train)
cv=transformFeats(cv)
test=transformFeats(test)

train.hist(bins=50,figsize=(13,13))
plt.savefig('transformed_hist.png')
plt.close()


#Evaluating on Cross-Validation set

props=train.describe().transpose()
train_mean=props['mean']
train_std=props['std']

clmns=train.columns

'''
#training probabilities
train_probs= np.exp((((train-train_mean)/(train_std)) ** 2) * (-1/2)) / (np.sqrt(2 * np.pi) * train_std)
train_pval=[1]*train[clmns[0]].size
for i in clmns:
	train_pval=train_pval*train_probs[i]
'''

#Cross-Validation set probabilities
cv_probs= np.exp((((cv-train_mean)/(train_std)) ** 2) * (-1/2)) / (np.sqrt(2 * np.pi) * train_std) #Features' individual probability value
cv_pval=[1]*cv[clmns[0]].size #Combined probability Value

for i in clmns:
	cv_pval=cv_pval*cv_probs[i]



bestF1=0
bestEpsilon=0

history=[]
for epsilon in cv_pval:
	pred=cv_pval<epsilon
	
	acc=accuracy_score(cv_labels,pred)
	recall=recall_score(cv_labels,pred)
	f1= 2*acc*recall/(acc+recall)
	if f1>bestF1:
		bestF1=f1
		bestEpsilon=epsilon
	history.append(bestF1)



#Test set probabilities

test_probs= np.exp((((test-train_mean)/(train_std)) ** 2) * (-1/2)) / (np.sqrt(2 * np.pi) * train_std) #Features' individual probability value
test_pval = [1] * test[clmns[0]].size #Combined probability Value

for i in clmns:
	test_pval=test_pval*test_probs[i]

pred=test_pval<bestEpsilon
acc=accuracy_score(test_labels,pred)
recall=recall_score(test_labels,pred)

testF1= 2*acc*recall/(acc+recall)


	

