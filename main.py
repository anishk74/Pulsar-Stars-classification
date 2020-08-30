import pandas as pd
import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt

import seaborn as sns

dataset=pd.read_csv('pulsar_stars.csv')



featureCount=0
columns=['x'+i for i in range(len(dataset.columns)-1)]
featurcount=i

dataset.columns=columns






frames = [frame for _,frame in dataset.groupby('target_class')]

negds=frames[0]
posds=frames[1]

#labels=dataset.pop('target_class')


'''
train_ds = negds.sample(frac=0.8,random_state=0)
test_ds = negds.drop(train_ds.index)
'''

train_ds, val_ds, test_ds = np.split(negds.sample(frac=1), [int(.7*len(negds)), int(.85*len(negds))])
width=len(clmn)//2

'''
plt.figure(figsize=(5,20))
count=1
for i in clmn:
	plt.subplot(2,4,count)
	plt.xlabel(i)
	dataset[i].hist()
	count+=1
plt.show()
count=1
for i in clmn:
	plt.subplot(2,4,count)
	plt.xlabel(i)
	negds[i].hist()
	count+=1
plt.show()
'''


'''
for i in clmn:
	count=1
	plt.figure(figsize=(45,5))
	plt.xlabel(i)
	for j in clmn:
		plt.subplot(1,8,count)
		plt.scatter(i,j,data=negds,s=1)
		plt.scatter(i,j,data=posds,s=1)
		plt.xlabel(i)
		plt.ylabel(j)
		count+=1
	plt.savefig(i+'.png')

plt.figure(figsize=(40,35))
count=1

for i in clmn:
	
	
	plt.xlabel(i)
	for j in clmn:
		plt.subplot(8,8,count)
		plt.scatter(i,j,data=negds,s=1)
		plt.scatter(i,j,data=posds,s=1)
		plt.xlabel(i)
		plt.ylabel(j)
		count+=1

plt.savefig('final'+'.png')
'''



'''
for i in clmn:
	count=1
	plt.figure(figsize=(20,5))
	for j in clmn[:width]:
		plt.subplot(1,width,count)
		plt.scatter(i,j,data=negds,s=1)
		#plt.scatter(i,j,data=posds,s=1)
		plt.ylabel(j)
		count+=1
	plt.show()

	count=1
	plt.figure(figsize=(20,5))
	for j in clmn[width:]:
		plt.subplot(1,len(clmn)-width,count)
		plt.scatter(i,j,data=negds,s=1)
		#plt.scatter(i,j,data=posds,s=1)
		count+=1

	plt.show()
'''
