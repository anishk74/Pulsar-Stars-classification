import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

class Gaussian:

	def __init__(self,mean,std,columns):
		
		self.mean=mean
		self.std=std
		self.clmns=columns
		self.bestEpsilon=None

	def fit(self,train,train_labels,cv,cv_labels):

		
		train_dens= np.exp((((train - self.mean)/(self.std)) ** 2) * (-1/2)) / (np.sqrt(2 * np.pi) * self.std) #Features' individual density values
		train_dval=np.ones(train[self.clmns[0]].size) #Combined density Value for each example

		#Cross-Validation set densities
		cv_dens= np.exp((((cv-self.mean)/(self.std)) ** 2) * (-1/2)) / (np.sqrt(2 * np.pi) * self.std) #Features' individual density values
		cv_dval=np.ones(cv[self.clmns[0]].size) #Combined density Value for each example
		
		for i in self.clmns:
			cv_dval=cv_dval*(cv_dens[i].values)
			train_dval=train_dval*(train_dens[i].values)
		
		train_labels=train_labels.values
		cv_labels=cv_labels.values
		bestF1=0
		bestEpsilon=0
				
		history=[]
		
		epsilon=cv_dval[cv_dval>0].min()
		while epsilon <= cv_dval.max():

			train_pred= train_dval < epsilon
			cv_pred= cv_dval < epsilon
			acc=accuracy_score(train_labels,train_pred)
			val_acc=accuracy_score(cv_labels,cv_pred)
			
			f1 = f1_score(cv_labels,cv_pred)
			if f1>bestF1:
				bestF1=f1
				bestEpsilon=epsilon
			history.append([acc,val_acc,f1])
			epsilon*=2
		
		self.bestEpsilon=bestEpsilon
		print('F1-score of the model on CV data: ',bestF1)

		history=pd.DataFrame(history,columns=['accuracy','val_accuracy','f1'])
		return history


		
	def evaluate(self,test,test_labels):

		if(self.bestEpsilon==None):
			print('!!!!')
			print('Model still to be trained.... Use model.fit()')
			return None,None

		test_dens= np.exp((((test - self.mean)/(self.std)) ** 2) * (-1/2)) / (np.sqrt(2 * np.pi) * self.std) #Features' individual probability value
		test_dval = np.ones(test[self.clmns[0]].size) #Combined probability Value

		for i in self.clmns:
			test_dval=test_dval*test_dens[i]

		pred=test_dval<self.bestEpsilon
		acc=accuracy_score(test_labels,pred)
		f1=f1_score(test_labels,pred)
		return acc,f1

	def predict(self,features):
		if(self.bestEpsilon==None):
			print('!!!!')
			print('Model still to be trained.... Use model.fit()')
			return None,None
		feat_dens= np.exp((((feat - self.mean)/(self.std)) ** 2) * (-1/2)) / (np.sqrt(2 * np.pi) * self.std) #Features' individual probability value
		feat_dval = np.ones(feat[self.clmns[0]].size) #Combined probability Value
		for i in self.clmns:
			feat_dval=feat_dval*feat_dens[i]



