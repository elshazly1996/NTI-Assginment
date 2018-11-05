import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import LabelEncode,StandardScaler,train_test_split
from sklearn.linear_model import LogisticRegression

class my_model:
	def __init__(self):
		self.sc = StandardScaler()
		self.classifier = LogisticRegression(random_state = 0)
		
	def read_df(self,path):
		self.dataset = pd.read_csv(path)

	def split_df(self):
		self.x = self.dataset.iloc[:, [2, 3]].values
		self.y = self.dataset.iloc[:, 4].values

	def scaling(self):
		self.x = self.sc.fit_transform(self.x)

	def train_test(self,test_size):
		self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size = test_size, random_state = 0)

	def train(self):
		self.read_df("C:\\Users\\Abdelrahman\\Desktop\\Social_Network_Ads.csv")
		self.split_df()
		self.scaling()
		self.train_test(0.25)
		self.classifier.fit(self.x_train, self.y_train)

	def evaluate(self):
		return self.classifier.score(self.x_test,self.y_test)

	def predict(self,test):
		test = self.sc.transform([test])
		return self.classifier.predict(test)

