import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


class read_file:
	def __init__(self, data_file):
		self.data_file = data_file
		
	def read(self):
		df = pd.read_csv(self.data_file)
		return df


class data_to_num:
	def __init__(self, data, feature):
		self.data = data
		self.feature = feature

	def convert(self):

		data_groupby = self.data.groupby(self.feature)[self.feature]
		
		final_data = []
		self.data_name_num = []
		data_num = []
		data_name = []
		cpt = 0
		
		for i in data_groupby:
			cpt = cpt+1
			self.data_name_num.append([i[0], cpt])
			data_name.append(i[0])
			data_num.append(cpt)
				
		for i in self.data[self.feature]:
			for j in self.data_name_num:
				if i == j[0]:
					self.data[self.feature] = self.data[self.feature].replace(i, j[1])
		
		self.data[self.feature] = self.data[self.feature].fillna(0)
		self.data['Age'] = self.data['Age'].fillna(0)
		self.data['Fare'] = self.data['Fare'].fillna(0)


class feature_target:
	def __init__(self, df, feature, target):
		self.df = df
		self.feature = feature
		self.target = target
	
	def feature(self):
		self.x = self.df[self.feature]

	def target(self):
		self.y = self.df[self.target]


class logistic_regression:
	def __init__(self, x_train, y_train, x_test):
		self.x_train = x_train
		self.y_train = y_train
		self.x_test = x_test
	
	def predict(self):
		model = LogisticRegression()
		model.fit(self.x_train, self.y_train)
		y_pred = model.predict(self.x_test)
		print(model.coef_)
		print('\n')
		print(self.x_train.columns)
		
		return y_pred


class kfold:
	def __init__(self, n_splits, X, y, random):
		self.n_splits = n_splits
		self.X = X
		self.y = y
		self.random = random
	
	def splits(self):
		kf = KFold(n_splits = self.n_splits, shuffle = True, random_state = self.random)
		splits = list(kf.split(self.X, self.y))
		train_test = []
		for i in splits:
			train, test = i
			x_train = self.X.iloc[train]
			x_test = self.X.iloc[test]
			
			y_train = self.y.iloc[train]
			y_test = self.y.iloc[test]
			
			train_test.append([x_train, x_test, y_train, y_test])
			
		return train_test


class decision_tree_classifier:
	def __init__(self, x_train, x_test, y_train):
		self.x_train = x_train
		self.x_test = x_test
		self.y_train = y_train

	def prediction(self):
		model = DecisionTreeClassifier()
		model.fit(self.x_train, self.y_train)
		y_pred = model.predict(x_test)
		
		return y_pred
	
	def gridSearch(self, max_depth, min_sample_leaf, max_leaf_node, cv):
		param_grid ={	'max_depth':max_depth,
						'min_samples_leaf':min_sample_leaf,
					    'max_leaf_nodes': max_leaf_node
					}
		
		dt = DecisionTreeClassifier()
		gs = GridSearchCV(dt, param_grid, scoring = 'accuracy', cv = cv)
		print(gs.estimator.get_params().keys())
		gs.fit(self.x_train, self.y_train)

		pred = gs.predict(self.x_test)
		print('best parameter : ',gs.best_params_)
		best_score = gs.best_score_
		print('best score : ',best_score)
		
		return pred 


class random_forest:
	def __init__(self, x_train, x_test, y_train):
		self.x_train = x_train
		self.x_test = x_test
		self.y_train = y_train
		
	def prediction(self):
		rf = RandomForestClassifier()
		rf.fit(self.x_train, self.y_train)
		pred = rf.predict(x_test)
		
		return [pred, best_score]
	
	def gridSearch(self, n_estimators, cross_valid):
		param_grid = {
						'n_estimators':n_estimators
						}
		rf = RandomForestClassifier()
		gs = GridSearchCV(rf, param_grid, scoring = 'accuracy', cv = cross_valid)
		gs.fit(self.x_train, self.y_train)

		pred = gs.predict(self.x_test)
		best_score = gs.best_score_
		print('best score : ',best_score)
		return [pred, best_score]


class neural_network:
	def __init__(self, x_train, y_train, x_test):
		self.x_train = x_train
		self.y_train = y_train
		self.x_test = x_test
	
	def predict(self, max_iter, hidden_layer, alpha, random):
		mlp = MLPClassifier(max_iter=max_iter, hidden_layer_sizes=hidden_layer, alpha=alpha, random_state=random)
		mlp.fit(self.x_train, self.y_train)
		pred = mlp.predict(self.x_test)
		return pred
		
		
class calcul_metrics:
	def __init__(self, y_test, y_pred, index_data):
		self.y_test = y_test
		self.y_pred = y_pred
		self.index_data = index_data
		
	def accuracy(self):
		accuracy = accuracy_score(self.y_test, self.y_pred)
		return [self.index_data, accuracy, self.y_pred]
	
	def f1(self):
		f1 = f1_score(self.y_test, self.y_pred)
		return [self.index_data, f1, self.y_pred]
	
	def precision(self):
		precision = precision_score(self.y_test, self.y_pred)
		return [self.index_data, precision, self.y_pred]
	
	def recall(self):
		recall = recall_score(self.y_test, self.y_pred)
		return [self.index_data, recall, self.y_pred]


class max_metrics:
	def __init__(self, metrics_list):
		self.metrics_list = metrics_list
		
	def find_max(self):
		max_metric = -1
		data_maxMetric = []
		for i in self.metrics_list:
			if max_metric < i[1]:
				max_metric = i[1]
				data = i[0]
				y_pred = i[2]
		data_maxMetric.append([data, max_metric])
		return [data, max_metric, y_pred]		
