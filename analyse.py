import Class
import pandas as pd
#pd.options.display.max_rows = 1000
x_feature = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Cabin', 'Fare']
y_target = 'Survived'

train_file = 'train.csv'
test_file = 'test.csv'

data = Class.read_file(train_file)
train = data.read()

data = Class.read_file(test_file)
test = data.read()

## conversion data to numeric 
 
train['Sex'] = train['Sex'] == 'male'
test['Sex'] = test['Sex'] == 'male'

convert_train = Class.data_to_num(train, 'Cabin')
convert_train.convert()

convert_test = Class.data_to_num(test, 'Cabin')
convert_test.convert()


## *********************


## build feature and target

ft_train = Class.feature_target(train, x_feature, y_target)
ft_test = Class.feature_target(test, x_feature, y_target)

x_train = train[x_feature]
y_train = train[y_target]
#print(y_train)
x_test = test[x_feature]
#print(x_test)
## ***************************

## decision tree classifier
'''dt = Class.decision_tree_classifier(x_train, x_test, y_train)
max_depth= list(range(1, 30))
min_sample_leaf = list(range(1, 10))
max_leaf_node = list(range(10, 40))
cv = 5
dt_pred = dt.gridSearch(max_depth, min_sample_leaf, max_leaf_node, cv)


## *****************************


## random forest classifier
rf = Class.random_forest(x_train, x_test, y_train)

n_estimators = list(range(1, 101))
cross_valid = 5
rf_pred = rf.gridSearch(n_estimators, cross_valid)

if rf_pred[1] > dt_pred[1]:
	pred = rf_pred[0]
else:
	pred = dt_pred[0]

test['Survived'] = pred
test[['PassengerId','Survived']].to_csv('prediction.csv', index=False)'''


## logistic regression

lg = Class.logistic_regression(x_train, y_train, x_test)
lg.predict()
