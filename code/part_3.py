# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 17:23:32 2020

@author: Ningt
"""
import pandas as pd

datapath = 'moment.csv'
data = pd.read_csv(datapath,encoding = 'gbk')
data = data.values

#划分训练集和测试集
#cross_validation在sklearn0.20中改为model_selection
from sklearn.model_selection  import train_test_split
train, test, train_target, test_target = train_test_split(data[:,0:],data[:,-1],test_size=0.2)
train_target = train_target.astype(int)
test_target = test_target.astype(int)

#构建SVM模型
from sklearn import svm
model = svm.SVC()
model.fit(train*30,train_target)

#save model
from sklearn.externals import joblib
joblib.dump(model,'svcmodel.pkl')

#read model
#model = joblib.load('svcmodel.pkl')

#混淆矩阵
from sklearn import metrics
cm_train = metrics.confusion_matrix(train_target, model.predict(train*30))
cm_test = metrics.confusion_matrix(test_target, model.predict(test*30))

train_accuracy = metrics.accuracy_score(train_target,model.predict(train*30))
test_accuracy = metrics.accuracy_score(test_target,model.predict(test*30))

print("train accuracy: %f"% train_accuracy) #1.000
print("test accuracy: %f"% test_accuracy) #0.9756

tr = pd.DataFrame(cm_train,index = range(1,6),columns = range(1,6)).to_excel('train.xls')
te = pd.DataFrame(cm_test,index = range(1,6),columns = range(1,6)).to_excel('test.xls')
