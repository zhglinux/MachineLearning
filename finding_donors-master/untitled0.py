#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 12:05:33 2018

@author: cocozhou
"""


# 检查你的Python版本
from sys import version_info
if version_info.major != 2 and version_info.minor != 7:
    raise Exception('请使用Python 2.7来完成此项目')
    
    
# 为这个项目导入需要的库
import numpy as np
import pandas as pd
from time import time
from IPython.display import display # 允许为DataFrame使用display()

# 导入附加的可视化代码visuals.py
import visuals as vs

# 为notebook提供更加漂亮的可视化
#%matplotlib inline


# 导入人口普查数据
data = pd.read_csv("census.csv")

# 成功 - 显示第一条记录
display(data.head(n=1))

# TODO：总的记录数
n_records = data.shape[0]

# TODO：被调查者的收入大于$50,000的人数
#print  data[data['income'] == '>50K']
n_greater_50k = data[data['income'] == '>50K'].shape[0]

# TODO：被调查者的收入最多为$50,000的人数
n_at_most_50k = data[data['income'] == '<=50K'].shape[0]

# TODO：被调查者收入大于$50,000所占的比例
greater_percent = ((float)(n_greater_50k)/(float)(n_records) *100)

# 打印结果
print "Total number of records: {}".format(n_records)
print "Individuals making more than $50,000: {}".format(n_greater_50k)
print "Individuals making at most $50,000: {}".format(n_at_most_50k)
print "Percentage of individuals making more than $50,000: {:.2f}%".format(greater_percent)

# 将数据切分成特征和对应的标签
income_raw = data['income']
print income_raw
features_raw = data.drop('income', axis = 1)
print features_raw

# 可视化 'capital-gain'和'capital-loss' 两个特征
vs.distribution(features_raw)


# 对于倾斜的数据使用Log转换
skewed = ['capital-gain', 'capital-loss']
features_raw[skewed] = data[skewed].apply(lambda x: np.log(x + 1))

# 可视化对数转换后 'capital-gain'和'capital-loss' 两个特征
vs.distribution(features_raw, transformed = True)

from sklearn.preprocessing import MinMaxScaler

# 初始化一个 scaler，并将它施加到特征上
scaler = MinMaxScaler()
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
features_raw[numerical] = scaler.fit_transform(data[numerical])

# 显示一个经过缩放的样例记录
display(features_raw.head(n = 10))


# TODO：使用pandas.get_dummies()对'features_raw'数据进行独热编码
features = pd.get_dummies(features_raw)

# TODO：将'income_raw'编码成数字值
income = income_raw.map(lambda x: 1 if x == '>50K' else 0)

# 打印经过独热编码之后的特征数量
encoded = list(features.columns)
print "{} total features after one-hot encoding.".format(len(encoded))

# 移除下面一行的注释以观察编码的特征名字
print encoded
print features
print income
display(features.head(n = 2))

# 导入 train_test_split
from sklearn.model_selection import train_test_split

# 将'features'和'income'数据切分成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features, income, test_size = 0.2, random_state = 0, 
                                                    stratify = income)
# 将'X_train'和'y_train'进一步切分为训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0, 
                                                  stratify = y_train)

# 显示切分的结果
print "Training set has {} samples.".format(X_train.shape[0])
print "Validation set has {} samples.".format(X_val.shape[0])
print "Testing set has {} samples.".format(X_test.shape[0])

#不能使用scikit-learn，你需要根据公式自己实现相关计算。

#TODO： 计算准确率
accuracy = 1 - 0.56

# TODO： 计算查准率 Precision
precision = 0.4

# TODO： 计算查全率 Recall
recall = 0.6

beta = 0.5

# TODO： 使用上面的公式，设置beta=0.5，计算F-score
fscore = (1+beta*beta)*((precision*recall)/(beta*beta*precision + recall))

# 打印结果
print "Naive Predictor on validation data: \n \
    Accuracy score: {:.4f} \n \
    Precision: {:.4f} \n \
    Recall: {:.4f} \n \
    F-score: {:.4f}".format(accuracy, precision, recall, fscore)
    
    


# TODO：从sklearn中导入两个评价指标 - fbeta_score和accuracy_score
from sklearn.metrics import fbeta_score, accuracy_score

def train_predict(learner, sample_size, X_train, y_train, X_val, y_val): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_val: features validation set
       - y_val: income validation set
    '''
    
    results = {}
    print "===>>>>>>----------3------"
    # TODO：使用sample_size大小的训练数据来拟合学习器
    # TODO: Fit the learner to the training data using slicing with 'sample_size'
    start = time() # 获得程序开始时间
#     learner = learner.fit(X_train.head(sample_size), y_train.head(sample_size))

    
    learner = learner.fit(X_train.head(sample_size), y_train.head(sample_size))
    end = time() # 获得程序结束时间
    
    
    print "===>>>>>>----------4------"
    
    # TODO：计算训练时间
    results['train_time'] = end - start 
    
    # TODO: 得到在验证集上的预测值
    #       然后得到对前300个训练数据的预测结果
    start = time() # 获得程序开始时间
    predictions_val = learner.predict(X_val)
    predictions_train = learner.predict(X_train.head(300))
    end = time() # 获得程序结束时间
    
    # TODO：计算预测用时
    results['pred_time'] = end - start
            
    # TODO：计算在最前面的300个训练数据的准确率
    results['acc_train'] = accuracy_score(y_train.head(300), predictions_train)
        
    # TODO：计算在验证上的准确率
    results['acc_val'] = accuracy_score(y_val, predictions_val)
    
    # TODO：计算在最前面300个训练数据上的F-score
    results['f_train'] =  fbeta_score(y_train.head(300),predictions_train, beta = 0.5 )
        
    # TODO：计算验证集上的F-score
    results['f_val'] = fbeta_score(y_val, predictions_val, beta = 0.5)
       
    # 成功
    print "{} trained on {} samples.".format(learner.__class__.__name__, sample_size)
        
    # 返回结果
    return results



# TODO：从sklearn中导入三个监督学习模型
from sklearn import svm  
from sklearn.naive_bayes import GaussianNB 
from sklearn import neighbors
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeRegressor

# TODO：初始化三个模型
clf_A = LinearSVC(random_state=0)  # GaussianNB() #svm.SVC(random_state = 0)
clf_B = LinearSVC(random_state=0)  #GaussianNB() # svm.SVC(random_state = 0) #GaussianNB()
#clf_C = LinearSVC(random_state=0)  #GaussianNB() #svm.SVC(random_state = 0) #neighbors.KNeighborsClassifier()
# logit_clf = LogisticRegression(penalty='l2')
clf_C = DecisionTreeRegressor(random_state=5)
    
print  X_train.shape[0]
 
# TODO：计算1%， 10%， 100%的训练数据分别对应多少点
samples_1 = (X_train.shape[0])/100  # int (0.01 * X_train.shape[0])
samples_10 =  (X_train.shape[0])/10  #  int (0.1 * X_train.shape[0])
samples_100 =   (X_train.shape[0])/1  #int (1* X_train.shape[0])

print  samples_1

clf_A.fit(X_train, y_train)


# 收集学习器的结果
results = {}
for clf in [clf_A, clf_B, clf_C]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    print "===>>>>>>----------1------"
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        print "===>>>>>>----------2------"
        results[clf_name][i] = train_predict(clf, samples, X_train, y_train, X_val, y_val)

# 对选择的三个模型得到的评价结果进行可视化
vs.evaluate(results, accuracy, fscore)

