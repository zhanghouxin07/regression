#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  dubins_path_regressin.py
#  
#  Copyright 2019 zhx <zhx@zhx123>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import graphviz
from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn import ensemble

data_all = [[120,15, 6000,10,0.390], [240,15, 6000,10,0.341], [120,25, 6000,10,0.378],
            [240,25, 6000,10,0.286], [120,15,10000,10,0.381], [240,15,10000,10,0.297],
            [120,25,10000,10,0.468], [240,25,10000,10,0.292], [120,15, 6000,20,0.459],
            [240,15, 6000,20,0.423], [120,25, 6000,20,0.418], [240,25, 6000,20,0.374],
            [120,15,10000,20,0.409], [240,15,10000,20,0.357], [120,25,10000,20,0.462],
            [240,25,10000,20,0.379], [120,20, 8000,15,0.462], [240,20, 8000,15,0.412],
            [180,15, 8000,15,0.441], [180,25, 8000,15,0.369], [180,20, 6000,15,0.393],
            [180,20,10000,15,0.415], [180,20, 8000,10,0.329], [180,20, 8000,20,0.432],
            [180,20, 8000,15,0.397], [180,20, 8000,15,0.397], [180,20, 8000,15,0.414],
            [180,20, 8000,15,0.400], [180,20, 8000,15,0.404], [180,20, 8000,15,0.404]]

def try_different_method(x_train, y_train, x_test, model):
    model.fit(x_train, y_train)
    result = model.predict(x_test)
    #print(result)
    # tree_file = tree.export_graphviz(model,
    #                                  out_file='tree.dot',
    #                                  feature_names=['grindhead', 'polishing', 'principal', 'feedrate'],
    #                                  filled=True,
    #                                  rounded=True,
    #                                  special_characters=True)
    # graph = graphviz.Source(tree_file)
    return result

        # 读取文件
f = open('length.txt', 'r')
def readfiles():
    
    data = f.readlines()
    for i in range(0, len(data)):
        data[i] = data[i].rstrip('\n')
        data[i] = data[i].split()
    data_dubins = []
    print(data)
    length = len(data)
    length = int(length)
    for i in range(length):
        tmp = []
        for j in range(7):
            tmp.append(data[j])
        data_dubins.append(tmp)
    print('the number of element in list',len(data_dubins))
    print('read ok!')
    print(len(data_dubins))
    return data
    
def main(args):
    return 0

if __name__ == '__main__':
    # 模型选择
    model_name = 'Adaboost'
    # a.线性回归
    if model_name == 'MultiLinear':
        model = LinearRegression()
    # b.决策树回归
    elif model_name == 'DecisionTree':
        model = tree.DecisionTreeRegressor()
    # c.随机森林回归
    elif model_name == 'RandomForest':
        model = ensemble.RandomForestRegressor(n_estimators=40) # 使用20个决策树
    # d.Adaboost回归
    elif model_name == 'Adaboost':
        model = ensemble.AdaBoostRegressor(n_estimators=200)  # 使用20个决策树
        
    data_all = readfiles()
    # readfiles()
    # 回归拟合
    score = 0
    predict = []
    target = []
    # for i in range(10):
        # data_test = data_all[i*3:(i+1)*3]
        # data_train = data_all.copy()
        # for i in data_test:
            # data_train.remove(i)
        # data_train = np.array(data_train)
        # data_test = np.array(data_test)
        # x_train, y_train = data_train[:, :4], data_train[:, 4]
        # x_test, y_test = data_test[:, :4], data_test[:, 4] 

        # # train的前4列是x，后一列是y
        # predict_i = try_different_method(x_train, y_train, x_test, model)
        # predict.extend(predict_i)
        # target.extend(y_test)
        # score_i = np.sum(predict_i-y_test)**2
        # score += score_i
    loop_toge = int(len(data_all)/5)
    loop_toge = 1
    for i in range(loop_toge):
        data_test = data_all[i*5:(i+100)*5]
        data_train = data_all.copy()
        for i in data_test:
            data_train.remove(i)
        data_train = np.array(data_train)
        data_test = np.array(data_test)
        x_train, y_train = data_train[:, :6], data_train[:, 6]
        x_test, y_test = data_test[:, :6], data_test[:, 6]
        # train的前6列是x，后1列是y
        x_train = x_train.astype('float64')
        y_train = y_train.astype('float64')
        x_test = x_test.astype('float64')
        predict_i = try_different_method(x_train, y_train, x_test, model)
        predict.extend(predict_i)
        target.extend(y_test)
        error = []
        for i in range(len(predict)):
            error.append(float(predict[i]) - float(target[i]))
        print(error)
        #print(predict_i-y_test)
        #print(y_test)
        tmp = 0
        # print(float(predict_i[0][0])-float(y_test[0][0]))
        # for j in range (len(predict_i)):
            # for k in range(len(predict_i[j])):
                # product_tmp = float(predict_i[j][k])-float(y_test[j][k])
                # tmmp = product_tmp*product_tmp
                # tmp += tmmp
        # #score_i = np.sum(predict_i-y_test)**2
        # score_i = tmp
        # score += score_i
    # 
    # 可视化
    plt.figure()
    plt.xlabel('number of samples')
    plt.ylabel('error')
    #plt.plot(np.arange(len(predict_once)), predict_once, "ro-", label="Predict value")
    plt.plot(np.arange(len(error)), error, "go-", label="True value")
    newticks = np.linspace(-5, 5, 9)
    plt.yticks(newticks)
    plt.title("method:{}---score:{:.6f}".format(model_name, score))
    #plt.legend(loc="best")
    plt.show()
    #print(len(data_test))
    
    
    f.close()
