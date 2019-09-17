#!/usr/bin/python
# -*- coding:utf-8 -*-
# 导入必要的包
import numpy as np  # 科学计算库
import pandas as pd  # 把txt转换成表格
import matplotlib.pyplot as plt  # 可视化工具库

from sklearn.model_selection import train_test_split  # 分割数据集
from sklearn.neighbors import KNeighborsClassifier  # kNN分类器

# 一、加载数据集
# pd.read_table("********")
# 二、分隔数据集
# train_test_split(X, y, test_size=1 / 4, random_state=0)
# 三、选择 / 建立模型
# KNeighborsClassifier(n_neighbors=1)
# 四、训练模型
# kNN_model.fit(X_train, y_train)
# 五、验证模型
# kNN_model.predict(X_test)

