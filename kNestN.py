#!/usr/bin/python
# -*- coding:utf-8 -*-
# 导入必要的包
import numpy as np  # 科学计算库
import pandas as pd  # 把txt转换成表格
import matplotlib.pyplot as plt  # 可视化工具库

from sklearn.model_selection import train_test_split  # 分割数据集
from sklearn.neighbors import KNeighborsClassifier  # kNN分类器
from sympy.physics.quantum.tests.test_circuitplot import mpl

mpl.rcParams['font.sans-serif'] = [u'SimHei'] #黑体
mpl.rcParams['axes.unicode_minus'] = False #默认输出

# 一、加载数据集
# pd.read_table("********")
fruit_df = pd.read_table("fruit_data_with_colors.txt")
print(fruit_df.head(10))
print("\n\n--------------------------\n\n")
print("样本个数=", len(fruit_df))
print("\n\n--------------------------\n\n")

# 创建目标标签和名称的字典
fruit_name_dict = dict(zip(fruit_df["fruit_label"], fruit_df["fruit_name"]))
print(fruit_name_dict)
print("\n\n--------------------------\n\n")

X = fruit_df[["mass", "width", "height", "color_score"]]
y = fruit_df["fruit_label"]

# 二、分隔数据集
# train_test_split(X, y, test_size=1 / 4, random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=0)

print("原始数据集：{}， 训练集：{}， 测试集：{}".format(len(X), len(X_train), len(X_test)))
print("\n\n--------------------------\n\n")

from mpl_toolkits.mplot3d import Axes3D  # 绘制3d图

label_color_dict = {1: "red", 2: "green", 3: "blue", 4: "yellow"}
colors = list(map(lambda label: label_color_dict[label], y_train))
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(X_train["width"], X_train["height"], X_train["color_score"], c=colors, marker="*", s=50)

ax.set_xlabel("width")
ax.set_ylabel("height")
ax.set_zlabel("color_score")
plt.show()



# 三、选择 / 建立模型
# KNeighborsClassifier(n_neighbors=1)

kNN = KNeighborsClassifier(n_neighbors=5)

# 四、训练模型
# kNN_model.fit(X_train, y_train)

kNN.fit(X_train, y_train)

# 五、验证模型
# kNN_model.predict(X_test)


y_pred = kNN.predict(X_test)
print("k=1时 \n", y_pred)

k_range = range(1, 20)  # 1~19

acc_score = []
for k in k_range:
    kNN = KNeighborsClassifier(n_neighbors=k)
    kNN.fit(X_train, y_train)
    acc_score.append(kNN.score(X_test, y_test))

plt.figure()
plt.xlabel("K", fontsize=18)
plt.ylabel("accuracy", fontsize=18)  # 准确率
plt.title(u"水果分类器", fontsize=20)
plt.scatter(k_range, acc_score)
plt.xticks([0, 5, 10, 15, 20])  # 间隔
plt.show()
