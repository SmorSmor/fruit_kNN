#!/usr/bin/python
# -*- coding:utf-8 -*-

# 导入必要的包
import numpy as np  # 科学计算库
import pandas as pd  # 把txt,csv转换成表格
import matplotlib.pyplot as plt  # 可视化工具
from sklearn.model_selection import train_test_split  # 分割数据集
from sklearn.neighbors import KNeighborsClassifier  # kNN分类器

# 一、加载数据集 pd.read_table("******")

fruit_df = pd.read_table("fruit_data_with_colors.txt")
print(fruit_df.head())  # 前5行
print("\n\n--------------------------------------\n\n")
print("样本个数= ", len(fruit_df))
print("\n\n--------------------------------------\n\n")

# 创建目标字典和名称的字段
fruit_name_dict = dict(zip(fruit_df["fruit_label"], fruit_df["fruit_name"]))
print(fruit_name_dict)
print("\n\n--------------------------------------\n\n")

"""
1.建立特征矩阵
2.建立标签矩阵
"""
X = fruit_df[["mass", "width", "height", "color_score"]]  # 建立特征矩阵 59行 * 45列的矩阵
y = fruit_df["fruit_label"]  # 标签矩阵  59行* 1列


# 二、分割数据集 #train_test_split(x , y ,test_size = 1/4 ,random_state = 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=0)

print("原始数据集:{} , 训练集:{} , 测试集:{}".format(len(X), len(X_train), len(X_test)))
print("\n\n--------------------------------------\n\n")

from mpl_toolkits.mplot3d import Axes3D  # 绘制3d图

label_color_dict = {1: "red", 2: "green", 3: "blue", 4: "yellow"}  # 给标签赋予颜色
colors = list(map(lambda label: label_color_dict[label], y_train))
fig = plt.figure()  # 初始化画板
ax = fig.add_subplot(111, projection="3d")  # 指定子图的格式 ，XYZ三维坐标
ax.scatter(X_train["width"], X_train["height"], X_train["color_score"], c=colors, marker="o", s=100)

ax.set_xlabel("width")
ax.set_ylabel("height")
ax.set_zlabel("color_score")
plt.show()
# 三、选择/建立模型 kNN_model = KNeighborsClassifier(n_neighbors=1/3/5)
knn = KNeighborsClassifier(n_neighbors=5)  # 超参数k=5


# 四、训练模型 kNN_model.fit(X_train , y_train)

knn.fit(X_train, y_train)  # 此刻训练模型已经ok了
# 五、验证模型  y_hat = kNN_model.predict(x_test )

y_pred = knn.predict(X_test)
print("k=1时 \n", y_pred)

k_range = range(1, 20)  # 1~19

acc_score = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    acc_score.append(knn.score(X_test, y_test))

plt.figure()
plt.xlabel("K")
plt.ylabel("accuracy")  # 准确率
plt.scatter(k_range, acc_score)
plt.xticks([0, 5, 10, 15, 20])  # 间隔
plt.show()
