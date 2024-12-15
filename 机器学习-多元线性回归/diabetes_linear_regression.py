from sklearn import datasets #datasets 模块内含多个函数和方法，可以用来加载一些经典的数据集
from sklearn.model_selection import train_test_split  # 模型数据工具
from sklearn.linear_model import LinearRegression  # 模型
from sklearn.metrics import mean_squared_error # metrics（度量）指的是用来评估模型性能的一系列标准或方法

diabetes=datasets.load_diabetes()  # 加载糖尿病数据集

X=diabetes.data
Y=diabetes.target

X_train, X_test, y_train, y_test=train_test_split(X,Y,test_size=0.2)

lr=LinearRegression()

lr.fit(X_train,y_train)

y_pred_test=lr.predict(X_test)
y_pred_train=lr.predict(X_train)
print("均方误差：%.2f" % mean_squared_error(y_test,y_pred_test))
print("均方误差：%.2f" % mean_squared_error(y_train,y_pred_train))
