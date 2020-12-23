#

from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split #分训练集和测试集
import  matplotlib.pyplot as plt
import  numpy as np #要不要用就先弄上
from time import time
import datetime  #将时间戳转换为真实时间
import pandas as pd
data=load_breast_cancer()#直接导入该类中的函数，不需要实例化就能导入，属于静态函数
X=data.data
y=data.target #将数据集导入data数据中
print(X.shape)
np.unique(y)
plt.scatter(X[:,0],X[:,1],c=y)#颜色等于哪个颜色
plt.show()

#训练集和测试集
Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,y,test_size=0.3,random_state=420)
#没必要画图，画图只是显示出来了，但是不画图也是有相应的图上的数据的
Kernel=["linear","poly","rbf","sigmoid"]
#Kernel=["linear","rbf","sigmoid"]
for kernel in Kernel:
    time0=time()
    clf=SVC(kernel=kernel,gamma="auto",degree=1,
           cache_size=3500 ).fit(Xtrain,Ytrain)
    #使用相应的核函数进行初始化分类器，然后使用训练集进行训练，生成训练好的
    #分类器，此时的分类器中存在着训练集上的决策边界
    print("%s核函数在训练集上的精确度为：%f"%(kernel,clf.score(Xtest,Ytest)))
    print(datetime.datetime.fromtimestamp(time()-time0).strftime("%M:%S:%f"))
#会发生第一个，多项式核函数的时候死循环，因为多项式核函数无法求解许多特征下的，维度下的计算
#因此要去下多项式核函数的核函数，进行训练
#将多项式核函数去掉之后，循环能够正常执行完毕了，而且时间很快，训练集一定的时候，同一个模型的精确度是一样的，只是每次时间运行会不一样
#因为每次，你计算机中的内存的消耗和其他系统资源的占用和分配是不一样的，所以时间不一样
# 将degree=1设置后则继续使用多项式核函数进行训练，
data=pd.DataFrame(X)
print(data.describe([0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.99]).T)

#量纲不统一，数据存在偏态
print("标准化后的数据为：")
from sklearn.preprocessing import StandardScaler
X=StandardScaler().fit_transform(X)
data=pd.DataFrame(X)
print(data.describe([0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.99]).T)
#处理完数据后继续进行训练，然后测试数据

#训练集和测试集
Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,y,test_size=0.3,random_state=420)
#没必要画图，画图只是显示出来了，但是不画图也是有相应的图上的数据的
print("处理后的数据的作为训练集，然后测试的结果如下所示：")
Kernel=["linear","poly","rbf","sigmoid"]
#Kernel=["linear","rbf","sigmoid"]
for kernel in Kernel:
    time0=time()
    clf=SVC(kernel=kernel,gamma="auto",degree=1,
           cache_size=3500 ).fit(Xtrain,Ytrain)
    #使用相应的核函数进行初始化分类器，然后使用训练集进行训练，生成训练好的
    #分类器，此时的分类器中存在着训练集上的决策边界
    print("%s核函数在训练集上的精确度为：%f"%(kernel,clf.score(Xtest,Ytest)))
    print(datetime.datetime.fromtimestamp(time()-time0).strftime("%M:%S:%f"))
#支持向量机是在高维的数据下进行计算，所以一个标准化后的数据是至关重要的
#线性核，尤其是多项式核函数在高次项时计算非常缓慢，2rbf和多项式核函数都不擅长处理量纲不统一的数据集
#因此在执行SVM之前，非常推荐先进行数据的无量纲化
#***************对高斯径向基核函数进行调参*********************
score=[] #建立一个空列表，等下进行append
gamma_range=np.logspace(-10,1,50)
for i in gamma_range:
    clf=SVC(kernel="rbf",gamma=i,cache_size=2500).fit(Xtrain,Ytrain)
    score.append(clf.score(Xtest,Ytest))#判断rbf模型在不同的gamma数值下，所训练出的分类器，然后再测试集上进行测试，得出的精确度有什么不同
print(max(score),gamma_range[score.index(max(score))])
plt.plot(gamma_range,score)
plt.show()

#画出，不同的gamma值所对应的精确度的图形，可以得出，随着gamma值的变化，精确度的变化情况，
#测试出来的精确度就是，使用测试集然后进行判断，看看100个样本中有多少个是预测正确的，即为精确度
#从图形上可以看出，很显然在0.5左右的是，精确度最高能够大于0.95
#gamma=0.012067的时候，此时的精确度为97.6608%此时与线性核函数是一样的，
#从而可以看出，是可以通过调参可以达到优化模型的目的

#************通过网格搜索，交叉验证的方式，30%做测试，70%做训练，取平均数作为精确度
#*******************对poly多项式核函数进行调参
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
time0=time()#获取当前时间的时间戳，是从1970年返回来的秒数
gamma_range=np.logspace(-10,1,50)
coef0_range=np.linspace(0,5,10)
param_grid=dict(gamma=gamma_range,coef0=coef0_range)#将两个列表加入到一个字典中，键分别是gamma，coef0
#print(param_grid)
cv=StratifiedShuffleSplit(n_splits=5,test_size=0.3,random_state=420)
grid=GridSearchCV(SVC(kernel="poly",degree=1,cache_size=5000),param_grid=param_grid,cv=cv)
grid.fit(X,y)#进行对标准化后的数据进行训练，初始化后的模型进行训练得到grid分类器
print("最好的参数是：%s,它的精确度是%0.5f"%(grid.best_params_,grid.best_score_))#格式化输出，表示小数点后5位小数，浮点数
print(datetime.datetime.fromtimestamp(time()-time0).strftime("%M:%S:%f"))#最好的参数输出是一个字典的形式。coef0：，gamma：
#使用初始化的时间戳与现在时间的时间戳进行做差，求出来的结果就是之前的差值，就是秒数，然后
#然后再将秒数值通过srtftime函数进行转换，转为时分秒，方便看
