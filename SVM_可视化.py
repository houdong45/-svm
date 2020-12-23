#导入需要导入的模块，必须自己代码练一遍，才能够深刻的了解支持向量机
#简单的线性支持向量机的可视化(基础一些，然后之后在敲不同核函数的代码)
from sklearn.datasets import make_blobs #使用云版本可以增大运算的速度
from sklearn.svm import  SVC #导入常数svc
import matplotlib.pyplot as plt
import  numpy as np #导入常用的科学数值库，python中的基本库，更深刻一些的是panda数据处理与可视化的专用库

#实例化数据集，可视化数据集
X,y=make_blobs(n_samples=50,centers=2,random_state=0,cluster_std=0.6)#两个中心，50个样本
#会自动出现相应的参数的位置，为创建数据集的类进行参数实例化操作
print(X.shape)
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap="rainbow") #把每个样本点的横坐标作为x，纵坐标作为y进行画图，两个列表一一对应x,y就可以画出来
#plt.xticks([]) #显示为空，不显示任何的数
#plt.yticks([])#把这个空取消然后就有坐标了
#plt.show()  #反应有点慢，不过是多行注释CTRL+/

#画决策边界，制作网格，理解函数meshgrid ,自动显示行

#首先要有散点图
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap='rainbow')
ax=plt.gca()#获得当前的子图，如果不存在，则创建新的子图，意思为如果没有该散点图的话就会创建一个子图，不过现在有图，就是之前画的散点图，ax为一个新对象
#获取平面上两条坐标轴的最大值和最小值
xlim=ax.get_xlim()
ylim=ax.get_ylim()
print(xlim)
print(ylim)
#在最大值和最小值之间形成30个规律的数据
axisx=np.linspace(xlim[0],xlim[1],30) #形成30个有规律的数据
axisy=np.linspace(ylim[0],ylim[1],30) #该函数使用从开始到结束，中间取30个数作为y，同理x
print(axisx)#一维数组
print(axisy)
axisy,axisx=np.meshgrid(axisy,axisx)  #画决策边界，制作
#我们将使用这里形成的二维数组作为我们contour函数中的x和y
#使用meshgrid函数将两个一维向量转换为特征矩阵
#核心是将两个特征向量广播，以便获取y.shape *x.shape这么多个坐标点二点横坐标和纵坐标
print(axisx)
print(axisx.shape) #30x30的维数，实际上也是900个点
#使用ravel()函数进行拉伸
print(axisx.ravel().shape) #python中不用保存代码，直接运行就是修改后的代码，很方便
xy=np.vstack([axisx.ravel(),axisy.ravel()]).T
#其中ravel()是降维函数，vstack能够将多个结构一致的一维数组进行按行堆叠起来。
#xy就是已经形成的网络，它是遍布在整个画布上的密集的点
print(xy.shape)#此时变成了900个，然后每个数据里面有两个元素
#开始绘画网格
#重新取在画布上取得的各个点，进行绘画，xy作为一个一维数据
plt.scatter(xy[:,0],xy[:,1],s=1,cmap="rainbow")#：代表的数据是从第一个数据到所有的数据，进行绘制，取第一个数据的，第一个元素和第二个元素
#plt.show()
#直到取出所有的点，然后绘制所有的点
                      ###****************############
# #理解函数meshgrid和vstack的作用
# a=np.array([1,2,3])#3x1,3行1列
# b=np.array([7,8])
# #两两组合，会得到多少个坐标？
# #答案是六个，分别组合
# v1,v2=np.meshgrid(a,b)
# print(v1.shape) #变成2x3的矩阵了，
# print(v2.shape)
# print(v1)#
# print(v2)
# #使用ravel函数进行降维处理,v1现在是2行3列的矩阵
# print(v1.ravel())
# print(v2.ravel())
# v=np.vstack([v1.ravel(),v2.ravel()]).T#[]相当于是传入一个数组，函数vstack里面需要两个参数，x,y,拉伸后的值，然后将结果进行转置，赋值给V
# print(v)
#
#      ###meshgrid和vstack的作用就是把两个特征向量，把一个当作横坐标，另一个当作纵坐标，组合成一个坐标点（这些坐标点被用来画图）
#      ###理解其函数的作用

#使用样本点来训练得出决策边界，然后使用网格点来将决策边界画出

##5 建模，计算决策边界并找出网格上每个点到决策边界的距离
#建模，通过fit计算出对应的决策边界,使用线性核函数进行映射，从而找在该样本集下的模型
#写的时候把项目列表给隐藏了，然后等使用的时候在点开就OK了,前几天看，只看内容了没有自己手写代码



clf=SVC(kernel="linear").fit(X,y) #fit的本质就是在寻找决策边界
Z=clf.decision_function(xy).reshape(axisx.shape)#用计算出来的决策边界的模型clf进行调用相应的函数
#重要接口decision_function,返回每个输入的样本所对应的到决策边界的距离
#然后再将这个距离转换为axisx的结构，这是由于画图的函数contour要求Z的结构必须与X和Y保持一致
ax.contour(axisx,axisy,Z,colors="k",levels=[-1,0,1],alpha=0.5,linestyles=["--","-","--"])
#有的函数没有提示，必须自己写上自己的东西
ax.set_xlim(xlim)                         #训练好的分类器，就是通过一个模型实例化一个分类器，然后使用训练集进行训练该分类器，训练完之后就可以得出决策边界了，clf就是训练好的分类器，可以直接调用接口
ax.set_ylim(ylim) #将子图上的坐标设置为相应的坐标
plt.show()  #先使用线性核函数模型初始化一个分类器类，然后调用分类器的fit函数进行训练自己的数据集
           #训练完毕后就得到了自己数据集的分类器，此时的clf就是分类器
           #然后利用分类器中的各种函数就可以实现分类了，而训练好的分类器中已经通过库函数求出来了训练样本的决策边界（也就是通过拉格朗日函数和对偶问题变换求出来模型的w,b了）
           #也就是求出来模型了，然后通过外界的样本或者数据集就可以进行分类了。
           #这就是支持向量机的线性模型的应用
           #确实有图片就可以显示 clf是已经训练好的分类器

#现在查看训练好的分类器clf中有多少个属性和接口（函数），接口就是函数，、属性就是类中的成员,在cell里面就是可以直接输出一个变量，不用写输出函数即可出来结果，而在pycharm中则需要print函数
print(clf.predict(X))#根据决策边界，对X中的样本进行分类，返回的结构为n_samples
print(clf.score(X,y))#返回给定测试数据和标签的平均准确度
print(clf.support_vectors_)#返回支持向量的各个横纵坐标
print(clf.n_support_)#返回每个类中支持向量的个数，第一类中有一个，第二类中有两个
