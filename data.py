import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
import random
# %matplotlib inline
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
plt.rcParams['axes.unicode_minus']=False   #这两行需要手动设置

# 生成高斯分布
def Gaussian_Distribution(N=2, M=3000, m=100, sigma=20):
    '''
    Parameters
    ----------
    N 维度
    M 样本数
    m 样本均值
    sigma: 样本方差
    
    Returns
    -------
    data  shape(M, N), M 个 N 维服从高斯分布的样本
    Gaussian  高斯分布概率密度函数
    '''
    mean = np.zeros(N) + m  # 均值矩阵，每个维度的均值都为 m
    cov = np.eye(N) * sigma  # 协方差矩阵，每个维度的方差都为 sigma
    # 产生 N 维高斯分布数据
    data = np.random.multivariate_normal(mean, cov, M)
    # N 维数据高斯分布概率密度函数
    Gaussian = multivariate_normal(mean=mean, cov=cov) 
    return data, Gaussian




# 生成均匀分布
def uniform_Distribution(a=0.6, b=1.2, M=3000):
    re = []
    for i in range(M):
        re.append(round(random.uniform(a, b),3))
    return re


def data_pre(lt=3000, lw=5000, m=100, sd=20):
    worker, _ = Gaussian_Distribution(M=lw)
    task, _ = Gaussian_Distribution(M=lt)
    w_e = uniform_Distribution(M=lw)
    t_e = uniform_Distribution(M=lt)
    w_x,w_y = worker.T 
    t_x,t_y = task.T
    re_worker = []
    re_task = []
    for i in range(lw):
        tmp = {
            'x' : round(w_x[i],3),
            'y' : round(w_y[i],3),
            'epsilon' : round(w_e[i],3)
        }
        re_worker.append(tmp)
    for i in range(lt):
        tmp = {
            'x' : round(t_x[i],3),
            'y' : round(t_y[i],3),
            'epsilon' : round(t_e[i],3)
        }
        re_task.append(tmp)
    # 打开一个文件
    fo = open(str(lt)+"_"+str(lw)+"_"+str(m)+"_"+str(sd)+".txt", "w")
    fo.write(str(re_task))
    fo.write('\n')
    fo.write(str(re_worker))
    # 关闭打开的文件
    fo.close()


def pre_defined(N=50):
    a = np.random.uniform(-100, 100, size=(N,2))
    re = []
    for i in a:
        tmp = {
            'x' : round(i[0],3),
            'y' : round(i[1],3)
        }
        re.append(tmp)
    plt.scatter(a[:,0],a[:,1])
    plt.show()
    # 打开一个文件
    fo = open("predefined"+str(N)+".txt", "w")
    fo.write(str(re))
    # 关闭打开的文件
    fo.close()
    return re

# 固定其中三项
T_size = [1000,2000,3000,4000,5000]
W_size = [3000,4000,5000,6000,7000]
mean = [50,75,100,125,150]
sigma = [10,15,20,25,30]


# 生成测试的点
# for i in range(len(T_size)):
#     data_pre(lt=T_size[i])
# for i in range(len(W_size)):
#     data_pre(lw=W_size[i])
# for i in range(len(mean)):
#     data_pre(m=mean[i])
# for i in range(len(sigma)):
#     data_pre(sd=sigma[i])


# 生成预定义的点
pre_node = pre_defined()
print(pre_node)

# '''二元高斯散点图举例'''
# data, _ = Gaussian_Distribution(N=2, M=3000, m=150, sigma=30)
# for i in data:
#     print(i)
# x, y = data.T
# print(len(data))
# plt.scatter(x, y)
# plt.title('服从均值=100，方差=20的二维正态分布散点图')
# plt.xlabel("x")
# plt.ylabel("y")
# plt.grid(linewidth='1')
# plt.show()
