import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
import random
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
plt.rcParams['axes.unicode_minus']=False   #这两行需要手动设置

# 生成二维高斯分布
def Gaussian_Distribution(N=2, M=300, m=100, sigma=20):
    '''
    Parameters
    ----------
    N 维度
    M 样本数
    m 样本均值
    sigma: 样本标准差（5.5修改）
    
    Returns
    -------
    data  shape(M, N), M 个 N 维服从高斯分布的样本
    Gaussian  高斯分布概率密度函数
    '''
    mean = np.zeros(N) + m  # 均值矩阵，每个维度的均值都为 m
    cov = np.eye(N) * pow(sigma,2)  # 协方差矩阵，每个维度的方差都为 sigma
    # 产生 N 维高斯分布数据
    data = np.random.multivariate_normal(mean, cov, M)
    # N 维数据高斯分布概率密度函数
    Gaussian = multivariate_normal(mean=mean, cov=cov) 
    return data, Gaussian




# 生成一维均匀分布
def uniform_Distribution(a=0.6, b=1.2, M=300):
    re = []
    for i in range(M):
        re.append(random.uniform(a, b))
    return re


def data_pre(lt=300, lw=500, m=100, sd=20):
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
            'x' : w_x[i],
            'y' : w_y[i],
            'epsilon' : w_e[i]
        }
        re_worker.append(tmp)
    for i in range(lt):
        tmp = {
            'x' : t_x[i],
            'y' : t_y[i],
            'epsilon' : t_e[i]
        }
        re_task.append(tmp)
    # 打开一个文件
    fo = open(str(lt)+"_"+str(lw)+"_"+str(m)+"_"+str(sd)+".txt", "w")
    fo.write(str(re_task))
    fo.write('\n')
    fo.write(str(re_worker))
    # 关闭打开的文件
    fo.close()

# 生成二维均匀分布
def pre_defined(N=4):
    x = []
    y = []
    re = []
    for i in range(5):
        for j in range(5):
            tmp = []
            a = [(random.uniform(j*4*10,(j+1)*4*10),random.uniform(i*4*10,(i+1)*4*10)) for _ in range(N)]
            for item in a:
                x.append(item[0])
                y.append(item[1])
                tmp.append({
                    'x' : item[0],
                    'y' : item[1],
                })
            re.append(tmp)
    plt.scatter(x,y)
    plt.show()
    # 打开一个文件
    fo = open("predefined.txt", "w")
    fo.write(str(re))
    # 关闭打开的文件
    fo.close()
    return re

def data_pre1(lt=300, lw=500, m=100, sd=20):
    worker, _ = Gaussian_Distribution(M=lw)
    task, _ = Gaussian_Distribution(M=lt)
    for i in range(5):
        a=0.2+0.2*i
        b=1.2
        w_e = uniform_Distribution(a,b,M=lw)
        t_e = uniform_Distribution(a,b,M=lt)
        w_x,w_y = worker.T 
        t_x,t_y = task.T
        re_worker = []
        re_task = []
        for i in range(lw):
            tmp = {
                'x' : w_x[i],
                'y' : w_y[i],
                'epsilon' : w_e[i]
            }
            re_worker.append(tmp)
        for i in range(lt):
            tmp = {
                'x' : t_x[i],
                'y' : t_y[i],
                'epsilon' : t_e[i],
            }
            re_task.append(tmp)
        # 打开一个文件
        fo = open(str(lt)+"_"+str(lw)+"_"+str(m)+"_"+str(sd)+'_'+str(a)+'_'+str(b)+".txt", "w")
        fo.write(str(re_task))
        fo.write('\n')
        fo.write(str(re_worker))
        # 关闭打开的文件
        fo.close()


# 固定其中三项，TBF的隐私参数默认0.6  PTBF的隐私参数默认0.6-1.2
T_size = [100,200,300,400,500]
W_size = [300,400,500,600,700]
mean = [50,75,100,125,150]
sigma = [10,15,20,25,30]


# # 生成测试的点
# for i in range(len(T_size)):
#     data_pre(lt=T_size[i])
# for i in range(len(W_size)):
#     data_pre(lw=W_size[i])
# for i in range(len(mean)):
#     data_pre(m=mean[i])
# for i in range(len(sigma)):
#     data_pre(sd=sigma[i])
# # 生成5个个性隐私参数的文件
# data_pre1()
# # 生成预定义的点
# # 5.4备注：在25个块中，各自生成4个点，共100个点，分别构建了25棵块内HST和1棵整体的HST树
# pre_defined()


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
