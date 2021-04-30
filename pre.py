
#确定一下测试数据的内容

import numpy as np
import matplotlib.pyplot as plt #绘图模块
import math

u = 0   # 均值μ
u01 = -2
sig = math.sqrt(0.2)  # 标准差δ
sig01 = math.sqrt(1)
sig02 = math.sqrt(5)
sig_u01 = math.sqrt(0.5)
x = np.linspace(u - 3*sig, u + 3*sig, 50)
x_01 = np.linspace(u - 6 * sig, u + 6 * sig, 50)
x_02 = np.linspace(u - 10 * sig, u + 10 * sig, 50)
x_u01 = np.linspace(u - 10 * sig, u + 1 * sig, 50)
y_sig = np.exp(-(x - u) ** 2 /(2* sig **2))/(math.sqrt(2*math.pi)*sig)
y_sig01 = np.exp(-(x_01 - u) ** 2 /(2* sig01 **2))/(math.sqrt(2*math.pi)*sig01)
y_sig02 = np.exp(-(x_02 - u) ** 2 / (2 * sig02 ** 2)) / (math.sqrt(2 * math.pi) * sig02)
y_sig_u01 = np.exp(-(x_u01 - u01) ** 2 / (2 * sig_u01 ** 2)) / (math.sqrt(2 * math.pi) * sig_u01)
plt.plot(x, y_sig, "r-", linewidth=2)
plt.plot(x_01, y_sig01, "g-", linewidth=2)
plt.plot(x_02, y_sig02, "b-", linewidth=2)
plt.plot(x_u01, y_sig_u01, "m-", linewidth=2)
# plt.plot(x, y, 'r-', x, y, 'go', linewidth=2,markersize=8)
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
def gen_clusters():
    mean1 = [0,0]
    cov1 = [[1,0],[0,10]]
    data = np.random.multivariate_normal(mean1,cov1,100)
    # mean2 = [10,10]
    # cov2 = [[10,0],[0,1]]
    # data = np.append(data,
    # np.random.multivariate_normal(mean2,cov2,100), 0)
    # mean3 = [10,0]
    # cov3 = [[3,0],[0,4]]
    # data = np.append(data,
    # np.random.multivariate_normal(mean3,cov3,100),0)
    return np.round(data,4)
def save_data(data,filename):
    with open(filename,'w') as file:
        for i in range(data.shape[0]):
            file.write(str(data[i,0])+','+str(data[i,1])+'\n')
def load_data(filename):
    data = []
    with open(filename,'r') as file:
        for line in file.readlines():
            data.append([ float(i) for i in line.split(',')])
    return np.array(data)
def show_scatter(data):
    x,y = data.T
    plt.scatter(x,y)
    plt.axis()
    plt.title("scatter")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
data = gen_clusters()
save_data(data,'3clusters.txt')
d = load_data('3clusters.txt')
show_scatter(d)

