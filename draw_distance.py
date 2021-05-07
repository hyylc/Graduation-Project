import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib.pyplot import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D
import random
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
plt.rcParams['axes.unicode_minus']=False   #这两行需要手动设置


T_size = [100,200,300,400,500]
W_size = [300,400,500,600,700]
mean = [50,75,100,125,150]
sigma = [10,15,20,25,30]
epsilon = [0.2,0.4,0.6,0.8,1.0]


def draw_T_vary():
    # 固定其中三项，TBF的隐私参数默认0.6  PTBF的隐私参数默认0.6-1.2
    fo = open("T_vary.txt", "r")
    run_result = fo.readlines()
    fo.close()

    Exp_TBF = eval(run_result[0])
    TBF = eval(run_result[1])
    PTBF = eval(run_result[2])
    Lap_Greedy = eval(run_result[3])
    plt.plot(T_size, Exp_TBF, "rd")
    plt.plot(T_size, Exp_TBF, "r-.", linewidth=1, label='Exp-TBF')
    plt.plot(T_size, TBF, "b*")
    plt.plot(T_size, TBF, "b-.", linewidth=1, label='TBF')
    plt.plot(T_size, PTBF, "m+")
    plt.plot(T_size, PTBF, "m-.", linewidth=1, label= 'PTBF')
    plt.plot(T_size, Lap_Greedy, "yx")
    plt.plot(T_size, Lap_Greedy, "y-.", linewidth=1, label='Lap-Greedy')
    plt.legend()
    plt.grid(True)
    plt.axis()
    # plt.title("标题")
    plt.xlabel("|T|")
    plt.ylabel("total distance")
    # plt.grid(linewidth='1')
    plt.show()

def draw_W_vary():
    # 固定其中三项，TBF的隐私参数默认0.6  PTBF的隐私参数默认0.6-1.2
    fo = open("W_vary.txt", "r")
    run_result = fo.readlines()
    fo.close()

    Exp_TBF = eval(run_result[0])
    TBF = eval(run_result[1])
    PTBF = eval(run_result[2])
    Lap_Greedy = eval(run_result[3])
    plt.plot(W_size, Exp_TBF, "rd")
    plt.plot(W_size, Exp_TBF, "r-.", linewidth=1, label='Exp-TBF')
    plt.plot(W_size, TBF, "b*")
    plt.plot(W_size, TBF, "b-.", linewidth=1, label='TBF')
    plt.plot(W_size, PTBF, "m+")
    plt.plot(W_size, PTBF, "m-.", linewidth=1, label= 'PTBF')
    plt.plot(W_size, Lap_Greedy, "yx")
    plt.plot(W_size, Lap_Greedy, "y-.", linewidth=1, label='Lap-Greedy')
    plt.legend()
    plt.grid(True)
    plt.axis()
    plt.xlabel("|W|")
    plt.ylabel("total distance")
    # plt.grid(linewidth='1')
    plt.show()

def draw_m_vary():
    # 固定其中三项，TBF的隐私参数默认0.6  PTBF的隐私参数默认0.6-1.2
    fo = open("m_vary.txt", "r")
    run_result = fo.readlines()
    fo.close()

    Exp_TBF = eval(run_result[0])
    TBF = eval(run_result[1])
    PTBF = eval(run_result[2])
    Lap_Greedy = eval(run_result[3])
    plt.plot(mean, Exp_TBF, "rd")
    plt.plot(mean, Exp_TBF, "r-.", linewidth=1, label='Exp-TBF')
    plt.plot(mean, TBF, "b*")
    plt.plot(mean, TBF, "b-.", linewidth=1, label='TBF')
    plt.plot(mean, PTBF, "m+")
    plt.plot(mean, PTBF, "m-.", linewidth=1, label= 'PTBF')
    plt.plot(mean, Lap_Greedy, "yx")
    plt.plot(mean, Lap_Greedy, "y-.", linewidth=1, label='Lap-Greedy')
    plt.legend()
    plt.grid(True)
    plt.axis()
    plt.xlabel("μ")
    plt.ylabel("total distance")
    # plt.grid(linewidth='1')
    plt.show()

def draw_s_vary():
    # 固定其中三项，TBF的隐私参数默认0.6  PTBF的隐私参数默认0.6-1.2
    fo = open("s_vary.txt", "r")
    run_result = fo.readlines()
    fo.close()

    Exp_TBF = eval(run_result[0])
    TBF = eval(run_result[1])
    PTBF = eval(run_result[2])
    Lap_Greedy = eval(run_result[3])
    plt.plot(sigma, Exp_TBF, "rd")
    plt.plot(sigma, Exp_TBF, "r-.", linewidth=1, label='Exp-TBF')
    plt.plot(sigma, TBF, "b*")
    plt.plot(sigma, TBF, "b-.", linewidth=1, label='TBF')
    plt.plot(sigma, PTBF, "m+")
    plt.plot(sigma, PTBF, "m-.", linewidth=1, label= 'PTBF')
    plt.plot(sigma, Lap_Greedy, "yx")
    plt.plot(sigma, Lap_Greedy, "y-.", linewidth=1, label='Lap-Greedy')
    plt.legend()
    plt.grid(True)
    plt.axis()
    plt.xlabel("σ")
    plt.ylabel("total distance")
    # plt.grid(linewidth='1')
    plt.show()

def draw_e_vary():
    # 固定其中三项，TBF的隐私参数默认0.6  PTBF的隐私参数默认0.6-1.2
    fo = open("e_vary.txt", "r")
    run_result = fo.readlines()
    fo.close()

    Exp_TBF = eval(run_result[0])
    TBF = eval(run_result[1])
    PTBF = eval(run_result[2])
    Lap_Greedy = eval(run_result[3])
    plt.plot(epsilon, Exp_TBF, "rd")
    plt.plot(epsilon, Exp_TBF, "r-.", linewidth=1, label='Exp-TBF')
    plt.plot(epsilon, TBF, "b*")
    plt.plot(epsilon, TBF, "b-.", linewidth=1, label='TBF')
    plt.plot(epsilon, PTBF, "m+")
    plt.plot(epsilon, PTBF, "m-.", linewidth=1, label= 'PTBF')
    plt.plot(epsilon, Lap_Greedy, "yx")
    plt.plot(epsilon, Lap_Greedy, "y-.", linewidth=1, label='Lap-Greedy')
    plt.legend()
    plt.grid(True)
    plt.axis()
    plt.xlabel("ε")
    plt.ylabel("total distance")
    # plt.grid(linewidth='1')
    plt.show()


# draw_T_vary()
# draw_W_vary()
# draw_m_vary()
# draw_s_vary()
# draw_e_vary()



# c--cyan--青色
# r--red--红色
# m--magente--品红
# g--green--绿色
# b--blue--蓝色
# y--yellow--黄色
# k--black--黑色
# w--white--白色
# -   实线
# --    虚线
# -.    形式即为-.
# :    细小的虚线
# s--方形
# h--六角形
# H--六角形
# *--*形
# +--加号
# x--x形
# d--菱形
# D--菱形
# p--五角形

