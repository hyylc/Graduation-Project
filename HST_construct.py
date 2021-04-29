import random
import math

def dis(i, j):
    """
    计算Euclid距离\n

    param i : 二维坐标下的点\n
    param j : 二维坐标下的点\n
    """
    return math.sqrt(math.pow((i['x'] - j['x']), 2) + math.pow((i['y'] - j['y']), 2))

def max_dis(V):
    """
    求最远距离\n
    param v : 点集\n
    """
    d = 0.0
    for i in V:
        for j in V:
            d = max(d, dis(i, j))
    return d

class HST_C(object):
    def __init__(self, metrixs): # 创建对象同时要执行的代码
        self.metrixs = metrixs
        self.maxD = max_dis(metrixs)
        self.D = math.ceil(math.log(2*self.maxD,2))
        self.num_of_nodes = 0
        # beta
        self.beta = random.uniform(0.5,1)
        # 每一层的结点集合
        self.S = [[] for i in range(self.D+1)]
        # 划分距离
        self.r = [0]*(self.D+1)
        # maximum number of branches in the tree
        self.c = 0

    def __del__(self): # 释放对象同时要执行的代码
        pass

    # 构建树的一个节点
    def HST(self, r):
        """
        param r : 新的节点\n
        """
        return [r]

    # 构建树
    def construct(self, tree, i):
        """
        构建HST树\n
        :param i: 新结点的层数\n
        """
        # 一棵只有根结点的树
        if i < 0:
            return
        T = tree[0]
        print('当前根节点；',T,' 集合大小：',len(T),' 当前层数：',i)
        self.r[i] = self.beta*pow(2,i)
        while T != []:
            temp = []
            new_T = []
            for vertex_j in T:
                if dis(T[0],vertex_j) <= self.r[i]:
                    temp.append(vertex_j)
            # T - U
            for vertex in T:
                if vertex not in temp:
                    new_T.append(vertex)
            if len(temp) != 0:
                HST_U = self.HST(temp)
                tree.append(HST_U)
                self.construct(HST_U, i-1)
                T = new_T

        self.c = max(self.c, len(tree))
            


    def print_tree(self, HST_tree, level):
        """
        先序遍历树\n
        构建S[D]\n
        :param HST_tree: 树\n
        :param level: 当前根结点的层数\n
        """
        if level < 0:
            return
        if len(HST_tree) != 0:
            print(HST_tree[0])
        elif level == 0:
            print(HST_tree)
        for i in range(len(HST_tree)-1):
            self.print_tree(HST_tree[i+1], level-1)

    def get_S(self, HST_tree, level):
        """
        构建S[D]\n
        :param HST_tree: 树\n
        :param level: 当前根结点的层数\n
        """
        if level < 0:
            return
        if len(HST_tree) != 0:
            self.S[level].append(HST_tree[0])
        elif level == 0:
            self.S[level].append(HST_tree)
        for i in range(len(HST_tree)-1):
            self.get_S(HST_tree[i+1], level-1)

    def LCA_lvl(self, i, j):
        re = 0
        if i == j:
            return re
        while i!=j:
            re = re + 1
            i = int(i/(self.c-1))
            j = int(j/(self.c-1))
            # print('i = ',i,', j = ',j)
        re = re + 1
        return re

    # 算法1
    def algorithm_1(self):
        """
        :param V: 原空间上的点的集合\n
        """
        # 构造HST树
        HST_tree = self.HST(self.metrixs)
        print('HST = ',HST_tree)
        self.construct(HST_tree, self.D)
        self.get_S(HST_tree, self.D)
        return HST_tree


metrixs = [{'x': 84.731, 'y': 96.679}, {'x': 73.091, 'y': 61.025}, {'x': 21.623, 'y': -8.691}, {'x': 50.285, 'y': 7.365}, {'x': 41.578, 'y': 17.63}, {'x': 0.318, 'y': -37.534}, {'x': -98.838, 'y': 53.716}, {'x': 58.639, 'y': -33.503}, {'x': -25.529, 'y': -49.588}, {'x': 23.783, 'y': -19.844}, {'x': -66.328, 'y': -67.78}, {'x': 62.693, 'y': 52.31}, {'x': -61.003, 'y': -9.276}, {'x': 34.647, 'y': -15.571}, {'x': -51.554, 'y': -47.571}, {'x': 86.764, 'y': -5.309}, {'x': 51.396, 'y': 37.299}, {'x': -6.494, 'y': -10.359}, {'x': 49.383, 'y': 24.996}, {'x': 25.421, 'y': -8.924}, {'x': -50.877, 'y': -54.115}, {'x': 32.117, 'y': -39.563}, {'x': 80.477, 'y': 0.959}, {'x': 98.764, 'y': -22.888}, {'x': 29.705, 'y': 34.432}, {'x': -51.847, 'y': -56.939}, {'x': 56.716, 'y': 43.291}, {'x': 29.84, 'y': -45.85}, {'x': 94.856, 'y': -8.871}, {'x': 79.392, 'y': -67.891}, {'x': 64.157, 'y': 43.384}, {'x': 2.768, 'y': 8.806}, {'x': -22.433, 'y': 58.618}, {'x': 53.405, 'y': 46.267}, {'x': 55.316, 'y': -80.415}, {'x': 81.293, 'y': 0.44}, {'x': -14.981, 'y': -90.645}, {'x': -39.731, 'y': -13.763}, {'x': 22.49, 'y': 50.28}, {'x': -56.507, 'y': 38.265}, {'x': -52.655, 'y': 39.0}, {'x': -67.125, 'y': -48.786}, {'x': -99.861, 'y': 6.899}, {'x': 37.119, 'y': -34.57}, {'x': 26.137, 'y': 16.669}, {'x': 98.419, 'y': 93.192}, {'x': 27.295, 'y': -91.1}, {'x': 79.559, 'y': -54.369}, {'x': 93.997, 'y': -86.435}, {'x': -26.949, 'y': 59.84}]
# V的一个随机序列PI
random.shuffle(metrixs)
# 随机序列作为建树的根节点
Test = HST_C(metrixs)
HST_tree = Test.algorithm_1()
print('最远距离：',Test.maxD)
print('最高层数：',Test.D)
print('beta = ',Test.beta)
print('根节点：',Test.S[Test.D])
print('叶子节点：',Test.S[0])
print('叶子节点数：',len(Test.S[0]))
