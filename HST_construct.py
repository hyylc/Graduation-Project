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
    def __init__(self, matrixs): # 创建对象同时要执行的代码
        self.matrixs = matrixs
        self.maxD = max_dis(matrixs)
        self.D = math.ceil(math.log(2*self.maxD,2))
        self.num_of_nodes = 0
        # beta
        self.beta = random.uniform(0.5,1)
        self.beta = 0.5
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
        # print('当前根节点；',T,' 集合大小：',len(T),' 当前层数：',i)
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
        # print('当前最大分支数：',self.c)

    def add_fake_nodes(self, HST_tree, level):
        """
        添加fake nodes\n
        :param HST_tree: 待补充fake nodes的树，第一个元素是根节点\n
        :param level: 新结点的层数\n
        """
        if level < 0:
            return
        l = len(HST_tree)
        # 对应数量 c-l 的fake nodes
        for i in range(self.c-l):
            HST_tree.append([])
        # 遍历HST_tree的子节点
        for i in range(len(HST_tree)-1):
            self.add_fake_nodes(HST_tree[i+1], level-1)

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

    # 算法1
    def algorithm_1(self):
        """
        :param V: 原空间上的点的集合\n
        """
        HST_tree = self.HST(self.matrixs)
        # print('HST = ',HST_tree)
        # 建树
        self.construct(HST_tree, self.D-1)
        self.add_fake_nodes(HST_tree, self.D-1)
        # self.print_tree(HST_tree, self.D)
        # 得到每层的节点（fake nodes没有建）
        self.get_S(HST_tree, self.D)
        return HST_tree


#使用所有预定义点构建HST树
fo1 = open('predefined.txt',"r")
matrixs = eval(fo1.readlines()[0])
fo1.close()
pre = []
for i in matrixs:
    pre += i
print(pre)
random.shuffle(pre)
Test = HST_C(pre)
# 构建的树
HST_tree = Test.algorithm_1()
Ture_S = []
for j in range(len(Test.S[0])):
    if Test.S[0][j] != []:
        Ture_S.append({
            'x' : Test.S[0][j][0]['x'],
            'y' : Test.S[0][j][0]['y'],
            'id' : j
        })
re = {
    'HST_D' : Test.D,
    'HST_c' : Test.c-1,
    'HST_true_S' : Ture_S,
}
# 打开一个文件
fo = open("HST.txt", "w")
fo.write(str(re))
# 关闭打开的文件
fo.close()


#分别构建25棵HST树
fo1 = open('predefined.txt',"r")
matrixs = eval(fo1.readlines()[0])
fo1.close()
# 随机序列作为建树的根节点
for i in range(25):
    pre = matrixs[i]
    print(pre)
    random.shuffle(pre)
    Test = HST_C(pre)
    # 构建的树
    HST_tree = Test.algorithm_1()
    Ture_S = []
    for j in range(len(Test.S[0])):
        if Test.S[0][j] != []:
            Ture_S.append({
                'x' : Test.S[0][j][0]['x'],
                'y' : Test.S[0][j][0]['y'],
                'id' : j
            })
    re = {
        'id' : i,
        'HST_D' : Test.D,
        'HST_c' : Test.c-1,
        'HST_true_S' : Ture_S,
    }
    # 打开一个文件
    fo = open(str(i)+"_HST.txt", "w")
    fo.write(str(re))
    # 关闭打开的文件
    fo.close()
