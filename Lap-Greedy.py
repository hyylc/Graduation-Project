from scipy.special import lambertw
import math
import random


def dis(i,j):
    return round(math.sqrt(math.pow((i['x'] - j['x']), 2) + math.pow((i['y'] - j['y']), 2)),3)

def polar_Laplace(x,y,epsilon):
    # epsilon = 0.1
    # 在极坐标空间扰动
    print(x,y)
    θ = round(random.uniform(0,2*math.pi),3)
    z = round(random.random(),3)
    # 指定-1分支
    r = round((-1.0/epsilon)*(lambertw((z-1)/math.e,k=-1)+1).real,3)
    # print('θ = ',θ*180.0/math.pi,'  r = ',r)
    x = round(x + r*math.cos(θ),3)
    y = round(y + r*math.sin(θ),3)
    print(x,y)
    return x,y

def Lap_Greedy():
    # worker的位置进行扰动
    W_w = []
    for i in range(len(workers)):
        # x,y = polar_Laplace(workers[i]['x'],workers[i]['y'],workers[i]['epsilon'])
        x,y = polar_Laplace(workers[i]['x'],workers[i]['y'],epsilon)
        W_w.append({
            'id' : i,
            'x' : x,
            'y' : y
        })
    print(W_w)
    # 按用户的顺序进行扰动，分配当前最近的worker
    for i in range(len(tasks)):
        # x,y = polar_Laplace(tasks[i]['x'],tasks[i]['y'],tasks[i]['epsilon'])
        x,y = polar_Laplace(tasks[i]['x'],tasks[i]['y'],epsilon)
        now = {
            'x' : x,
            'y' : y
        }
        match = dict
        min_dis = 1e5
        for j in range(len(W_w)):
            tmp = dis(now,W_w[j]) 
            if tmp < min_dis:
                match = W_w[j]
                min_dis = tmp
        M.append({
            't' : i,
            'w' : match['id']
        })
        W_w.remove(match)




workers = [
    {'x': 2,'y': 3,'epsilon': 0.01},
    {'x': 2,'y': 2,'epsilon': 0.1},
    {'x': 5,'y': 5,'epsilon': 0.2},
]

tasks = [
    {'x': 1,'y': 0,'epsilon': 0.1},
    {'x': 1,'y': 3,'epsilon': 0.1},
    {'x': 6,'y': 2,'epsilon': 0.1},
]
M = []
epsilon = 0.2
Lap_Greedy()
print(M)