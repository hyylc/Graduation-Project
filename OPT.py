from munkres import Munkres,print_matrix
import math

def dis(i,j):
    return round(math.sqrt(math.pow((i['x'] - j['x']), 2) + math.pow((i['y'] - j['y']), 2)),3)

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

n = len(tasks)
m = len(workers)
matrix = [[0 for i in range(n)] for i in range(m)]

# 计算权重矩阵（即真实距离）
for i in range(m):
    for j in range(n):
        matrix[i][j] = dis(tasks[i],workers[j])

# 匹配序列
m = Munkres()
indexes = m.compute(matrix)
print_matrix(matrix, msg='Lowest cost through this weight matrix:')
total = 0
for row, column in indexes:
    value = matrix[row][column]
    total += value
    print(f'({row}, {column}) -> {value}')
print(f'total cost: {total}')