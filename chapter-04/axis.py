import numpy as np

np.set_printoptions(precision=1)

arr = np.random.rand(2, 3)*10
print(arr)
arr[0][0] = 0
print(arr.shape)

# 对于二维数组来说
# axis=0 表示纵向（列）
# axis=1 表示横向（行）
print(arr.sum(0))
print(arr.sum(1))

print(arr.cumprod(0))
print(arr.cumprod(1))

print(arr.cumsum(0))
print(arr.cumsum(1))


