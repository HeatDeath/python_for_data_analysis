import numpy as np
from matplotlib.pyplot import imshow, title
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)

# # 起始点，终止点，步长
# points = np.arange(-5, 5, 0.01) # 1000 equally spaced points
# print(points)
#
# # 接受两个一维数组，产生两个二维矩阵
# xs, ys = np.meshgrid(points, points)
#
# print('------------------')
# # 行上从 -5.0 到 4.99
# print(xs)
# print('------------------')
# # 列上从 -5.0 到 4.99
# print(ys)
#
# # xs和ys 两二维数组的元素，分别乘方，后两数组加和，再开方
# z = np.sqrt(xs ** 2 + ys ** 2)
# print(z)
# -----------------------------------------

# xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
# yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
# cond = np.array([True, False, True, True, False])
#
# print(xarr[~cond])
#
# # 二者等价
# # np.where 是三元表达式 x if condition else y 的矢量化版本
# # result = [(x if c else y) for x, y, c in zip(xarr, yarr, cond)]
# result = np.where(cond, xarr, yarr)
# print(result)
#
# arr = np.random.randn(4, 4)
# # 大于 0 的替换为 2， 否则替换为 -2
# arr_1 = np.where(arr > 0, 2, -2)
# print(arr_1)
# # 大于 0 的替换为 2， 否则 保留原值
# arr_2 = np.where(arr > 0, 2, arr) # set only positive values to 2
# print(arr_2)
# -----------------------------------------

# arr = np.random.randn(5, 4) # normally-distributed data
# print(arr.mean())
# print(np.mean(arr))
# print(arr.sum())
#
# # mean 和 sum 可以接受一个 axis 参数（用于计算该轴向上的统计值），最终结果是一个少一维的数组
# print(arr.mean(axis=1))
# print(arr.sum(0))
#
# arr = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
# print(arr)
# print(arr.cumsum(0))
# print(arr.cumsum(1))
# print(arr.cumprod(1))
# -----------------------------------------

# arr = np.random.randn(100)
# print((arr > 0).sum()) # Number of positive values
# print(arr[arr>0])
#
# bools = np.array([False, False, True, False])
# # any 用于检测数组中是否存在一个或多个True 存在
# # all 检查数组中是否所有值都是 True
# bools.any()
# bools.all()
# -----------------------------------------

# arr = np.random.randn(8)
# print(arr)
# arr.sort()
# print(arr)
#
# arr = np.random.randn(2,5)
# print(arr)
# # 升序排列，按行排列
# arr.sort()
# print(arr)
#
# arr.sort(1)
# print(arr)
#
# arr.sort(0)
# print(arr)
#
#
# large_arr = np.random.randn(1000)
# large_arr.sort()
# # 5%分位数
# print(large_arr[int(0.05 * len(large_arr))]) # 5% quantile
# -----------------------------------------

# # np.unique()用于找出数组中的唯一值，并返回已排序的结果
# names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
# print(np.unique(names))
# ints = np.array([3, 3, 3, 2, 2, 1, 1, 4, 4])
# print(np.unique(ints))
#
# # np.in1d() 用于测试一个数组中的值在另一个数组中的成员资格，返回一个布尔型数组
# values = np.array([6, 0, 0, 3, 2, 5, 6])
# print(np.in1d(values, [2, 3, 6]))
# print(np.in1d([2, 3, 6], values))
# -----------------------------------------

# arr = np.arange(10)
# np.save('some_array', arr)
#
# arr = np.load('some_array.npy')
# print(arr)
# -----------------------------------------

# # 加载 array_ex.txt 以 , 为分隔符
# arr = np.loadtxt('array_ex.txt', delimiter=',')
# print(arr)
# -----------------------------------------

# x = np.array([[1., 2., 3.], [4., 5., 6.]])
# y = np.array([[6., 23.], [-1, 7], [8, 9]])
#
# # 2*3 · 3*2 = 2*2
# print(x.dot(y))  # equivalently np.dot(x, y)
#
# np.dot(x, np.ones(3))

# np.random.seed(12345)
#
# from numpy.linalg import inv, qr
# X = np.random.randn(5, 5)
# print(X)
# # 转置
# print(X.T)
# mat = X.T.dot(X)
#
# print(mat)
# # 矩阵的逆
# print(inv(mat))
#
#
# mat.dot(inv(mat))
# q, r = qr(mat)
# print(r)
# -----------------------------------------

# nsteps = 1000
#
# # 生成 1000 个 或为 0 ，或者为 1 的随机数
# draws = np.random.randint(0, 2, size=nsteps)
# # print(draws)
#
# # 将 draws 中的 0 转换为 -1，将 1 转换为 0
# steps = np.where(draws > 0, 1, -1)
# print(steps)
#
# walk = steps.cumsum()
# print(walk)
#
# print(walk.min())
# print(walk.max())
# # argmax() 返回该布尔型数组第一个最大值得索引（True就是最大值）
# print((np.abs(walk) >= 10).argmax())

nwalks = 5000
nsteps = 1000
draws = np.random.randint(0, 2, size=(nwalks, nsteps)) # 0 or 1
steps = np.where(draws > 0, 1, -1)
# 行累加
walks = steps.cumsum(1)
print(walks)

walks.max()
walks.min()

# 以行为轴
hits30 = (np.abs(walks) >= 30).any(1)
print(hits30)
print(hits30.sum()) # Number that hit 30 or -30

# 每行中，最大数的索引
crossing_times = (np.abs(walks[hits30]) >= 30).argmax(1)
print(crossing_times)
print(crossing_times.mean())