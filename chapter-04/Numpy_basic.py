import numpy as np
from pprint import pprint
# '''
# precision 浮点数输出精度位数（默认值8位）
# suppress 是否 禁止 使用 科学记数法（默认为False）打印小浮点值
# '''
np.set_printoptions(precision=4, suppress=True)
#
# '''
# numpy.random.randn(d0, d1, …, dn)是从标准正态分布中返回一个或多个样本值。
# numpy.random.rand(d0, d1, …, dn)的随机样本位于[0, 1)中。
# '''
# data_1 = np.random.randn(2, 3)
#
# print(data_1)
#
# # data_2 = np.random.rand(2, 3)
# #
# # print(data_2)
#
# print(data_1*10)
#
# print(data_1+data_1)
#
# # 数组 形状
# print('数组 形状:', data_1.shape)
#
# # 数组 中 数据 的 类型
# print('数组 中 数据 的 类型:', data_1.dtype)
#
# # 数组 的 维度
# print('数组 的 维度:', data_1.ndim)
#
# # --------------------------------------

# # 将 data1 转换为 1*4 的数组
# data1 = [6, 7.5, 8, 0, 1]
# arr1 = np.array(data1)
# print(arr1)
#
# # 将 data2 转换为 2*4 的数组
# data2 = [[1, 2, 3, 4], [5, 6, 7, 8]]
# arr2 = np.array(data2)
# print(arr2)
#
# # np.array 函数可以自动推断出 dtype,也可以指定 dtype
# print(arr1.dtype, arr2.dtype)
#
# # 10*1 的 0矩阵（10行 1列）
# print(np.zeros((10, 1)))
#
# # 1*10 的 0矩阵（1行 10列）
# print(np.zeros((1, 10)))
#
# # 3*6 的单位矩阵
# print(np.ones((3, 6)))
#
# # 创建一个与 arr2 形状相同的 单位矩阵
# print(np.ones_like(arr2))
#
# # 很多情况下， np.empty 返回的是 垃圾值
# print(np.empty((2, 3, 2)))
#
# # 功能类似 Python 内置的 range，但 返回的不是 list 是 ndarray
# print(np.arange(15))
# # --------------------------------------
#
# # 创建 数组 时，指定 数组中 数据 的 类型
# arr1 = np.array([1, 2, 3], dtype=np.float64)
# arr2 = np.array([1, 2, 3], dtype=np.int32)
#
# print(arr1.dtype)
# print(arr2.dtype)
#
# # 当创建 数组 时，未指定 数据 的类型，则由 numpy 自己推断
# arr = np.array([1, 2, 3, 4, 5])
# print(arr.dtype)
#
# # 通过 ndarray.astype(np.dytype) 可以修改 数组 的 数据类型
# float_arr = arr.astype(np.float64)
# print(float_arr.dtype)
#
# # 修改 数组 数据类型
# arr = np.array([3.7, -1.2, -2.6, 0.5, 12.9, 10.1])
# print(arr)
# print(arr.astype(np.int32))
#
# # 将 数据类型 从 np.string_ 转换为 np.float64
# numeric_strings = np.array(['1.25', '-9.6', '42'], dtype=np.string_)
# print(numeric_strings.astype(float))
#
# # 数据类型
# int_array = np.arange(10)
# print(int_array.dtype)
# calibers = np.array([.22, .270, .357, .380, .44, .50], dtype=np.float64)
# print(int_array.astype(calibers.dtype).dtype)
# print(int_array.dtype)
#
# # 简写形式
# empty_uint32 = np.empty(8, dtype='u4')
# print(empty_uint32.dtype)
#
# # 注意：ndarray.astype ,无论如何都会创建出一个新的数组
#
# # ------------------------------------------------
#
# '''
# 大小相等 的 数组 之间的 任何 【算术运算】 都会将 运算 应用到 【元素级别】
#
# 数组 与 标量 之间的 【算术运算】 也会将 标量值 传播到 【各个元素】
# '''
#
# arr = np.array([[1., 2., 3.], [4., 5., 6.]])
# print(arr)
# print(arr * arr)
# print(arr - arr)
#
# print(1 / arr)
# print(arr ** 0.5)
#
# # 矩阵 点乘  2*3 · 3*2 = 2*2
# print(np.dot(arr, arr.T))
# # ------------------------------------------------
#
#
# arr = np.arange(10)
# print(arr)
# print(arr[5])
# print(arr[5:8])
#
# # 将 arr 数组 索引 从 5 到 7 的元素替换为 12
# arr[5:8] = 12
#
# # 数组切片是原始数组的视图，数据【不会被复制】，视图上的任何修改都会【直接反映】 到 【原始数组】
# print(arr)
#
# arr_slice = arr[5:8]
#
# # arr 数组中，索引为 6 的元素被替换为 12345
# arr_slice[1] = 12345
# print(arr)
#
# # arr 数组中，索引为 5,6,7 的元素被替换为 64
# arr_slice[:] = 64
# print(arr)
#
# # 如果想要得到的是 ndarray 切片的一份副本而非视图的话，就需要显示的复制操作
# copy_slice = arr[5:8].copy()
# print(copy_slice)
# copy_slice[:] = 6
# print(copy_slice)
# # 此时 arr 数组 未被修改
# print(arr)
#
# # 在二维数组中，各索引位置上的元素不再是标量，而是一维数组
# arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# pprint(arr2d)
#
# # arr2d 数组 的 索引为 2 的 行
# print(arr2d[2])
#
# # arr2d 数组 的 一行 和 二行
# print(arr2d[:2])
#
# # arr2d 数组 第 0 行，第 2 列 的元素
# print(arr2d[0][2])
# print(arr2d[0, 2])
#
# arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
# pprint(arr3d)
#
# print(arr3d[0])
#
# # 4
# print(arr3d[0][1][0])
#
# old_values = arr3d[0].copy()
# arr3d[0] = 42
# pprint(arr3d)
#
# arr3d[0] = old_values
# pprint(arr3d)
#
# print(arr3d[1, 0])
#
# # ----------------------------------------------------------

# arr = np.array([[1., 2., 3.], [4., 5., 6.]])
# arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# pprint(arr2d)
#
# print(arr[1:6])
#
# print(arr2d[:2])
#
# # 0,1 行 和 1,2 列 取交集
# print(arr2d[:2, 1:])
#
# # 1 行 的 0,1 列
# print(arr2d[1, :2])
#
#
# print(arr2d[2, :1])
#
# # 第 0 列
# print(arr2d[:, :1])
#
# # 修改 数值
# arr2d[:2, 1:] = 0
# print(arr2d)
#
# # --------------------------------

# names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
# data = np.random.randn(7, 4)
# print(names)
# print(data)
#
# # 对 names 和 字符串 'Bob' 的比较运算符会产生一个布尔型数组
# print(names == 'Bob')
#
# # 这个布尔型数组可以用于索引
# print(data[names == 'Bob'])
#
# # data 数组的第一个下标对应 行数，第二个下标 对应 列数
# print(data[names == 'Bob', 2:])
# print(data[names == 'Bob', 3])
#
# # 不等于
# print(names != 'Bob')
#
# # 非
# print(data[~(names == 'Bob')])
#
# # 或
# mask = (names == 'Bob') | (names == 'Will')
# print(mask)
# print(data[mask])
#
# # 令 data 数组中，小于 0 的数据，变为 0
# data[data < 0] = 0
# print(data)
#
# #
# data[names != 'Joe'] = 7
# print(data)
#
#
# names = np.array([0, 0, 0, 0, 1, 0, 1])
# # 打印一个由 data 数组的 0,0,0,0,1,0,1 行 组成的数组
# print(data[names])
#
# # -----------------------------------------------------------


# arr = np.arange(15).reshape((3, 5))
# print(arr)
# # 转置
# print(arr.T)
#
# arr = np.random.randn(6, 3)
# # 点乘
# print(np.dot(arr.T, arr))
#
# # 高维数组需要一个 由轴编号 组成的元组才能对这些轴进行转置（费脑子）
# arr = np.arange(16).reshape((2, 2, 4))
# print(arr)
# print(arr.transpose((1, 0, 2)))
#
# # np.ndarray.swapaxes() 接受一对 轴编号
# print(arr)
# print(arr.swapaxes(1, 2))
#
# # -----------------------------------------------------------
#
# arr = np.arange(10)
#
# # 开方
# print(np.sqrt(arr))
#
# # 自然对数的底数 e = 2.718..... 的 x 次幂
# print(np.exp(arr))
#
# x = np.random.randn(8)
# y = np.random.randn(8)
# print(x)
# print(y)
#
# # 接受两个数组
# print(np.maximum(x, y)) # element-wise maximum
#
# arr = np.random.randn(7) * 5
# print(arr)
#
# # 接受 一个数组，返回 两个数组，（将输入数组的 小数部分 与 整数部分 拆分）
# print(np.modf(arr))
# # -----------------------------------------------------------



