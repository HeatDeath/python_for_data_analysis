import pandas as pd
from pandas import Series, DataFrame
import numpy as np

np.set_printoptions(precision=4)

# Series 是一种很类似于一维数组的对象，它由一组数据以及一组与直线馆的数据标签（索引）组成
# 索引在左，值在右边
obj = Series([4, 7, -5, 3])
print(obj)
print(obj.index)
print(obj.values)

obj2 = Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
print(obj2)
print(obj2.index)
print(obj2.values)
print(obj2['a'])
print(obj2[['c', 'a', 'd']])
# numpy 数组运算会保留索引与值之间的连接
print(obj2[obj2 > 0])
# Series 可以看成是一个定长的有序字典，因为他是索引值到数值得一个映射
print('b' in obj2)
print(7 in obj2)

sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
obj3 = Series(sdata)
print(obj3)

states = ['California', 'Ohio', 'Oregon', 'Texas']
obj4 = Series(sdata, index=states)
print(obj4)

pd.isnull(obj4)
pd.notnull(obj4)
obj4.isnull()

# 在算术运算中，自动对齐不同索引的数据
print(obj3)
print(obj4)
print(obj3 + obj4)

# Series 对象本身及其索引都有一个 name 属性，该属性跟 pandas其他的关键功能关系非常密切
obj4.name = 'population'
obj4.index.name = 'state'
print(obj4)
