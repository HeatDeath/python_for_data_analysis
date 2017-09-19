import pandas as pd
from pandas import Series, DataFrame
import numpy as np

np.set_printoptions(precision=4)

# data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
#         'year': [2000, 2001, 2002, 2001, 2002],
#         'pop': [1.5, 1.7, 3.6, 2.4, 2.9]}
# frame = DataFrame(data)
#
# print(frame)
#
# # 指定列序列
# print(DataFrame(data, columns=['year', 'state', 'pop']))
#
# # 如果传入的列在数据中找不到，就会产生 NA 值
# frame2 = DataFrame(data, columns=['year', 'state', 'pop', 'debt'],
#                    index=['one', 'two', 'three', 'four', 'five'])
# print(frame2)
#
# print(frame2.columns)
# # 获取列
# print(frame2['state'])
# print(frame2.year)
# # 获取行
# print(frame2.ix['three'])
#
# frame2['debt'] = 16.5
# frame2
#
# frame2['debt'] = np.arange(5.)
# frame2
#
# val = Series([-1.2, -1.5, -1.7], index=['two', 'four', 'five'])
# frame2['debt'] = val
# frame2
#
# # 创建新列
# frame2['eastern'] = frame2.state == 'Ohio'
# frame2
#
# # 删除列
# del frame2['eastern']
# frame2.columns

# 传入嵌套的字典，外层字典的键作为 列，内层字典的键作为 行
pop = {'Nevada': {2001: 2.4, 2002: 2.9},
       'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}

frame3 = DataFrame(pop)
print(frame3)
print(frame3.T)

print(DataFrame(pop, index=[2001, 2002, 2003]))

pdata = {'Ohio': frame3['Ohio'][:-1],
         'Nevada': frame3['Nevada'][:2]}
print(DataFrame(pdata))

frame3.index.name = 'year'
frame3.columns.name = 'state'
print(frame3)
print(frame3.values)