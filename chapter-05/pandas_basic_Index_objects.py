import pandas as pd
from pandas import Series, DataFrame
import numpy as np

np.set_printoptions(precision=4)

obj = Series(range(3), index=['a', 'b', 'c'])
index = obj.index
print(index)
print(obj)
print(index[1:])