
# coding: utf-8

# In[1]:

from pandas import Series, DataFrame
import pandas as pd
import numpy as np


# In[2]:

fd = pd.read_csv('ratings.csv')



# In[3]:

# for col in fd[:2]:
#     print(col)
# DataFrame(fd[fd['userId'] == 1]['rating'].values, index=fd[fd['userId'] == 1]['movieId'].values, columns=[1])
# fd[fd['userId'] == 1]['rating'].values
# fd[fd['userId'] == 1]['movieId'].values


# In[33]:
#
# new_df = DataFrame(np.zeros((671, 9066)), index=fd['userId'].unique(), columns=fd['movieId'].unique())
# # print(DataFrame(fd[fd['userId'] == 1]['rating'].values, index=fd[fd['userId'] == 1]['movieId'].values, columns=[1]).T)
# # new_df = new_df.add(DataFrame(fd[fd['userId'] == 1]['rating'].values, index=fd[fd['userId'] == 1]['movieId'].values, columns=[1]).T)
# # new_df.ix[1][31]
# # new_df
# for user_Id in fd['userId'].unique()[:]:
#     new_df = new_df.add(DataFrame(fd[fd['userId'] == user_Id]['rating'].values, index=fd[fd['userId'] == user_Id]['movieId'].values, columns=[user_Id]).T)
#     print(new_df.ix[1][31])
#     print((new_df == 0.0).all().all())
# # new_df.ix[1][31]
#
#
# # In[28]:
#
# # new_df.to_csv('new_df.csv')
#
#
# # In[43]:
#
# # (new_df == 0.0).all().all()
#
#
# # In[ ]:

my_df = DataFrame(fd[fd['userId'] == 1]['rating'].values, index=fd[fd['userId'] == 1]['movieId'].values, columns=[1]).T + \
DataFrame(fd[fd['userId'] == 2]['rating'].values, index=fd[fd['userId'] == 2]['movieId'].values, columns=[2]).T
# DataFrame(fd[fd['userId'] == 3]['rating'].values, index=fd[fd['userId'] == 3]['movieId'].values, columns=[3]).T

print(my_df.isnull().all().all())
print(my_df.ix[1][31])
print(my_df)

