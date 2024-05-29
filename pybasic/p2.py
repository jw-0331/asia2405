#%%
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
# %%
from sklearn.datasets import load_iris
iris = load_iris()
print(type(iris), iris)
# %%
iris.keys()
# %%
print(iris['data'].shape)
# %%
df=pd.DataFrame(iris['data'])
df
# %%
print(iris['feature_names'])
# %%
cols=['sl', 'sw', 'pl', 'pw']
df=pd.DataFrame(iris['data'], columns=cols)
df
# %%
print(iris['target'].shape)
# %%
print(iris['target_names'])
# %%
iris.keys()
print(iris['DESCR'])
# %%
# 대표값을 통한 EDA
df.describe()
# %%
# csv로 출력
df.to_csv('data/iris.csv')
# %%
plt.plot(df)
# %%
df.plot()
df.plot(kind='line')
df.plot(kind='kde')
df.plot(kind='scatter', x='sl', y='sw')
df.plot(kind='box', x='sl', y='sw')
df.plot(kind='box')
df.plot(kind='box', vert=False)
# %%
import seaborn as sns

sns.boxplot(df)
# %%
df
# %%
df['label']=iris['target']
df
# %%
sns.pairplot(df)
# %%
sns.pairplot(df,hue='label')
# %%
df.values
# %%
df.shape
# %%
df.describe()
# %%
df.dtypes
# %%
df.head()
# %%
df.tail()
# %%
# 중복 데이터값 있는지 확인
df[df.duplicated(keep=False)]
# %%
# 중복값 제외(drop)하고 df 업데이트
df=df.drop_duplicates()
df.shape
# %%
df.isna().sum() # 결측값 있는지 확인
df=df.dropna()  # 결측값 제거
df.shape
# %%
# 시각화
df['sl'].plot(kind='hist', bins=30)
# %%
sns.pairplot(df,hue='label')
# 시각적으로 어느정도 분리될것으로 보여지나
# 겹치는 부분이 많아서 통계적 분리는 어려울것으로 보여진다.
# 우상향 하는 그래프로 볼때 상관성 분석이 필요할 것으로 보여진다.
# %%
df.corr()
# %%
df.iloc[:,0:4].corr()
# %%
# pw와 pl의 상관성 시각화
sns.heatmap(df.iloc[:, 0:4].corr(), annot=True)
# %%``
