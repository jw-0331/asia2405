
#%%
import numpy as np
import pandas as pd
import matplotlib as mlp
import matplotlib.pyplot as plt
import seaborn as sns
# %%

from sklearn.datasets import load_breast_cancer 
# %%
# 인공지능 라이브러리 안에 있는 데이터는 별거아님 ~
breast = load_breast_cancer ()
# %%
df=pd.DataFrame(breast['data'],columns=breast['feature_names'])
df
breast['target']
# %%
#지도 학습은 분류와 회귀로 나뉜다.(꽃 분류)
#비지도 학습은 ~~
df
# %%
missing_values = df.isnull().sum()
# %%
print(missing_values)

cols=['radius', 'texture', 'perimeter', 'area',
       'smoothness', 'compactness', 'concavity',
       'concave points', 'symmetry', 'fractal dimension',
       'radius error', 'texture error', 'p_error', 'a_error',
       's_error', 'c_error', 'concavity error',
       'con_po_error', 'symmetry error',
       'fractal d_error', 'worst radius', 'worst texture',
       'worst perimeter', 'worst area', 'worst smoothness',
       'worst compactness', 'worst concavity', 'wo_con_points',
       'wo_symmetry', 'wo_fr_dimension']
df=pd.DataFrame(breast['data'],columns=cols)
df
# %%
df['label']=breast['target']
df

# %%
breast['feature_names']
# %%
df
df.head()
# %%
df.tail()
# %%
df.isna().sum()
#결측치 제거
# %%
df=df.dropna()
# %%
df[df.duplicated(keep=False)]
df=df.drop_duplicates()

df.dtypes

# %%
df.plot(kind='scatter',x='radius',y='texture')
# %%
sns.pairplot(df)
#%%
df.keys()
# %%
tdf=df.loc[:,['perimeter', 'area','worst radius', 'worst texture',
       'worst perimeter', 'worst area','label']]

sns.pairplot(tdf,hue='label')
# %%
#시험준비를 해야한다.
# train_test_split 를 어떻게 써야하는지 정확히 알아야함 
# 학습데이터와 테스트데이터를 분리해야함

from sklearn.model_selection import train_test_split
df.shape

# %%
"""
train_test_split(data,                 << 데이터
                 target,               << 라벨
                 test_sizs=0.2,        << 시험용데이터 비율  
                 shuffle=True,         << 섞을까?    (데이터에따라 섞는게 좋음)
                 stratify=target,      << 라벨편향방지 
                 random_state=34)      << 매번 똑같은 분리 (뒤에 34 는 바꿔도 노상관 근데 고정값으로 해야함)

"""

data=df.iloc[:,:-1]
target=df['label']


X_train,x_test,Y_train,y_test=train_test_split(data,target,test_size=0.2,shuffle=True,stratify=target,random_state=34)      

print(X_train.shape,Y_train.shape)
print(x_test.shape,y_test.shape)

# %%
# 분류모델 선정
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# 모델 초기화
knn=KNeighborsClassifier(n_neighbors=5)

# 학습하기
knn.fit(X_train, Y_train)
# %%
score = knn.score(x_test, y_test)
print(np.round(score,2))
# %%
logi = LogisticRegression()
logi.fit(X_train, Y_train)
score = logi.score(x_test, y_test)
print(np.round(score,2))
# %%
svc = SVC()
svc.fit(X_train, Y_train)
score = svc.score(x_test, y_test)
print(np.round(score,3))
# %%
dt = DecisionTreeClassifier()
dt.fit(X_train,Y_train)
score = dt.score(x_test, y_test)
print(np.round(score,3))
# %%
