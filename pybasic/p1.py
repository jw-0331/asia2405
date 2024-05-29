# %%
# #%% shift + enter

print('hello world')
# %%
# 리스트
a=[1,'1',[2,3,'사']]
print(a)
print(a, '타입 :', type(a), type(a[0]), type(a[1]))

a[0]=5 # 수정 가능하다
print(a)
# %%
# 튜플
a=(1,'1',[2,3,'사'])
print(a)
print(a, '타입 :', type(a), type(a[0]), type(a[1]))

a[0]=5 # 수정불가
print(a)
# %%
# 딕셔너리
b={'한국' : '서울', '일본' : '도쿄'}
print(b)
print(b, b['한국'], '타입 :', type(b), type(b['한국'])) # 접근은 키값을 넣어서 접근
# %%
# 집합
c=set([6,6, 1,1,2,5,3,4,5])
print(c,type(c))
# %%
import numpy as np

a = [1,2,3,'4']
ar = np.array(a).astype(int)
print(a, type(a), ar,type(ar))
# %%
# 자유로운 동일한 혀야변환
a = range(12)
ar = np.array(a)
print(a, type(a), ar, type(ar))
for i in range(10) :
    print(i)
# %%
# 형태변환이 자유로움
print(ar, "형태:", ar.shape)
ar34 = ar.reshape(3,4)
print(ar34, ar34.shape)
# %%
## 슬라이싱
ar34[1:]
# %%
## 연산
