#09033486413を循環させ、3進数に書き換えてじゃんけんの手に変換する
import random
def Base_10_to_n(X, n):
    if (int(X/n)):
        return Base_10_to_n(int(X/n), n)+str(X%n)
    return str(X%n)

a=[0,9,0,3,3,4,8,6,4,1,3]
b=[]
for i in range(10):
    b.append(int(Base_10_to_n(a[i],3)))
    print(b[i])
