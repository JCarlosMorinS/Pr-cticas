from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
plt.figure()
X, Y_true=make_blobs(n_samples=300, centers=3, cluster_std=.90, random_state=3)
plt.scatter(X[:,0],X[:,1],s=10)
plt.show()
grupos=3
v=np.random.randn(grupos,2)

z=np.zeros((grupos,2))
Error=1
c=0
while Error>0.001:
    gamma=[]
    c+=1
    for j in range(300):
        l=[]
        for i in range(grupos):
            l.append(np.linalg.norm(X[j]-v[i]))
        minimo=np.amin(l)
        linea=[]
        for i in range(grupos):
            if l[i]==minimo:
                linea.append(1)
            else:
                linea.append(0)
        gamma.append(linea)    
    sum=np.sum(gamma, axis=0) 
    for i in range(grupos):
        s=[0,0]
        z[i]=v[i]
        for j in range(300):
            s=gamma[j][i]*X[j]+s
        if sum[i]==0:
           v[i]=v[i]
        else:
            v[i]=s/sum[i]
           
    Error=np.linalg.norm(z[0]-v[0])
    print(Error)
x=np.zeros((300,), dtype=int) 
for i in range(300):
    if gamma[i][0]==1:
        x[i]=2
    elif gamma[i][1]==1:
        x[i]=1
    else:
        x[i]=0
colores=['red','green','blue']
asignar=[]
for row in x:
     asignar.append(colores[row])
plt.scatter(X[:,0], X[:,1], c=asignar, s=10)
plt.show()  


