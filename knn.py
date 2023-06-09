from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

plt.figure()
X, Y_true=make_blobs(n_samples=300, centers=3, cluster_std=1.3, random_state=0)

z=np.random.randint(-2,6,size=2)
#np.random.uniform([-3,-1],[4,6],size=2)
#np.array([0.5,2.5])
colores=['red','green','blue']
asignar=[]
for row in Y_true:
     asignar.append(colores[row])
plt.scatter(z[0],z[1], marker="x")     
plt.scatter(X[:,0], X[:,1], c=asignar, s=10)
plt.show()
l=[]
for j in range(300):
    l.append(np.linalg.norm(X[j]-z))
r=np.argsort(l)
kvecinos=5 #número de vecinos a tomar
x=np.zeros(kvecinos,dtype=int)
for i in range(kvecinos):
    x[i]=Y_true[r[i]]
vals, counts = np.unique(x, return_counts=True)
mode_value = vals[counts == np.max(counts)]#obtener la moda
print("El punto pertenece al grupo",colores[mode_value[0]])