import matplotlib.pyplot as plt
import numpy as np
z=[]
calf=[[90,60,90],[90,90,30],[60,60,60],[60,60,90],[30,30,30]]
print("Las calificaciones son:")
print(calf)
print("Los valores promedios para cada materia son:")
for i in range(3):
  x=0
  for j in range(5):
    x=calf[j][i]+x
  z.append(x/(j+1))
  print(z[i])

cov=[[0,0,0],[0,0,0],[0,0,0]]
for j in range(i+1):
  for k in range(i+1):
    sum=0
    for r in range(5):
      sum=(calf[r][j]-z[j])*(calf[r][k]-z[k])+sum
    cov[j][k]=sum/(r+1)
print("La matriz covariante es:")
print(cov)

y,v=np.linalg.eig(cov)


r=np.argsort(y)[::-1]
e1=v[:,r[0]]
e2=v[:,r[1]]
x=[e1,e2]
print("Tomando los dos eigenvectores para los dos eigenvalores más grandes son:")
print(np.transpose(x))
print("Multiplicando esto por la matriz de calificaciones:")
b=calf@np.transpose(x)
print(b)
print("Multiplicando el primer eigenvector por la matriz de calificaciones:")
print(calf@e1)
x1, y1=b[:,0],b[:,1]
plt.plot(x1, y1,"x")
plt.axis("equal")
plt.show()


