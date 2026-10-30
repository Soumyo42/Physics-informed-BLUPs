# Coding the toy example.
# f + f'' = 0
# solution: c sin x
import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.axes_grid.inset_locator import inset_axes
import random
from src.covderivatives import sympydiff1D, torchdiff1D, sqexpnumpy, sqexp1D, sqexptorch
from krigingfunctions import simplekrig, construct_block, construct_triangle, construct_blocktriangle_matrix, construct_block_matrix, subidea_one, subidea_two, subidea_one_EXP

# user input
L: float = 0
U: float = 2 * np.pi
disc: int = 100
c: float = 1

x = np.linspace(L, U, disc) # total x
y = c * np.sin(x)           # c sin x

# x_in = np.random.uniform(L, U)
# x_out = np.random.uniform(L, U)

# x_in = np.pi/4 + 0.8                                         # CK observation
# x_in = np.pi/4                                             # LK observation 
x_in = np.pi/4 + 0.4
x_out = np.pi/4 + 1.5

y_in = c * np.sin(x_in)
diff_out = 0

sigma: float = 1
# theta: float = 2.0                                          # CK figure
theta: float = 1.0                                         # LK figure

# sympykwargsCK = {'kernelexp': sqexp1D,                       # fixed for co-Kriging
#                'sigmaval': sigma,
#                'thetaval': thetaCK}
# torchkwargsCK = {'kernelexp': sqexptorch,
#                'sigmaval': sigma,
#                'thetaval': thetaCK}

sympykwargs = {'kernelexp': sqexp1D,                        # fixed for Lagrangian Kriging
               'sigmaval': sigma,
               'thetaval': theta}
torchkwargs = {'kernelexp': sqexptorch,
               'sigmaval': sigma,
               'thetaval': theta}



def sk(xout):
    h = sympydiff1D(x_in, xout, xindex=0, yindex=0, **sympykwargs)
    k = sympydiff1D(x_in, x_in, xindex=0, yindex=0, **sympykwargs)
    return (h/k) * y_in

def skder(xout):
  K = np.zeros((2,2))
  H = np.zeros((2,1))
  Vars1 = sympydiff1D(x_in,
                      x_in,
                      xindex=0,
                      yindex=0,
                      **sympykwargs)
  
  Varsc = sympydiff1D(xout,
                      xout,
                      xindex=2,
                      yindex=2,
                      **sympykwargs)
  
  Varsstar = sympydiff1D(xout,
                        xout,
                        xindex=0,
                        yindex=0,
                        **sympykwargs)
  
  Covs1sstar = sympydiff1D(x_in,
                          xout,
                          xindex=0,
                          yindex=0,
                          **sympykwargs)
  
  Covs1sc = sympydiff1D(x_in,
                        xout,
                        xindex=0,
                        yindex=2,
                        **sympykwargs)
  
  Covscstar = sympydiff1D(xout,
                          xout,
                          xindex=2,
                          yindex=0,
                          **sympykwargs)
  K[0,0] = Vars1
  K[0,1] = Covs1sstar + Covs1sc
  K[1,1] = Varsstar + Varsc + (2 * Covs1sstar)
  K[1,0] = Covs1sstar + Covs1sc
  Kp = K
  
  H = np.zeros((2,1))
  H[0,0] = Covs1sstar
  H[1,0] = Varsstar + Covscstar
  Hp = H

  Kinv = np.linalg.inv(Kp)
  ypred = Hp.T @ Kinv @ [y_in, diff_out]
  return ypred

def skconst(xout):
   Covs1sstar = sympydiff1D(x_in,
                           xout,
                           xindex=0,
                           yindex=0,
                           **sympykwargs)
   Covs1sc = sympydiff1D(x_in,
                         xout,
                         xindex=0,
                         yindex=2,
                         **sympykwargs)
   Vars1 = sympydiff1D(x_in,
                       x_in,
                       xindex=0,
                       yindex=0,
                       **sympykwargs)
   return ((Covs1sstar - Covs1sc)/(2 * Vars1)) * y_in

# print(Kp)
# print(Kinv)
# print(Kinv @ Kp)
# print(Hp)

# print(f'{h} times {1/k} times {y_in}')
# print(ysk)
# print(ypred) 

#plotting
fig, ax = plt.subplots(figsize = (10,10))

ypred = skder(x_out)
ysk = sk(x_out)
# yconst = skconst(x_out)

fpred = np.vectorize(skder)(x)
fsk = np.vectorize(sk)(x)
# fconst = np.vectorize(skconst)(x)

ax.plot(x, y, linestyle = 'dashed', c='black')
ax.scatter(x_out, ypred, c='blue', marker = '*', s = 200)
# ax.scatter(x_out, yconst, c='brown', marker = '*')
ax.scatter(x_out, diff_out, c='black', marker = 'x', s = 200)
ax.scatter(x_out, ysk, c = 'orange', marker = '+', linewidths=4, s = 200)
ax.plot(100*[x_out], np.linspace(0,1.5,100), c='gray', linestyle = 'dotted')
ax.plot(x,len(x)*[0], c='gray', alpha = 0.5)
ax.plot(x, fpred, label = 'Co-Krig')
ax.plot(x, fsk, label = 'Simple Krig')
# ax.plot(x, fconst)

mid = ypred - 0.01

# inset Axes....
# x1, x2, y1, y2 = x_out - 0.05, x_out + 0.05, mid - 0.05, mid + 0.05  # subregion of the original image
# axins = ax.inset_axes(
#     [0.7, 0.8, 0.1, 0.1],
#     xlim=(x1, x2), ylim=(y1, y2), xticklabels = [])
# axins.scatter(x_out, ypred, c='green', marker = '*', s = 100)
# axins.scatter(x_out, ysk, c ='black', marker = 'x', s = 100)
# axins.plot(x, fpred) #, c = 'blue')
# axins.plot(x, fsk)# c = 'orange')
# ax.indicate_inset_zoom(axins, edgecolor="black")

ax.scatter(x_in, y_in, c = 'red', s = 100)
ax.text(x_in - 0.1, y_in - 0.1, '$Z(\mathbf{s}^1)$', fontsize=20)
ax.text(x_out-0.4, -0.07, '$Z(\mathbf{s}^c) + Z(\mathbf{s}^{*}) = 0$', fontsize = 20)
# ax.text(x_out+0.1, ypred+0.1, 'Derivative obs.')
# ax.text(x_out+0.1, ysk-0.1, 'Simple Kriging')
plt.legend(fontsize = 23, loc = 3)
plt.xlim(0, np.pi)
# plt.text(x_in, y_in-0.05, '$Z(\mathbf{s}^1)$', )

plt.ylim(-0.2, 1.1)
plt.savefig('ToyExampleCK.pdf', dpi = 200, bbox_inches = 'tight', pad_inches = 0)
plt.show()


fig, ax = plt.subplots(figsize = (10,10))
yck = skconst(x_out)
fck = np.vectorize(skconst)(x)

ax.plot(x, y, linestyle = 'dashed', c='black')
ax.scatter(x_out, yck, c='blue', marker = '*', s = 200)
# ax.scatter(x_out, yconst, c='brown', marker = '*')
ax.scatter(x_out, diff_out, c='black', marker = 'x', s = 200)
ax.scatter(x_out, ysk, c = 'orange', marker = '+', linewidths=4, s = 200)
ax.plot(x,len(x)*[0], c='gray', alpha = 0.5)
ax.plot(100*[x_out], np.linspace(0,1.5,100), c='gray', linestyle = 'dotted')
ax.plot(x, fck, label = 'Lag. Krig')
ax.plot(x, fsk, label = 'Simple Krig')
ax.text(x_in - 0.02, y_in - 0.1, '$Z(\mathbf{s}^1)$', fontsize=20)
ax.text(x_out-0.7, -0.07, '$Z(\mathbf{s}^c) + Z(\mathbf{s}^{*}) = 0$', fontsize = 20)
ax.scatter(x_in, y_in, c = 'red', s = 100)
plt.xlim(0, np.pi)
plt.ylim(-0.2, 1.1)
plt.legend(fontsize = 23, loc = 1)
plt.savefig('ToyExampleLK.pdf', dpi = 200, bbox_inches = 'tight', pad_inches = 0)
plt.show()

