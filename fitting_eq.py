import numpy as np
import matplotlib.pyplot as plt

#--Data generation proccess--------------
N = 25
X = np.reshape(np.linspace(0,0.9,N),(N,1))
Y=np.cos(10*X**2)+0.1*np.sin(100*X)
#-----------------------------------------

#---Function to generate the basis functions of different order--------
def phi_poly(x,i):
    return x**i

def phi_sin(x,j):
    return np.sin(2*np.pi*j*x)

def phi_cos(x,j):
    return np.cos(2*np.pi*j*x)

def phi_gauss(x,lam,mu_j):
    return np.exp(-((x-mu_j)**2)/(2*lam**2))
#-----------------------------------------------------------------------
	
	
	
phi = np.zeros([N,11])
for i in range(11):
    phi[:,i]=np.reshape(phi_poly(X,i),(25))

#-----Fitting POLYNOMIALS using maximum likelihood estimation of the weights
w_mle0 = np.dot(np.dot(np.linalg.pinv(np.dot(phi[:,:1].T,phi[:,:1])),phi[:,:1].T),Y)
w_mle1 = np.dot(np.dot(np.linalg.inv(np.dot(phi[:,:2].T,phi[:,:2])),phi[:,:2].T),Y)
w_mle2 = np.dot(np.dot(np.linalg.inv(np.dot(phi[:,:3].T,phi[:,:3])),phi[:,:3].T),Y)
w_mle3 = np.dot(np.dot(np.linalg.inv(np.dot(phi[:,:4].T,phi[:,:4])),phi[:,:4].T),Y)
w_mle11 = np.dot(np.dot(np.linalg.inv(np.dot(phi[:,:12].T,phi[:,:12])),phi[:,:12].T),Y)

########################################

N = 200
X_test = np.reshape(np.linspace(-0.3,1.3,N),(N,1))


phi_test = np.zeros([N,11])

j=0
for i in range(11):
    phi_test[:,j]=np.reshape(phi_poly(X_test,i),(N))
    j = j + 1

y_hat0_test = np.dot(phi_test[:,:1],w_mle0)
y_hat1_test = np.dot(phi_test[:,:2],w_mle1)
y_hat2_test = np.dot(phi_test[:,:3],w_mle2)
y_hat3_test = np.dot(phi_test[:,:4],w_mle3)
y_hat11_test = np.dot(phi_test[:,:12],w_mle11)


axes = plt.gca()
axes.set_xlim([-0.3,1.3])
axes.set_ylim([-1,1.6])

major_ticks_x = np.arange(-0.3, 1.3, 0.1)
major_ticks_y = np.arange(-1, 1.6, 0.1)
axes.set_xticks(major_ticks_x)
axes.set_yticks(major_ticks_y)


plt.plot(X_test,y_hat0_test,'b',label='0 order polynomial')
plt.plot(X_test,y_hat1_test,'r',label='1 order polynomial')
plt.plot(X_test,y_hat2_test,'g',label='2 order polynomial')
plt.plot(X_test,y_hat3_test,'c',label='3 order polynomial')
plt.plot(X_test,y_hat11_test,'y',label='11 order polynomial')
plt.scatter(X,Y,label='Data')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.grid(True)
plt.title('Polynomial fit regression')
plt.legend(bbox_to_anchor=(0.05, 0.1), loc=3, borderaxespad=0.)
plt.show()



##########################
##Trigonometric basis functions
N=25

phi_trigo = np.ones([N,23])

j=1
for i in range(1,12):
    phi_trigo[:,j]=np.reshape(phi_sin(X,i),(25))
    j = j + 1
    phi_trigo[:,j]=np.reshape(phi_cos(X,i),(25))
    j = j + 1


w_mle1 = np.dot(np.dot(np.linalg.inv(np.dot(phi_trigo[:,:3].T,phi_trigo[:,:3])),phi_trigo[:,:3].T),Y)
w_mle11 = np.dot(np.dot(np.linalg.inv(np.dot(phi_trigo[:,:24].T,phi_trigo[:,:24])),phi_trigo[:,:24].T),Y)


######################
#Trigonometric test


N = 200
X_test = np.reshape(np.linspace(-1,1.2,N),(N,1))

phi_trigo_test = np.ones([N,23])

j=1
for i in range(1,12):
    phi_trigo_test[:,j]=np.reshape(phi_sin(X_test,i),(N))
    j = j + 1
    phi_trigo_test[:,j]=np.reshape(phi_cos(X_test,i),(N))
    j = j + 1


y_hat1_test = np.dot(phi_trigo_test[:,:3],w_mle1)
y_hat11_test = np.dot(phi_trigo_test[:,:23],w_mle11)


axes = plt.gca()
axes.set_xlim([-1,1.2])
axes.set_ylim([-1,1.6])

major_ticks_x = np.arange(-1, 1.2, 0.1)
major_ticks_y = np.arange(-1, 1.6, 0.1)
axes.set_xticks(major_ticks_x)
axes.set_yticks(major_ticks_y)


plt.plot(X_test,y_hat1_test,'r',label='1st order trig')
plt.plot(X_test,y_hat11_test,'y',label='11th order trig')
plt.scatter(X,Y,label='Data')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.grid(True)
plt.title('Trigonometric fit regression')
plt.legend(bbox_to_anchor=(0.02, 0.1), loc=3, borderaxespad=0.)
plt.show()


####################################
#Gaussian basis functions
mu= np.linspace(0,1,20)

test=np.reshape(phi_gauss(X,0.1,mu),(25,20))
phi_ga=np.hstack([np.ones([25,1]),test])


axes = plt.gca()
axes.set_xlim([-0.3,1.3])
axes.set_ylim([-1.1,1.6])

for l_reg in [0,10,100]:

    w_mle = np.dot(np.dot(np.linalg.inv(np.dot(phi_ga.T,phi_ga)+l_reg*np.eye(21)),phi_ga.T),Y)
    x_test = np.reshape(np.linspace(-0.3,1.3,200),(200,1))
    phi_ga_test = np.hstack([np.ones([200,1]),np.reshape(phi_gauss(x_test,0.1,mu),(200,20))])
    y_hat = np.dot(phi_ga_test,w_mle)
    plt.plot(x_test,y_hat,label='lambda = %s' % l_reg)
    

plt.scatter(X,Y)
plt.legend(bbox_to_anchor=(1, 0.2), loc=1, borderaxespad=0.)
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Gaussian basis functions')
plt.grid(True)
plt.show()
