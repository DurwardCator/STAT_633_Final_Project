import corner
import numpy as np
import matplotlib.pyplot as plt
from pyhmc import hmc
from statsmodels.graphics.tsaplots import plot_acf
np.random.seed(0)

burnin = 10000
N_samples = 10000
p = [100]#[2,10,100] # dimensions
rwmh_sd = [0.005]#[0.1,0.05,0.005]
hmc_eps = [0.0005]#[0.2,0.005,0.0005]

niter = burnin + N_samples

def prob(x):
    x2 = x**2
    return np.exp(-(np.prod(x2) + np.sum(x2) - 8 * np.sum(x))/2)

def logprob(x):
    x2 = x**2
    logp = -(np.prod(x2) + np.sum(x2) - 8 * np.sum(x))/2
    grad = np.array([-np.prod(x2)/xi + xi - 4 for xi in x])
    return logp, grad

x = np.linspace(-1,6,100)
Z = np.zeros((100,100))
for i in range(100):
    for j in range(100):
        Z[i,j] = prob(np.array([x[i],x[j]]))
X,Y = np.meshgrid(x,x)

plt.figure()
plt.contour(X,Y,Z)
plt.xlabel(r'$\theta_1$')
plt.ylabel(r'$\theta_2$')

for i in range(3):
    theta_0 = np.random.multivariate_normal(np.ones(p[i]), np.eye(p[i]))
    thetas_rwmh = np.zeros((niter,p[i]))
    
    theta = theta_0
    for j in range(niter):
      theta_new = np.random.multivariate_normal(theta, rwmh_sd[i] * np.eye(p[i]))
      accept = min(1,prob(theta_new)/prob(theta))
      if (np.random.rand() < accept):
        theta = theta_new
      thetas_rwmh[j] = theta
    
    thetas_hmc = hmc(logprob, x0=theta_0, n_burn=burnin, n_samples=N_samples, n_steps=10, epsilon=hmc_eps[i])
    
    plt.figure()
    plt.plot(thetas_rwmh[burnin:,0])
    plt.xlabel('Iterations')
    plt.ylabel(r'$\theta_1$')
    
    plt.figure()
    plot_acf(thetas_rwmh[burnin:,0])
    plt.ylim([-0.05,1.05])
    
    plt.figure()
    plt.hist(thetas_rwmh[burnin:,0],density=True)
    plt.xlabel(r'$\theta_1$')
    plt.ylabel('Density')
    
    plt.figure()
    plt.plot(thetas_hmc[:,0])
    plt.xlabel('Iterations')
    plt.ylabel(r'$\theta_1$')
    
    plt.figure()
    plot_acf(thetas_hmc[:,0])
    plt.ylim([-0.05,1.05])
    
    plt.figure()
    plt.hist(thetas_hmc[:,0],density=True)
    plt.xlabel(r'$\theta_1$')
    plt.ylabel('Density')