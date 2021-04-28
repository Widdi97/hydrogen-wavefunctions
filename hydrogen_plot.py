import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sympy import *


def Phi(phi,m_l):
    return 1/np.sqrt(2*np.pi)*np.exp(1j*m_l*phi)

def legendre(x,n):
    return sum([ (-1)**k*sp.math.factorial(2*n-2*k)/(sp.math.factorial(n-k)*sp.math.factorial(n-2*k)*sp.math.factorial(k)*2**n)*x**(n-2*k) for k in range(0,n//2+1)])

# theta with sympy
def P_l(l):
    x = Symbol("x",real=True)
    d = (x**2 - 1)**l
    for k in range(l):
        d = Derivative(d,x,evaluate=True)
    return Rational(1,2**l)/factorial(l)*d

def Theta(theta,m_l,l):
    Pl = P_l(l)
    try: # for n=1,l=0,m_s=0 there is no x in P_l to substitute
        x = list(Pl.free_symbols)[0]
        for k in range(abs(m_l)):
            Pl = Derivative(Pl,x,evaluate=True)
        Pl = Pl.subs(x,cos(theta))
    except:
        pass
    return float((sqrt((2*l+1)/2*factorial(l-abs(m_l))/factorial(l+abs(m_l)))*(1-cos(theta)**2)**(abs(m_l)/2)*Pl).evalf())

# # Plot Spherical harmonics
# theta = np.linspace(0, np.pi, 50)
# phi = np.linspace(0, 2*np.pi, 50)
# l = 3
# m_l = 2
# z = np.array([[Theta(t,m_l,l)*Phi(p,m_l) for p in phi]  for t in theta])
# X, Y = np.meshgrid(phi,theta)
# Z = z
# plt.contourf(X, Y, Z,100)
# plt.show()

def Laguerre(x,n,m): # https://qudev.phys.ethz.ch/static/content/science/BuchPhysikIV/PhysikIVap9.html#x105-294000I.5
    return float(sum([ (-1)**(j+m)*factorial(n)**2/(factorial(j)*factorial(j+m)*factorial(n-j-m))*x**j for j in range(0,n-m+1)]).evalf())

def Radial(r,n,l):
    a0 = 1 # Bohr-Radius
    L = Laguerre(r,n+l,2*l+1)
    return -sqrt(factorial(n-l-1)/(2*n*factorial(n+l)**3))*(2/(n*a0))**(3/2)*(2*r/(n*a0))**l*exp(-r/(n*a0))*L

# #plot radial part
# l = 1
# n = 3
# xs = np.linspace(0,20,100)
# plt.plot(xs,[Radial(x,n,l) for x in xs])

def Psi(r,theta,phi,n,l,m_l):
    return Radial(r,n,l)*Phi(phi,m_l)*Theta(theta,m_l,l)


# #plot z-axis cuts    
def plot_slice_spherical(phi,n,l,m_l):
    rs = np.linspace(0,40,30)
    thetas = np.linspace(0,np.pi,30)
    
    radials = [ Radial(r,n,l) for r in rs ]
    angulars_positive = [ Phi(phi,m_l)*Theta(theta,m_l,l) for theta in thetas ]
    angulars_negative = [ Phi(phi+np.pi,m_l)*Theta(theta,m_l,l) for theta in thetas ]
    
    wavefunc_positive = [[ R*A for R in radials ] for A in angulars_positive ]
    wavefunc_negative = [[ R*A for R in radials ] for A in angulars_negative ]
    wavefunc = np.concatenate((np.flip(wavefunc_negative),wavefunc_positive),axis=0)
    
    
    X = np.array([[ r*np.sin(theta) for r in rs ] for theta in thetas ])
    X_all = np.concatenate((np.flip(-X),X))
    Y = np.array([[ r*np.cos(theta) for r in rs ] for theta in thetas ])
    Y_all = np.concatenate((np.flip(Y),Y))
    
    plt.style.use('dark_background')
    fig,ax = plt.subplots(figsize=(8,8))
    plt.contourf(X_all, Y_all, np.absolute(wavefunc),256,cmap="inferno")
    plt.axis('off')
    plt.show()
    
d = plot_slice_spherical(0,3,2,0) 









