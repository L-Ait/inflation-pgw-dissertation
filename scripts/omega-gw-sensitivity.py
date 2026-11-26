import os
import numpy as np
import matplotlib.pyplot as plt

#Define constants
Eul_Masch = 0.57721 # Euler-Mascheroni Constant

'''Both constants below defined in S. Chongchitnan and G. Efstathiou. “Prospects for direct detection
of primordial gravitational waves”. In: Physical Review D—Particles, Fields,
Gravitation, and Cosmology 73.8 (2006), p. 083511.'''

C_Const = 4*(np.log(2) + Eul_Masch) - 5 
preFactor = 4.36e-15 

# Define I(k) function (41 in paper)
def IFunc(epsPivScale, epsVals, expInput):
    C = C_Const
    fact1Num = 1 - (C+1)*epsVals/4
    fact1Denom = 1 - (C+1)*epsPivScale/4
    fact1 = (fact1Num/fact1Denom)**2
    fact2 = np.exp(-2*expInput)
    return fact1*fact2    

#Below: y is N, t is ln(k) - Uses the Runge-Kutta method
def ODE_RK(f,y0,h,a,b):
    M = int(round((b-a)/h))
    y = np.zeros(M+1)
    t = np.linspace(a,b,M+1)
    y[0] = y0
    for i in np.arange(0,M):
        s1 = f(t[i], y[i])
        s2 = f(t[i]+h/2, y[i]+h*s1/2)
        s3 = f(t[i]+h/2, y[i]+h*s2/2)
        s4 = f(t[i]+h, y[i]+h*s3)
        y[i+1] = y[i]+h/6*(s1+2*(s2+s3)+s4)
    return y,t

# Step for all subsequent Forward-Euler passes
h = 0.001

# Generic ICs
y0 = 60
a, b = np.log(0.002), np.log(20*np.pi)


''' R^{2} Inflation Case  '''
# Define ODE - R2
def R2InfN(lnk,N):
    R2Eps = 3/(4*N**2)
    return 1/(R2Eps - 1)

# Solve ODE
R2N, R2lnk = ODE_RK(R2InfN, y0, h, a, b)

# Find R2 epsilon value
R2Epsilon = 3/(4*(R2N**2))

# Compute I(k) for the R2 model
R2Expo = 1/40
R2Ivals = IFunc(R2Epsilon[0],R2Epsilon,R2Expo)

# Compute grav. wave spectrum values for R2 model
# Note: Middle factor comes from calculating r at N = 60
R2Omega = preFactor*0.1*R2Ivals

# Put in terms of frequency
R2k = np.exp(R2lnk)
R2f = (2*np.pi/R2k)

R2lnf = np.log(2*np.pi)-R2lnk

''' Polynomial n = 2 Case '''
def polyInf(lnk, N):
    polyEps = 1/(2*N + 1)
    return 1/(polyEps - 1)

polyN, polylnk = ODE_RK(polyInf, y0, h, a, b)

polyEpsilon = 1/(2*polyN + 1)

polyExpo = np.log(11) - np.log(41)/2
polyIvals = IFunc(polyEpsilon[0],polyEpsilon,polyExpo)

polyOmega = preFactor*0.1*polyIvals

polyk = np.exp(polylnk)
polyf = (2*np.pi/polyk)

polylnf = np.log(2*np.pi)-polylnk

''' Alpha Attractors alpha = 1 Case '''
# Define ODE - Alpha
def alphaInfN_1(lnk, N):
    alphaEps_1 = 3/(4*N**2 + 2*N*np.sqrt(21) + 3)
    return 1/(alphaEps_1 - 1)

# Solve ODE
alphaN_1, alphalnk_1 = ODE_RK(alphaInfN_1, y0, h, a, b)
# Find alpha epsilon values
alphaEpsilon_1 = 3/(4*alphaN_1**2 + 2*alphaN_1*np.sqrt(21) + 3)

# Compute I(k) for the alpha attractors model
alphaExpo_1 = 0.023217
alphaIvals_1 = IFunc(alphaEpsilon_1[0],alphaEpsilon_1,alphaExpo_1)

# Compute grav. wave spectrum values for alpha attractors model
alpha_r_1 = 12/(60**2 + 30*np.sqrt(21) + 3/4)
alphaOmega_1 = preFactor*0.1*alphaIvals_1

# Put in terms of frequency
alphak_1 = np.exp(alphalnk_1)
alphaf_1 = (2*np.pi/alphak_1)

alpha_1lnf = np.log(2*np.pi)-alphalnk_1

''' Alpha Attractors alpha = 3 Case '''
# Define ODE - Alpha
def alphaInfN_3(lnk, N):
    alphaEps_3 = 9/(4*N**2 + 2*N*np.sqrt(117) + 9)
    return 1/(alphaEps_3 - 1)

# Solve ODE
alphaN_3, alphalnk_3 = ODE_RK(alphaInfN_3, y0, h, a, b)
# Find alpha epsilon values
alphaEpsilon_3 = 3/(4*alphaN_3**2 + 2*alphaN_3*np.sqrt(117) + 3)

# Compute I(k) for the alpha attractors model
alphaExpo_3 = 0.0635289
alphaIvals_3 = IFunc(alphaEpsilon_3[0],alphaEpsilon_3,alphaExpo_3)

# Compute gravitational wave spectrum values for alpha attractors model
alpha_r_3 = 12/(60**2 + 30*np.sqrt(117) + 3/4)
alphaOmega_3 = preFactor*0.1*alphaIvals_3

# Put in terms of frequency
alphak_3 = np.exp(alphalnk_3)
alphaf_3 = (2*np.pi/alphak_3)

alpha_3lnf = np.log(2*np.pi)-alphalnk_3

''' Plotting sensitivity curves '''

# Ensures the sensitivity data is found correctly
data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

# Extract sensitivity values
xLISA, yLISA = np.loadtxt(os.path.join(data_dir, "lisa.csv"),
                          delimiter=",", unpack=True, skiprows=1)
xDECIGO, yDECIGO = np.loadtxt(os.path.join(data_dir, "decigo.csv"),
                          delimiter=",", unpack=True, skiprows=1)
xSKA, ySKA = np.loadtxt(os.path.join(data_dir, "ska.csv"),
                          delimiter=",", unpack=True, skiprows=1)
xIPTA, yIPTA = np.loadtxt(os.path.join(data_dir, "ipta.csv"),
                          delimiter=",", unpack=True, skiprows=1)

fig, ax = plt.subplots(2,1, figsize = (6,6))

ax[0].loglog(R2lnf, R2Omega, linestyle = '-', color = 'lime', lw = 1,label = r'$R^{2}$ Inflation')
ax[0].loglog(alpha_1lnf, alphaOmega_1, linestyle = '-', color = 'yellow', lw = 1, label = r'$\alpha$ Inflation ($\alpha = 1$)')
ax[0].loglog(alpha_3lnf, alphaOmega_3, linestyle = '-', color = 'red', lw = 1, label = r'$\alpha$ Inflation ($\alpha = 3$)')
ax[0].loglog(polylnf, polyOmega, linestyle = '-', color = 'gold', lw = 1, label = r'$\phi^{2}$ Inflation')
ax[0].loglog(xSKA, ySKA, linestyle = '--', color = 'brown', label = r'SKA')
ax[0].loglog(xIPTA, yIPTA, linestyle = '--', color = 'darkgreen', label = r'IPTA')
ax[0].loglog(xLISA, yLISA, linestyle = '--', color = 'black', label = r'LISA')
ax[0].loglog(xDECIGO,yDECIGO, linestyle = '--', color = 'darkblue', label = r'DECIGO')


ax[1].loglog(R2lnf, R2Omega, linestyle = '-', color = 'lime', lw = 1,label = r'$R^{2}$ Inflation')
ax[1].loglog(alpha_1lnf, alphaOmega_1, linestyle = '-', color = 'yellow', lw = 1, label = r'$\alpha$ Inflation ($\alpha = 1$)')
ax[1].loglog(alpha_3lnf, alphaOmega_3, linestyle = '-', color = 'red', lw = 1, label = r'$\alpha$ Inflation ($\alpha = 3$)')
ax[1].loglog(polylnf, polyOmega, linestyle = '-', color = 'gold', lw = 1, label = r'$\phi^{2}$ Inflation')

plt.xlabel(r'$f/Hz$')
ax[0].set_ylabel(r'$\omega_{gw}$')
ax[1].set_ylabel(r'$\omega_{gw}$')
ax[1].set_yscale('linear')
ax[0].set_xlim(10e-10,5)
ax[0].set_ylim(10e-18,10e-2)
ax[1].set_xlim(10e-10,1)
ax[0].legend(loc = 'center left', bbox_to_anchor=(1.05, -0.2))

# Save figure to the correct place
fig.savefig(os.path.join(os.path.dirname(__file__), "..", "figures", "omega_gw_vs_detectors.png"),
            dpi=300, bbox_inches="tight")

plt.show()