import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Polygon
import getdist.plots as gplot

'''Since the Planck 2018 base_r values are not within this repo (they are too large but freely available at pla.esac.esa.int), below 
sets up the framework to have them appear on the figure. Download them from the link and place them in the path below.'''

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
FIGURES_DIR = os.path.join(REPO_ROOT, "figures")

DEFAULT_CHAIN_DIR = os.path.join(os.path.dirname(REPO_ROOT), "external",
                                 "planck_chains", "base_r")
CHAIN_DIR = os.environ.get("PLANCK_CHAIN_DIR", DEFAULT_CHAIN_DIR)


'''Setting up different power-law potentials'''
# V proportional to Phi^{2} Case
n_sN60 = 1 - 2*(2 + 2)/(4*60 + 2)
rN60 = (16*2)/(4*60 + 2)

n_sN50 = 1 - 2*(2 + 2)/(4*50 + 2)
rN50 = (16*2)/(4*50 + 2) 

# V proportional to Phi Case
n_s_LiN60 = 1 - 2*(1 + 2)/(4*60 + 1)
r_LiN60 = (16*1)/(4*60 + 1)

n_s_LiN50 = 1 - 2*(1 + 2)/(4*50 + 1)
r_LiN50 = (16*1)/(4*50 + 1) 

# V proportional to Phi^{4/3} Case
n_s_43N60 = 1 - 2*(4/3 + 2)/(4*60 + 4/3)
r_43N60 = (16*4/3)/(4*60 + 4/3)

n_s_43N50 = 1 - 2*(4/3 + 2)/(4*50 + 4/3)
r_43N50 = (16*4/3)/(4*50 + 4/3) 

# V proportional to Phi^{2/3} Case
n_s_23N60 = 1 - 2*(2/3 + 2)/(4*60 + 2/3)
r_23N60 = (16*2/3)/(4*60 + 2/3)

n_s_23N50 = 1 - 2*(2/3 + 2)/(4*50 + 2/3)
r_23N50 = (16*2/3)/(4*50 + 2/3) 

# Power Law Inflation (See Planck Paper (2013/2015) for discussion)
specIndex = np.arange(0.935,1,0.001)
rExp = -8*(specIndex - 1)

# Defining Concave vs. Convex geometry divide
rConCon = 8/3*(1-specIndex)

# R^{2} Inflation
nsR2_N60 = 1 - 2/60
rR2_N60 = 12/(60**2)
nsR2_N50 = 1 - 2/50
rR2_N50 = 12/(50**2)

# Alpha Attractors 
alphaArr = np.logspace(-1, 4, 2000)

def nsAAFUNC(N):
    return 1-4/(2*N+1)

# This uses the approximation found in the large alpha limit
def rAAFUNC(n,N):
    alpha = alphaArr
    return 24*n*alpha/(N*(2*n*N + 3*alpha))

AA_N60VAL = nsAAFUNC(60)
nsAA_N60 = np.full(
  shape = len(alphaArr),
  fill_value = AA_N60VAL,
)
rAA_N60 = rAAFUNC(1,60)

AA_N50VAL = nsAAFUNC(50)
nsAA_N50 = np.full(
  shape = len(alphaArr),
  fill_value = AA_N50VAL,
)
rAA_N50 = rAAFUNC(1,50)

#Setting up arrays of values
xvalsArr = [n_sN50, n_sN60, n_s_LiN50, n_s_LiN60, n_s_43N50, n_s_43N60, 
            n_s_23N50, n_s_23N60, nsR2_N50, nsR2_N60]
yvalsArr = [rN50, rN60, r_LiN50, r_LiN60, r_43N50, r_43N60,
            r_23N50, r_23N60, rR2_N50, rR2_N60]

'''Contour plots - see note about base_r above'''

g = gplot.getSinglePlotter(chain_dir=CHAIN_DIR)
roots = ['base_r_plikHM_TTTEEE_lowl_lowE_lensing', 'base_r_plikHM_TTTEEE_lowl_lowE_BK15_lensing']

g.plot_2d(roots, 'ns', 'r', filled = True, shaded = False,
          colors = ['grey', 'blue'])

# MatPlotLib Plots
plt.plot(specIndex, rConCon, linestyle = '-', color = 'black')

point1 = [xvalsArr[0], yvalsArr[0]]
point2 = [xvalsArr[1], yvalsArr[1]]
point3 = [xvalsArr[2], yvalsArr[2]]
point4 = [xvalsArr[3], yvalsArr[3]]
point5 = [xvalsArr[4], yvalsArr[4]]
point6 = [xvalsArr[5], yvalsArr[5]]
point7 = [xvalsArr[6], yvalsArr[6]]
point8 = [xvalsArr[7], yvalsArr[7]]
point9 = [xvalsArr[8], yvalsArr[8]]
point10 = [xvalsArr[9], yvalsArr[9]]
x1 = [point1[0], point2[0]]
y1 = [point1[1], point2[1]]
x2 = [point3[0], point4[0]]
y2 = [point3[1], point4[1]]
x3 = [point5[0], point6[0]]
y3 = [point5[1], point6[1]]
x4 = [point7[0], point8[0]]
y4 = [point7[1], point8[1]]
x5 = [point9[0], point10[0]]
y5 = [point9[1], point10[1]]
plt.plot(x1, y1, linestyle="-", color = 'black', lw = 2)
plt.plot(x1, y1, linestyle="-", color = 'gold', lw = 1, label = r'$V \propto \phi^{2}$')
plt.plot(x3, y3, linestyle="-", color = 'black', lw = 2)
plt.plot(x3, y3, linestyle="-", color = 'orange', lw = 1, label = r'$V \propto \phi^{4/3}$')
plt.plot(x2, y2, linestyle="-", color = 'black', lw = 2)
plt.plot(x2, y2, linestyle="-", color = 'orangered', lw = 1, label = r'$V \propto \phi$')
plt.plot(x4, y4, linestyle="-", color = 'black', lw = 2)
plt.plot(x4, y4, linestyle="-", color = 'red', lw = 1, label = r'$V \propto \phi^{2/3}$')
plt.plot(x5, y5, linestyle="-", color = 'black', lw = 2)
plt.plot(specIndex, rExp, linestyle = 'dashed', color = 'black', label = 'Power-law inflation')
plt.plot(x5, y5, linestyle="-", color = 'lime', lw = 1, label = r'$R^{2}$ inflation')

# Alpha attractor borders
plt.plot(nsAA_N60, rAA_N60, color='yellow', lw=1.5)
plt.plot(nsAA_N50, rAA_N50, color='yellow', lw=1.5)

plt.plot(xvalsArr[0], yvalsArr[0], marker='.', markersize='10', color='gold', mec = 'black', mew = 0.5)
plt.plot(xvalsArr[1], yvalsArr[1], marker='.', markersize='15', color='gold', mec = 'black', mew = 0.5)
plt.plot(xvalsArr[4], yvalsArr[4], marker='.', markersize='10', mec = 'black', mew = 0.5, color='orange')
plt.plot(xvalsArr[5], yvalsArr[5], marker='.', markersize='15', mec = 'black', mew = 0.5, color='orange')
plt.plot(xvalsArr[2], yvalsArr[2], marker='.', markersize='10', mec = 'black', mew = 0.5, color='orangered')
plt.plot(xvalsArr[3], yvalsArr[3], marker='.', markersize='15', mec = 'black', mew = 0.5, color='orangered')
plt.plot(xvalsArr[6], yvalsArr[6], marker='.', markersize='10', mec = 'black', mew = 0.5, color='red')
plt.plot(xvalsArr[7], yvalsArr[7], marker='.', markersize='15', mec = 'black', mew = 0.5, color='red')
plt.plot(xvalsArr[8], yvalsArr[8], marker='.', markersize='10', mec = 'black', mew = 0.5, color='lime')
plt.plot(xvalsArr[9], yvalsArr[9], marker='.', markersize='15', mec = 'black', mew = 0.5, color='lime')

plt.plot(0,0, marker = '.',markersize='10',color = 'black',label = r'$N_{\ast} = 50$')
plt.plot(0,0.1, marker = '.',markersize='15',color = 'black',label = r'$N_{\ast} = 60$')

# Plots the filled alpha attractor range based on limits, uses matplotlib.patches Polygon
upper = np.column_stack([nsAA_N50, rAA_N50])
lower = np.column_stack([nsAA_N60, rAA_N60])   
verts = np.vstack([upper, lower[::-1]])        
ax = plt.gca()
ax.add_patch(Polygon(verts, closed=True, facecolor='yellow', alpha=0.3,
                     label=r'$\alpha$-attractors'))

plt.ylim(0,0.2)
plt.xlim(0.935,1)

plt.xlabel('Spectral index $(n_{s})$')
plt.ylabel('Tensor-to-scalar ratio $(r_{0.002})$')

# Plot text above geometry dividing line
plt.text(0.9365, 0.158, 'Concave', color = 'black', rotation = 327, rotation_mode = 'anchor')
plt.text(0.9365, 0.175, 'Convex', color = 'black', rotation = 327, rotation_mode = 'anchor')

# Uses matplotlib.patches to render both legends at once
mpl_handles, mpl_labels = plt.gca().get_legend_handles_labels()

contour_proxies = [
    Patch(facecolor='grey', alpha=0.5, label='TT,TE,EE + lowE (Lensed)'),
    Patch(facecolor='blue', alpha=0.5, label='TT,TE,EE + lowE + BK15 (Lensed)'),
    ]

plt.gca().legend(contour_proxies + mpl_handles, ['TT,TE,EE + lowE (Lensed)', 'TT,TE,EE + lowE + BK15 (Lensed)'] + mpl_labels,
    loc='center left', bbox_to_anchor=(1, 0.5))

# gcf() included since getdist creates its own figure
plt.gcf().savefig(os.path.join(FIGURES_DIR, "ns_r_inflation_models.png"),
                  dpi=300, bbox_inches="tight")
plt.show()
