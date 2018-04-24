import PROMPI_RANS_eqs as pre
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# CHOOSE DATA SOURCE 
eht_data = 'tseries_ransout.npy' 

# CHOOSE PREFIX for figure names in RESULTS
data_prefix = 'tseries_ransdat'+'_'

# CHOOSE GEOMETRY OF THE SIMULATION
# ig = 1 CARTESIAN x,y,z
# ig = 2 SPHERICAL r,theta,phi

ig = 2

# CHOOSE CENTRAL TIME INDEX
intc = 4


# IF LGRID = 1, LIMIT THE GRID (get rid of boundary noise)
# IF LGRID = 0, DO NOT LIMIT THE GRID
LGRID = 1

# LIMIT GRID 
xbl = 3.72e8
xbr = 9.8e8

# estimate size of convection zone
lc = 4.3e8

# INSTANTIATE 
RANS = pre.PROMPI_eqs(eht_data,ig,intc,LGRID,lc)

# PLOT CONTINUITY EQUATION 

RANS.plot_rho(xbl,xbr,data_prefix)
RANS.plot_continuity_equation(xbl,xbr,data_prefix)
RANS.plot_continuity_equation_bar(xbl,xbr,data_prefix)

# PLOT UX MOMENTUM EQUATION

RANS.plot_ux(xbl,xbr,data_prefix)
RANS.plot_Rmomentum_equation(xbl,xbr,data_prefix)

# PLOT TURBULENT KINETIC ENERGY EQUATION

RANS.plot_tke(xbl,xbr,data_prefix)
RANS.plot_tke_equation(xbl,xbr,data_prefix)

# PLOT INTERNAL ENERGY EQUATION

RANS.plot_ei(xbl,xbr,data_prefix)
RANS.plot_ei_equation(xbl,xbr,data_prefix)

# PLOT ENTROPY EQUATION

RANS.plot_ss(xbl,xbr,data_prefix)
RANS.plot_ss_equation(xbl,xbr,data_prefix)

# PRINT PROPERTIES OF YOUR SIMULATION

RANS.properties(xbl,xbr,data_prefix)
