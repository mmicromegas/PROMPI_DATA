import PROMPI_RANS_bgr as prb
import warnings

warnings.filterwarnings("ignore")

#tycho_input = 'DATA/imodel.tycho'
fl_rans = 'DATA/ob3d.45.lrez.00041.ransdat' 
ob_rans = prb.PROMPI_bgr(fl_rans)

xbl = 3.72e8
xbr = 9.8e8

ob_rans.SetMatplotlibParams()

# USAGE:

ob_rans.PlotNucEnergyGen(xbl,xbr)

#ob_rans.plot_log_q1(xbl,xbr,'x0007',r'r (10$^{8}$ cm)','X','ne20')
#ob_rans.plot_log_q1(xbl,xbr,'dd',r'r (10$^{8}$ cm)','T','T')
#ob_rans.plot_lin_q1(xbl,xbr,'dd',r'r (10$^{8}$ cm)','rho','rho')

#ob_rans.plot_log_q1q2(xbl,xbr,'dd','tt',\
#                      r'r (10$^{8}$ cm)',\
#                      r'log $\rho$ (g cm$^{-3}$)',r'log T (K)',\
#                      r'$\rho$',r'$T$')

#ob_rans.plot_lin_q1q2(xbl,xbr,'enuc','ei',\
#                      r'r (10$^{8}$ cm)',\
#                      r'$\varepsilon_{nuc}$ (erg $s^{-1}$)',\
#                      r'$\epsilon$ (ergs)',\
#                      r'$\varepsilon_{nuc}$',r'$\epsilon$')
