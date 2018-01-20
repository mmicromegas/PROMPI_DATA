import PROMPI_RANS_bgr as prb

fl_rans = 'DATA/ob3d.45.lrez.00041.ransdat' 
ob_rans = prb.PROMPI_bgr(fl_rans)
ob_rans.SetMatplotlibParams()
data    = ob_rans.rans()

xbl = 3.72e8
xbr = 9.8e8

ob_rans.plot_dd_tt(xbl,xbr)
ob_rans.plot_enuc_ei(xbl,xbr)
#ob_rans.plot_dd_tt_tycho_ini()
#ob_rans.plot_x_tycho_ini()
