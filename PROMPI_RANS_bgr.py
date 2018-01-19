import PROMPI_data as prd
import numpy as np
import matplotlib.pyplot as plt

class PROMPI_bgr(prd.PROMPI_ransdat,object):

    def __init__(self,filename):
        super(PROMPI_bgr,self).__init__(filename) 
        self.data = self.rans()

    def SetMatplotlibParams(self):
        """This routine sets some standard values for matplotlib to obtain publication-quality figures"""

        # plt.rc('text',usetex=True)
        # plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
        plt.rc('font',**{'family':'serif','serif':['Times New Roman']})
        plt.rc('font',size=22.)
        plt.rc('lines',linewidth=2,markeredgewidth=2.,markersize=10)
        plt.rc('axes',linewidth=1.5)
        plt.rcParams['xtick.major.size']=8.
        plt.rcParams['xtick.minor.size']=4.
        plt.rcParams['figure.subplot.bottom']=0.15
        plt.rcParams['figure.subplot.left']=0.17		
        plt.rcParams['figure.subplot.right']=0.85		

    def idx_bndry(self,xbl,xbr):
        rr = np.asarray(self.data['xzn0'])
        xlm = np.abs(rr-xbl)
        xrm = np.abs(rr-xbr)
        idxl = int(np.where(xlm==xlm.min())[0])
        idxr = int(np.where(xrm==xrm.min())[0])	
        return idxl,idxr

    def rr2mm(self,position,rr,mm):
        rm = np.interp(position,rr,mm)
        return np.round(rm,1)

		
    def plot_dd_tt_tycho_ini(self):
        tdata = np.loadtxt('DATA/imodel.tycho',skiprows=26)
        msun = 1.989e33 # in grams
        nxmax = 500
        mm = tdata[1:nxmax,1]/msun
        rr = tdata[1:nxmax,2]/1.e8
        dd = 1./tdata[1:nxmax,5]
        tt = tdata[1:nxmax,4]


        plt.rcParams['figure.subplot.bottom']=0.13
        plt.rcParams['figure.subplot.left']=0.15
        plt.rcParams['figure.subplot.right']=0.85

        fig, ax1 = plt.subplots(figsize=(7,6))
        ax1.axis([3.72,9.8,5.2,7.])
        ax1.plot(rr,np.log10(dd),color='b',label=r'$\rho$')
        ax1.xaxis.set_ticks(np.arange(4.,10.,1.))
        ax1.set_xlabel(r'r (10$^{8}$ cm)')
        ax1.set_ylabel(r'log $\rho$ (g cm$^{-3}$)')
        ax1.legend(loc=7,prop={'size':18})

        ax2 = ax1.twinx()
        ax2.axis([3.72,9.8,9.15,9.45])
        ax2.plot(rr, np.log10(tt),color='r',label=r'$T$')
        ax2.set_ylabel(r'log T (K)')
        ax2.tick_params('y')
        ax2.legend(loc=1,prop={'size':18})

        ax3 = ax1.twiny()
        ax3.set_xlim(ax1.get_xlim())

        newxlabel=[3.7,4.7,5.5,6.4,7.2,8.5,9.5]
        ax3.set_xticks(newxlabel)
        ax3.set_xticklabels(self.rr2mm(newxlabel,rr,mm))
        ax3.set_xlabel('enclosed mass (msol)')

        ax2.axvline(x=4.3,linestyle='--',color='k',linewidth=1)
        ax2.axvline(x=7.2,linestyle='--',color='k',linewidth=1)

        ax2.text(4.5,9.42,r"initial convection",fontsize=21)
        ax2.text(5.4,9.4,r"zone",fontsize=21)

        fig.tight_layout()
        plt.show(block=False)
		
#        plt.savefig('ob3dB_initial_model_rho_t.eps')
#        plt.savefig('ob3dB_initial_model_rho_t.png')
		
	
    def plot_x_tycho_ini(self):
        tdata = np.loadtxt('DATA/imodel.tycho',skiprows=26)
        msun = 1.989e33 # in grams
        nxmax = 500
        mm = tdata[1:nxmax,1]/msun
        rr = tdata[1:nxmax,2]/1.e8

        fig, ax4 = plt.subplots(figsize=(7,6))

        n  = tdata[1:nxmax,10]
        p  = tdata[1:nxmax,11]
        he4= tdata[1:nxmax,12]
        c12= tdata[1:nxmax,13]
        o16= tdata[1:nxmax,14]
        ne20= tdata[1:nxmax,15]
        na23= tdata[1:nxmax,16]
        mg24= tdata[1:nxmax,17]
        si28= tdata[1:nxmax,18]
        p31= tdata[1:nxmax,19]
        s32= tdata[1:nxmax,20]
        s34= tdata[1:nxmax,21]
        cl35= tdata[1:nxmax,22]
        ar36= tdata[1:nxmax,23]
        ar38= tdata[1:nxmax,24]
        k39= tdata[1:nxmax,25]
        ca40= tdata[1:nxmax,26]
        ca42= tdata[1:nxmax,27]
        ti44= tdata[1:nxmax,28]
        ti46= tdata[1:nxmax,29]
        cr48= tdata[1:nxmax,30]
        cr50= tdata[1:nxmax,31]
        fe52= tdata[1:nxmax,32]
        fe54= tdata[1:nxmax,33]
        ni56= tdata[1:nxmax,34]

        ## the composition is abundance, to convert to mass fraction you need 
        ## to multiply by number of nucleons per isotope 

        # sumx = n + p + he4 + c12 + o16 + ne20 + na23 + mg24 + si28 + p31 + s32 + s34 + cl35 + ar36 + ar38 + k39 + ca40 + ca42 + ti44 + ti46 + cr48 + cr50 + fe52 + fe54 + ni56
        sumx = 1.*n + 1.*p + 4.*he4 + 12.*c12 + 16.*o16 + 20.*ne20 + 23.*na23 + 24.*mg24 + 28.*si28 + 31.*p31 + 32.*s32 + 34.*s34 + 35.*cl35 + 36.*ar36 + 38.*ar38 + 39.*k39 + 40.*ca40 + 42.*ca42 + 44.*ti44 + 46.*ti46 + 48.*cr48 + 50.*cr50 + 52.*fe52 + 54.*fe54 + 56.*ni56


        # print('SUM OF X: ',sumx)

        ax4.axis([3.72,9.8,1.e-10,1.])                                                                                               
        ax4.semilogy(rr, 12.*c12,color='r',label=r'C$^{12}$')
        ax4.semilogy(rr, 16.*o16,color='g',label=r'O$^{16}$')
        ax4.semilogy(rr, 20.*ne20,color='b',label=r'Ne$^{20}$')
        ax4.semilogy(rr, 28.*si28,color='y',label=r'Si$^{28}$')
        ax4.xaxis.set_ticks(np.arange(4.,10.,1.))

        ax4.set_xlabel(r'r (10$^{8}$ cm)')
        ax4.set_ylabel(r'mass fraction')

        ax4.text(4.5,1.e-8,r"initial convection",fontsize=21)
        ax4.text(5.4,3.e-9,r"zone",fontsize=21)

        ax4.axvline(x=4.3,linestyle='--',color='k',linewidth=1)
        ax4.axvline(x=7.2,linestyle='--',color='k',linewidth=1)

        ax4.legend(loc = (0.2, 0.5),prop={'size':18})

        ax5 = ax4.twiny()
        ax5.set_xlim(ax4.get_xlim())

        newxlabel=[3.7,4.7,5.5,6.4,7.2,8.5,9.5]
        ax5.set_xticks(newxlabel)
        ax5.set_xticklabels(self.rr2mm(newxlabel,rr,mm))
        ax5.set_xlabel('enclosed mass (msol)')
		
        plt.show(block=False)

		
		
#        plt.savefig('ob3dB_initial_model_x_new.eps')
#        plt.savefig('ob3dB_initial_model_x_new.png')
		
    def plot_dd_tt(self,xbl,xbr):
        rr = np.asarray(self.data['xzn0'])		
        f_1 = self.data['eh_dd']
        f_2 = self.data['eh_tt']
		
        xlabel_1 = r'r (10$^{8}$ cm)'
        ylabel_1 = r'log $\rho$ (g cm$^{-3}$)'	

        xlabel_2 = ''
        ylabel_2 = r'log T (K)' 		

        plabel_1 = r'$\rho$'
        plabel_2 = r'$T$'
		
        idxl, idxr = self.idx_bndry(xbl,xbr)
		
        to_plt1 = np.log10(f_1)
        to_plt2 = np.log10(f_2)
	
        fig, ax1 = plt.subplots(figsize=(7,6))
		
        ax1.axis([xbl,xbr,np.min(to_plt1[idxl:idxr]),np.max(to_plt1[idxl:idxr])])
        ax1.plot(rr,to_plt1,color='b',label = plabel_1)
#        ax1.xaxis.set_ticks(np.arange(4.e8,10.e8,1.e8))

        ax1.set_xlabel(xlabel_1)
        ax1.set_ylabel(ylabel_1)
        ax1.legend(loc=7,prop={'size':18})

        ax2 = ax1.twinx()
        ax2.axis([xbl,xbr,np.min(to_plt2[idxl:idxr]),np.max(to_plt2[idxl:idxr])])
        ax2.plot(rr, to_plt2,color='r',label = plabel_2)
        ax2.set_ylabel(ylabel_2)
        ax2.tick_params('y')
        ax2.legend(loc=1,prop={'size':18})
		
        plt.show(block=False)

    def plot_enuc_ei(self,xbl,xbr):
        rr = np.asarray(self.data['xzn0'])
        f_1 = self.data['eh_enuc1']+self.data['eh_enuc2']
        f_2 = self.data['eh_ei']
		
        xlabel_1 = r'r (10$^{8}$ cm)'
        ylabel_1 = r'erg ($s^{-1}$)'	

        xlabel_2 = ''
        ylabel_2 = r'$\epsilon$ (ergs)' 		

        plabel_1 = r'$\varepsilon_{nuc}$'
        plabel_2 = r'$\epsilon$'
		
        idxl, idxr = self.idx_bndry(xbl,xbr)
		
        to_plt1 = f_1  # change here if you need log plot
        to_plt2 = f_2  # change here if you need log plot 
	
        fig, ax1 = plt.subplots(figsize=(7,6))
		
        ax1.axis([xbl,xbr,np.min(to_plt1[idxl:idxr]),np.max(to_plt1[idxl:idxr])])
        ax1.plot(rr,to_plt1,color='b',label = plabel_1)
#        ax1.xaxis.set_ticks(np.arange(4.e8,10.e8,1.e8))

        ax1.set_xlabel(xlabel_1)
        ax1.set_ylabel(ylabel_1)
        ax1.legend(loc=7,prop={'size':18})

        ax2 = ax1.twinx()
        ax2.axis([xbl,xbr,np.min(to_plt2[idxl:idxr]),np.max(to_plt2[idxl:idxr])])
        ax2.plot(rr, to_plt2,color='r',label = plabel_2)
        ax2.set_ylabel(ylabel_2)
        ax2.tick_params('y')
        ax2.legend(loc=1,prop={'size':18})
		
        plt.show(block=False)

		
		

    def plot_x(self,xbl,xbr,rr,X):
        pass
		
