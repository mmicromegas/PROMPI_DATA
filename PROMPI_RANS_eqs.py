import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import CALCULUS as calc

# theoretical foundation https://arxiv.org/abs/1401.5176

class PROMPI_eqs(calc.CALCULUS,object):

    def __init__(self,filename,ig,intc,LGRID,lc):
        super(PROMPI_eqs,self).__init__(ig) 
	
        # load data to structured array
        eht = np.load(filename)	
		
        self.lgrid = LGRID
        self.intc = intc
        self.lc = lc
		
        # assign data and if needed convert to numpy array
        self.nx      = eht.item().get('nx')
        self.ny      = eht.item().get('ny')
        self.nz      = eht.item().get('nz')	
        self.timec     = eht.item().get('timec')[intc] 
        self.tavg      = np.asarray(eht.item().get('tavg')) 
        self.trange    = np.asarray(eht.item().get('trange')) 		
        self.xzn0      = np.asarray(eht.item().get('rr'))
        self.xznl      = np.asarray(eht.item().get('xznl'))
        self.xznr      = np.asarray(eht.item().get('xznr'))  		
        self.dd        = np.asarray(eht.item().get('dd')[intc])
        self.ux        = np.asarray(eht.item().get('ux')[intc])	
        self.mm        = np.asarray(eht.item().get('mm')[intc])
        self.pp        = np.asarray(eht.item().get('pp')[intc])
        self.tt        = np.asarray(eht.item().get('tt')[intc])		
        self.ddux      = np.asarray(eht.item().get('ddux')[intc])
        self.dduy      = np.asarray(eht.item().get('dduy')[intc])
        self.dduz      = np.asarray(eht.item().get('dduz')[intc])
        self.ddhh      = np.asarray(eht.item().get('ddhh')[intc])	
        self.ddcp      = np.asarray(eht.item().get('ddcp')[intc])
        self.ddhhux    = np.asarray(eht.item().get('ddhhux')[intc])
        self.ddttsq    = np.asarray(eht.item().get('ddttsq')[intc])
        self.ddtt      = np.asarray(eht.item().get('ddtt')[intc])		
        self.dduxux    = np.asarray(eht.item().get('dduxux')[intc])
        self.dduxuy    = np.asarray(eht.item().get('dduxuy')[intc])
        self.dduxuz    = np.asarray(eht.item().get('dduxuz')[intc])		
        self.dduyuy    = np.asarray(eht.item().get('dduyuy')[intc])
        self.dduzuz    = np.asarray(eht.item().get('dduzuz')[intc])
        self.ppdivu    = np.asarray(eht.item().get('ppdivu')[intc])
        self.divu      = np.asarray(eht.item().get('divu')[intc])
        self.ppdivu    = np.asarray(eht.item().get('ppdivu')[intc])		
        self.ddei      = np.asarray(eht.item().get('ddei')[intc])		
        self.ddek      = np.asarray(eht.item().get('ddek')[intc])		
        self.ddss      = np.asarray(eht.item().get('ddss')[intc])		
        self.ddekux    = np.asarray(eht.item().get('ddekux')[intc])	
        self.ddeiux    = np.asarray(eht.item().get('ddeiux')[intc])	
        self.ddssux    = np.asarray(eht.item().get('ddssux')[intc])		
        self.ppux      = np.asarray(eht.item().get('ppux')[intc])	
        self.enuc1     = np.asarray(eht.item().get('enuc1')[intc])
        self.enuc2     = np.asarray(eht.item().get('enuc2')[intc])	
        self.ddenuc1     = np.asarray(eht.item().get('ddenuc1')[intc])
        self.ddenuc2     = np.asarray(eht.item().get('ddenuc2')[intc])		
        self.gg        = np.asarray(eht.item().get('gg')[intc])	  

        # print(self.gg)		
        # self.gam1      = np.asarray(eht.item().get('gam1')[intc])# gam1 not stored yet		
	
        print('####################################')
        print('Plotting RANS for central time (in s): ',round(self.timec,1))
        print('####################################')	
        print('Averaging windows (in s): ',self.tavg)
        print('Time range (in s from-to): ',round(self.trange[0],1),round(self.trange[1],1))		
        print('####################################')		

        # store time series for time derivatives
        self.t_timec   = np.asarray(eht.item().get('timec'))
        self.t_mm      = np.asarray(eht.item().get('mm')) 		
        self.t_dd      = np.asarray(eht.item().get('dd')) 
        self.t_ddux    = np.asarray(eht.item().get('ddux')) 
        self.t_dduy    = np.asarray(eht.item().get('dduy'))
        self.t_dduz    = np.asarray(eht.item().get('dduz'))		
        self.t_ddei    = np.asarray(eht.item().get('ddei'))
        self.t_ddss    = np.asarray(eht.item().get('ddss'))		
		
		
        self.t_dduxux = np.asarray(eht.item().get('dduxux'))
        self.t_dduyuy = np.asarray(eht.item().get('dduyuy'))
        self.t_dduzuz = np.asarray(eht.item().get('dduzuz'))
		
        self.t_uxfuxf = self.t_dduxux/self.t_dd - self.t_ddux*self.t_ddux/(self.t_dd*self.t_dd)
        self.t_uyfuyf = self.t_dduyuy/self.t_dd - self.t_dduy*self.t_dduy/(self.t_dd*self.t_dd)
        self.t_uzfuzf = self.t_dduzuz/self.t_dd - self.t_dduz*self.t_dduz/(self.t_dd*self.t_dd)
		
        self.t_tke = 0.5*(self.t_uxfuxf+self.t_uyfuyf+self.t_uzfuzf)
		
        #####################
        # CONTINUITY EQUATION 
        #####################
				
        # LHS dq/dt 		
        self.dtdd = self.dt(self.t_dd,self.xzn0,self.t_timec,intc)
        # LHS ux Grad dd
        #self.uxgraddd = vexp*self.Grad(self.dd,self.xzn0)
        self.uxgraddd = (self.ddux/self.dd)*self.Grad(self.dd,self.xzn0)
				
        # RHS dd Div ux 
        self.dddivux = self.dd*self.Div(self.ddux/self.dd,self.xzn0) 

        # res
        self.resContEquation = - self.dtdd - self.uxgraddd - self.dddivux
		
        #########################	
        # END CONTINUITY EQUATION
        #########################
		
        #####################
        # R MOMENTUM EQUATION 
        #####################

        # LHS dq/dt 		
        self.dtddux = self.dt(self.t_ddux,self.xzn0,self.t_timec,intc)
     
        # LHS div rho ux ux
        self.divdduxux = self.Div(self.ddux*self.ddux/self.dd,self.xzn0)	 
		
        # RHS div Rxx
        self.divrxx = self.Div(self.dd*(self.dduxux/self.dd - self.ddux*self.ddux/(self.dd*self.dd)),self.xzn0)
		
        # RHS G
        self.geom = -(self.dduyuy+self.dduzuz)/self.xzn0
		
        # RHS grad P - rho g
        self.gradpmrhog = self.Grad(self.pp,self.xzn0) - self.dd*self.gg		
		
        # res
        self.resResMomentumEquation = -self.dtddux - self.divdduxux - self.divrxx - self.geom - self.gradpmrhog 
		
        #########################
        # END R MOMENTUM EQUATION 
        #########################		
		
		
        ###################################
        # TURBULENT KINETIC ENERGY EQUATION 
        ###################################        
		
        uxfuxf = (self.dduxux/self.dd - self.ddux*self.ddux/(self.dd*self.dd)) 
        uyfuyf = (self.dduyuy/self.dd - self.dduy*self.dduy/(self.dd*self.dd)) 
        uzfuzf = (self.dduzuz/self.dd - self.dduz*self.dduz/(self.dd*self.dd)) 		
		
        self.tke = 0.5*(uxfuxf + uyfuyf + uzfuzf)
        #self.tke = self.t_tke[:,intc]

        # LHS dq/dt 			
        self.dtddtke = self.dt(self.t_dd*self.t_tke,self.xzn0,self.t_timec,intc)
        # LHS div dd ux k
        self.divdduxtke = self.Div(self.ddux*self.tke,self.xzn0)
		
        # RHS 
        # warning ax = overline{+u''_x} 
        self.ax = self.ux - self.ddux/self.dd		
		
        # buoyancy work
        self.wb = self.ax*self.Grad(self.pp,self.xzn0)
		
        # pressure dilatation
        self.wp = self.ppdivu-self.pp*self.divu
		
        # div kinetic energy flux
        self.divfekx  = self.Div(self.dd*(self.ddekux/self.dd - (self.ddux/self.dd)*(self.ddek/self.dd)),self.xzn0)

        # div acoustic flux		
        self.divfpx = self.Div(self.ppux - self.pp*self.ux,self.xzn0)
		
        # R grad u
		
        fht_rxx = self.dd*(self.dduxux/self.dd - self.ddux*self.ddux/(self.dd*self.dd))
        fht_rxy = self.dd*(self.dduxuy/self.dd - self.ddux*self.dduy/(self.dd*self.dd))
        fht_rxz = self.dd*(self.dduxuz/self.dd - self.ddux*self.dduz/(self.dd*self.dd))
		
        self.rgradu = fht_rxx*self.Grad(self.ddux/self.dd,self.xzn0) + \
                      fht_rxy*self.Grad(self.dduy/self.dd,self.xzn0) + \
                      fht_rxz*self.Grad(self.dduz/self.dd,self.xzn0)
		
		
        self.resTkeEquation = - self.dtddtke - self.divdduxtke + self.wb + self.wp - self.divfekx - self.divfpx - self.rgradu
		
        #######################################
        # END TURBULENT KINETIC ENERGY EQUATION 
        #######################################  

        ##########################
        # INTERNAL ENERGY EQUATION 
        ##########################

        # LHS dq/dt 		
        self.dtddei = self.dt(self.t_ddei,self.xzn0,self.t_timec,intc)	

        # LHS div dd ux ei		
        self.divdduxei = self.Div(self.ddux*self.ddei/self.dd,self.xzn0)
		
        # RHS div fI
        self.divfI = self.Div(self.dd*(self.ddeiux/self.dd - self.ddux*self.ddei/(self.dd*self.dd)),self.xzn0)
		
        # RHS div fT (not included)
        self.divFT = np.zeros(self.nx)
		
        # RHS P d
        self.Pd = self.pp*self.Div(self.ux,self.xzn0)		
				
        # RHS Wp
        self.wp = self.ppdivu - self.pp*self.divu
		
        # RHS source dd enuc
        self.ddenuc = self.dd*(self.ddenuc1/self.dd+self.ddenuc2/self.dd)		
		
        # RHS dissipated turbulent kinetic energy
        self.disstke = self.resTkeEquation  	

        self.resEiEquation = -self.dtddei - self.divdduxei - self.divfI - self.divFT - self.Pd - self.wp + self.ddenuc + self.disstke
		
        ##############################
        # END INTERNAL ENERGY EQUATION 
        ##############################	
		
        ##########################
        # ENTROPY EQUATION 
        ##########################
		
        # LHS dq/dt 		
        self.dtddss = self.dt(self.t_ddss,self.xzn0,self.t_timec,intc)	

        # LHS div dd ux ss		
        self.divdduxss = self.Div(self.ddux*self.ddss/self.dd,self.xzn0)		
		
        # RHS div fs
        self.divfs = self.Div(self.dd*(self.ddssux/self.dd - self.ddux*self.ddss/(self.dd*self.dd)),self.xzn0)		
		
        # RHS div fT / T (not included)
        self.divfTT = np.zeros(self.nx)
		
        # RHS rho enuc / T
        self.ddenucT = self.dd*(self.ddenuc1/self.dd+self.ddenuc2/self.dd)/self.tt
		
        # RHS diss tke / T
        self.disstkeT = self.resTkeEquation/self.tt
	
        self.resSEquation = -self.dtddss - self.divdduxss - self.divfs - self.divfTT + self.ddenucT + self.disstkeT
	
        ##############################
        # END ENTROPY EQUATION 
        ##############################			
		
		
		
    def plot_rho(self,xbl,xbr,data_prefix):
        """Plot rho stratification in the model""" 
		
        # load x GRID
        grd1 = self.xzn0
	
        # load DATA to plot
        plt1 = self.dd
		
        # calculate INDICES for grid boundaries 
        idxl, idxr = self.idx_bndry(xbl,xbr)
		
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		
		
        # limit x/y axis
        if self.lgrid == 1:
            plt.axis([xbl,xbr,np.min(plt1[idxl:idxr]),np.max(plt1[idxl:idxr])])
        else:
            plt.axis([grd1[0],grd1[-1],np.min(plt1[0:-1]),np.max(plt1[0:-1])])	
		
        # plot DATA 
        plt.title('density')
        plt.plot(grd1,plt1,color='brown',label = r'$\overline{\rho}$')

        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"$\overline{\rho}$ (g cm$^{-3}$)"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=7,prop={'size':18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+data_prefix+'mean_rho.png')
	
    def plot_continuity_equation(self,xbl,xbr,data_prefix):
        """Plot continuity equation in the model""" 
		
        # load x GRID
        grd1 = self.xzn0

        lhs0 = - self.dtdd
        lhs1 = - self.uxgraddd
		
        rhs0 = - self.dddivux
		
        res = - self.resContEquation
		
        # calculate INDICES for grid boundaries 
        idxl, idxr = self.idx_bndry(xbl,xbr)
		
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		
		
        # limit x/y axis by global min/max from all terms
        if self.lgrid == 1:
            minx = np.min([np.min(lhs0[idxl:idxr]),np.min(lhs1[idxl:idxr]),np.min(rhs0[idxl:idxr]),np.min(res[idxl:idxr])])
            maxx = np.max([np.max(lhs0[idxl:idxr]),np.max(lhs1[idxl:idxr]),np.max(rhs0[idxl:idxr]),np.max(res[idxl:idxr])])
            plt.axis([xbl,xbr,minx,maxx])
        else:
            minx = np.min([np.min(lhs0[0:-1]),np.min(lhs1[0:-1]),np.min(rhs0[0:-1]),np.min(res[0:-1])])
            maxx = np.max([np.max(lhs0[0:-1]),np.max(lhs1[0:-1]),np.max(rhs0[0:-1]),np.max(res[0:-1])])			
            plt.axis([grd1[0],grd1[-1],minx,maxx])
		
        # plot DATA 
        plt.title('continuity equation')
        plt.plot(grd1,lhs0,color='g',label = r'$-\partial_t (\overline{\rho})$')
        plt.plot(grd1,lhs1,color='r',label = r'$- \widetilde{u}_r \partial_r (\overline{\rho})$')		
        plt.plot(grd1,rhs0,color='b',label=r'$-\overline{\rho} \nabla_r (\widetilde{u}_r)$')
        plt.plot(grd1,res,color='k',linestyle='--',label='res')

        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"g cm$^{-3}$ s$^{-1}$"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=1,prop={'size':12})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+data_prefix+'continuity_eq.png')

    def plot_continuity_equation_bar(self,xbl,xbr,data_prefix):
        """Plot continuity equation in the model""" 
		
        # load x GRID
        grd1 = self.xzn0

        term1 = - self.dtdd
        term2 = - self.uxgraddd
        term3 = - self.dddivux
        term4 = - self.resContEquation
		
        # calculate INDICES for grid boundaries 
        idxl, idxr = self.idx_bndry(xbl,xbr)
		
        term1_sel = term1[idxl:idxr]
        term2_sel = term2[idxl:idxr]
        term3_sel = term3[idxl:idxr]
        term4_sel = term4[idxl:idxr]
   
        rc = self.xzn0[idxl:idxr]

        Sr = 4.*np.pi*rc**2

        int_term1 = integrate.simps(term1_sel*Sr,rc)
        int_term2 = integrate.simps(term2_sel*Sr,rc)
        int_term3 = integrate.simps(term3_sel*Sr,rc) 
        int_term4 = integrate.simps(term4_sel*Sr,rc)     

        fig = plt.figure(figsize=(7,6))
    
        ax = fig.add_subplot(1,1,1)
        ax.yaxis.grid(color='gray', linestyle='dashed')
        ax.xaxis.grid(color='gray', linestyle='dashed')

#        plt.ylim([-80.,80.])
     
        fc = 1.
    
        # note the change: I'm only supplying y data.
        y = [int_term1/fc,int_term2/fc,int_term3/fc,int_term4/fc]

        # Calculate how many bars there will be
        N = len(y)
 
        # Generate a list of numbers, from 0 to N
        # This will serve as the (arbitrary) x-axis, which
        # we will then re-label manually.
        ind = range(N)
 
        # See note below on the breakdown of this command
        ax.bar(ind, y, facecolor='#0000FF',
               align='center', ecolor='black')
 
        #Create a y label
        ax.set_ylabel(r'g s$^{-1}$')
 
        # Create a title, in italics
#        ax.set_title('continuity equation')
 
        # This sets the ticks on the x axis to be exactly where we put
        # the center of the bars.
        ax.set_xticks(ind)
 
        # Labels for the ticks on the x axis.  It needs to be the same length
        # as y (one label for each bar)
        group_labels = [r"$-\overline{\rho} \nabla_r \widetilde{u}_r$",r"$-\partial_t \overline{\rho}$",r"$-\widetilde{u}_r \partial_r \overline{\rho}$",'res']
                         
        # Set the x tick labels to the group_labels defined above.
        ax.set_xticklabels(group_labels,fontsize=16)
 
        # Extremely nice function to auto-rotate the x axis labels.
        # It was made for dates (hence the name) but it works
        # for any long x tick labels
        fig.autofmt_xdate()
        
        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+data_prefix+'continuity_eq_bar.png')
	
    def plot_ux(self,xbl,xbr,data_prefix):
        """Plot Favrian ux stratification in the model""" 
		
        # load x GRID
        grd1 = self.xzn0
	
        # load DATA to plot
        plt1 = self.ddux/self.dd
		
        # calculate INDICES for grid boundaries 
        idxl, idxr = self.idx_bndry(xbl,xbr)
		
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		
		
        # limit x/y axis
        if self.lgrid == 1:
            plt.axis([xbl,xbr,np.min(plt1[idxl:idxr]),np.max(plt1[idxl:idxr])])
        else:
            plt.axis([grd1[0],grd1[-1],np.min(plt1[0:-1]),np.max(plt1[0:-1])])	
		
        # plot DATA 
        plt.title('ux')
        plt.plot(grd1,plt1,color='brown',label = r'$\widetilde{u}_x$')

        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"$\widetilde{u}_x$ (cm s$^{-1}$)"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=7,prop={'size':18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+data_prefix+'mean_ux.png')
	
    def plot_Rmomentum_equation(self,xbl,xbr,data_prefix):
        """Plot continuity equation in the model""" 
		
        # load x GRID
        grd1 = self.xzn0

        lhs0 = - self.dtddux
        lhs1 = - self.divdduxux
		
        rhs0 = - self.divrxx
        rhs1 = - self.geom
        rhs2 = - self.gradpmrhog 
		
        res = - self.resResMomentumEquation
		
        # calculate INDICES for grid boundaries 
        idxl, idxr = self.idx_bndry(xbl,xbr)
		
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		
		
        # limit x/y axis by global min/max from all terms
        if self.lgrid == 1:
            minx = np.min([np.min(lhs0[idxl:idxr]),np.min(lhs1[idxl:idxr]),np.min(rhs0[idxl:idxr]),\
			np.min(rhs1[idxl:idxr]),np.min(rhs2[idxl:idxr]),np.min(res[idxl:idxr])])
            maxx = np.max([np.max(lhs0[idxl:idxr]),np.max(lhs1[idxl:idxr]),np.max(rhs0[idxl:idxr]),\
			np.max(rhs1[idxl:idxr]),np.max(rhs2[idxl:idxr]),np.max(res[idxl:idxr])])
            plt.axis([xbl,xbr,minx,maxx])
        else:
            minx = np.min([np.min(lhs0[0:-1]),np.min(lhs1[0:-1]),np.min(rhs0[0:-1]),np.min(rhs1[0:-1]),np.min(rhs2[0:-1]),np.min(res[0:-1])])
            maxx = np.max([np.max(lhs0[0:-1]),np.max(lhs1[0:-1]),np.max(rhs0[0:-1]),np.max(rhs1[0:-1]),np.max(rhs2[0:-1]),np.max(res[0:-1])])			
            plt.axis([grd1[0],grd1[-1],minx,maxx])
		
        # plot DATA 
        plt.title('r momentum equation')
        plt.plot(grd1,lhs0,color='c',label = r"$-\partial_t ( \overline{\rho} \widetilde{u}_r ) $")
        plt.plot(grd1,lhs1,color='m',label = r"$-\nabla_r (\overline{\rho} \widetilde{u}_r \widetilde{u}_r ) $")		
        plt.plot(grd1,rhs0,color='b',label=r"$-\nabla_r (\widetilde{R}_{rr})$")
        plt.plot(grd1,rhs1,color='g',label=r"$-\overline{G^{M}_r}$")
        plt.plot(grd1,rhs2,color='r',label=r"$-(\partial_r \overline{P} - \bar{\rho}\tilde{g}_r)$")		
        plt.plot(grd1,res,color='k',linestyle='--',label='res')

        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"g cm$^{-2}$  s$^{-2}$"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=1,prop={'size':12})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+data_prefix+'rmomentum_eq.png')	
		
    def plot_tke(self,xbl,xbr,data_prefix):
        """Plot turbulent kinetic energy stratification in the model""" 
		
        # load x GRID
        grd1 = self.xzn0
	
        # load DATA to plot 		
        plt1 = self.tke
		
        # calculate INDICES for grid boundaries 
        idxl, idxr = self.idx_bndry(xbl,xbr)
		
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		
		
        # limit x/y axis
        if self.lgrid == 1:
            plt.axis([xbl,xbr,np.min(plt1[idxl:idxr]),np.max(plt1[idxl:idxr])])
        else:
            plt.axis([grd1[0],grd1[-1],np.min(plt1[0:-1]),np.max(plt1[0:-1])])	
		
        # plot DATA 
        plt.title('turbulent kinetic energy')
        plt.plot(grd1,plt1,color='brown',label = r'$\frac{1}{2} \widetilde{u''}_i \widetilde{u''}_i$')

        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"$\widetilde{k}$ (erg g$^{-1}$)"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=1,prop={'size':18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+data_prefix+'mean_tke.png')		

    def plot_tke_equation(self,xbl,xbr,data_prefix):
        """Plot turbulent kinetic energy equation in the model""" 
		
        # load x GRID
        grd1 = self.xzn0

        lhs0 = - self.dtddtke
        lhs1 = - self.divdduxtke
		
        rhs0 = + self.wb
        rhs1 = + self.wp		
        rhs2 = - self.divfekx
        rhs3 = - self.divfpx
        rhs4 = - self.rgradu
		
        res = self.resTkeEquation
		
        # calculate INDICES for grid boundaries 
        idxl, idxr = self.idx_bndry(xbl,xbr)
		
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		
		
        # limit x/y axis by global min/max from all terms
        if self.lgrid == 1:
            minx = np.min([np.min(lhs0[idxl:idxr]),np.min(lhs1[idxl:idxr]),np.min(rhs0[idxl:idxr]),\
			np.min(rhs1[idxl:idxr]),np.min(rhs2[idxl:idxr]),np.min(rhs3[idxl:idxr]),np.min(rhs4[idxl:idxr]),np.min(res[idxl:idxr])])
            maxx = np.max([np.max(lhs0[idxl:idxr]),np.max(lhs1[idxl:idxr]),np.max(rhs0[idxl:idxr]),\
			np.max(rhs1[idxl:idxr]),np.max(rhs2[idxl:idxr]),np.max(rhs3[idxl:idxr]),np.max(rhs4[idxl:idxr]),np.max(res[idxl:idxr])])
            plt.axis([xbl,xbr,minx,maxx])
        else:
            minx = np.min([np.min(lhs0[0:-1]),np.min(lhs1[0:-1]),np.min(rhs0[0:-1]),np.min(rhs1[0:-1]),\
			np.min(rhs2[0:-1]),np.min(rhs3[0:-1]),np.min(rhs4[0:-1]),np.min(res[0:-1])])
            maxx = np.max([np.max(lhs0[0:-1]),np.max(lhs1[0:-1]),np.max(rhs0[0:-1]),np.max(rhs1[0:-1]),\
			np.max(rhs2[0:-1]),np.max(rhs3[0:-1]),np.max(rhs4[0:-1]),np.max(res[0:-1])])			
            plt.axis([grd1[0],grd1[-1],minx,maxx])
		
        # plot DATA 
        plt.title('turbulent kinetic energy equation')
        plt.plot(grd1,-lhs0,color='#FF6EB4',label = r'$-\partial_t (\widetilde{k})$')
        plt.plot(grd1,-lhs1,color='k',label = r"$-\nabla_r (\overline{\rho} \widetilde{u}_r \widetilde{k})$")	
		
        plt.plot(grd1,rhs0,color='r',label = r'$+W_b$')     
        plt.plot(grd1,rhs1,color='c',label = r'$+W_p$') 
        plt.plot(grd1,rhs2,color='#802A2A',label = r"$-\nabla_r f_k$") 
        plt.plot(grd1,rhs3,color='m',label = r"$-\nabla_r f_P$")
        plt.plot(grd1,rhs4,color='b',label = r"$-\widetilde{R}_{ri}\partial_r \widetilde{u_i}$")		
        plt.plot(grd1,res,color='k',linestyle='--',label=r"res $\sim N_k$")
 
        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"erg cm$^{-3}$ s$^{-1}$"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=1,prop={'size':12})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+data_prefix+'tke_eq.png')		
	
    def plot_ei(self,xbl,xbr,data_prefix):
        """Plot mean Favrian internal energy stratification in the model""" 
		
        # load x GRID
        grd1 = self.xzn0
	
        # load DATA to plot
        plt1 = self.ddei/self.dd
		
        # calculate INDICES for grid boundaries 
        idxl, idxr = self.idx_bndry(xbl,xbr)
		
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		
		
        # limit x/y axis
        if self.lgrid == 1:
            plt.axis([xbl,xbr,np.min(plt1[idxl:idxr]),np.max(plt1[idxl:idxr])])
        else:
            plt.axis([grd1[0],grd1[-1],np.min(plt1[0:-1]),np.max(plt1[0:-1])])	
		
        # plot DATA 
        plt.title(r'internal energy')
        plt.plot(grd1,plt1,color='brown',label = r'$\widetilde{\varepsilon}_I$')

        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"$\widetilde{\varepsilon}_I$ (erg g$^{-1}$)"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=1,prop={'size':18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+data_prefix+'mean_ei.png')
	

    def plot_ei_equation(self,xbl,xbr,data_prefix):
        """Plot internal energy equation in the model""" 
		
        # load x GRID
        grd1 = self.xzn0

        lhs0 = - self.dtddei
        lhs1 = - self.divdduxei
		
        rhs0 = - self.divfI
        rhs1 = - self.divFT		
        rhs2 = - self.Pd
        rhs3 = - self.wp
        rhs4 = + self.ddenuc
        rhs5 = + self.disstke
		
        res = self.resEiEquation
		
        # calculate INDICES for grid boundaries 
        idxl, idxr = self.idx_bndry(xbl,xbr)
		
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		
		
        # limit x/y axis by global min/max from all terms
        if self.lgrid == 1:
            minx = np.min([np.min(lhs0[idxl:idxr]),np.min(lhs1[idxl:idxr]),np.min(rhs0[idxl:idxr]),\
			np.min(rhs1[idxl:idxr]),np.min(rhs2[idxl:idxr]),np.min(rhs3[idxl:idxr]),np.min(rhs4[idxl:idxr]),np.min(rhs5[idxl:idxr]),np.min(res[idxl:idxr])])
            maxx = np.max([np.max(lhs0[idxl:idxr]),np.max(lhs1[idxl:idxr]),np.max(rhs0[idxl:idxr]),\
			np.max(rhs1[idxl:idxr]),np.max(rhs2[idxl:idxr]),np.max(rhs3[idxl:idxr]),np.max(rhs4[idxl:idxr]),np.max(rhs5[idxl:idxr]),np.max(res[idxl:idxr])])
            plt.axis([xbl,xbr,minx,maxx])
        else:
            minx = np.min([np.min(lhs0[0:-1]),np.min(lhs1[0:-1]),np.min(rhs0[0:-1]),np.min(rhs1[0:-1]),\
			np.min(rhs2[0:-1]),np.min(rhs3[0:-1]),np.min(rhs4[0:-1]),np.min(rhs5[0:-1]),np.min(res[0:-1])])
            maxx = np.max([np.max(lhs0[0:-1]),np.max(lhs1[0:-1]),np.max(rhs0[0:-1]),np.max(rhs1[0:-1]),\
			np.max(rhs2[0:-1]),np.max(rhs3[0:-1]),np.max(rhs4[0:-1]),np.max(rhs5[0:-1]),np.max(res[0:-1])])			
            plt.axis([grd1[0],grd1[-1],minx,maxx])
		
        # plot DATA 
        plt.title('internal energy equation')
        plt.plot(grd1,lhs0,color='#FF6EB4',label = r"$-\partial_t (\overline{\rho} \widetilde{\epsilon}_I )$")
        plt.plot(grd1,lhs1,color='k',label = r"$-\nabla_r (\overline{\rho}\widetilde{u}_r \widetilde{\epsilon}_I$)")	
		
        plt.plot(grd1,rhs0,color='#FF8C00',label = r"$-\nabla_r f_I $")     
        plt.plot(grd1,rhs1,color='c',label = r"$-\nabla_r f_T$ (not incl.)") 
        plt.plot(grd1,rhs2,color='#802A2A',label = r"$-\bar{P} \bar{d}$") 
        plt.plot(grd1,rhs3,color='r',label = r"$-W_P$")
        plt.plot(grd1,rhs4,color='b',label = r"$+\overline{\rho}\widetilde{\epsilon}_{nuc}$")
        plt.plot(grd1,rhs5,color='m',label = r"$+\varepsilon_k$")
		
        plt.plot(grd1,res,color='k',linestyle='--',label=r"res $\sim N_\epsilon$")
 
        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"erg cm$^{-3}$ s$^{-1}$"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=1,prop={'size':12})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+data_prefix+'ei_eq.png')
		
    def plot_ss(self,xbl,xbr,data_prefix):
        """Plot mean Favrian entropy stratification in the model""" 
		
        # load x GRID
        grd1 = self.xzn0
	
        # load DATA to plot
        plt1 = self.ddss/self.dd
		
        # calculate INDICES for grid boundaries 
        idxl, idxr = self.idx_bndry(xbl,xbr)
		
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		
		
        # limit x/y axis
        if self.lgrid == 1:
            plt.axis([xbl,xbr,np.min(plt1[idxl:idxr]),np.max(plt1[idxl:idxr])])
        else:
            plt.axis([grd1[0],grd1[-1],np.min(plt1[0:-1]),np.max(plt1[0:-1])])	
		
        # plot DATA 
        plt.title(r'entropy')
        plt.plot(grd1,plt1,color='brown',label = r'$\widetilde{s}$')

        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"$\widetilde{s}$ (erg g$^{-1}$ K$^{-1}$)"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=1,prop={'size':18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+data_prefix+'mean_ss.png')
		
    def plot_ss_equation(self,xbl,xbr,data_prefix):
        """Plot entropy equation in the model""" 
		
        # load x GRID
        grd1 = self.xzn0

        lhs0 = - self.dtddss
        lhs1 = - self.divdduxss
		
        rhs0 = - self.divfs
        rhs1 = - self.divfTT		
        rhs2 = + self.ddenucT
        rhs3 = + self.disstkeT
		
        res = self.resSEquation
		
        # calculate INDICES for grid boundaries 
        idxl, idxr = self.idx_bndry(xbl,xbr)
		
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		
		
        # limit x/y axis by global min/max from all terms
        if self.lgrid == 1:
            minx = np.min([np.min(lhs0[idxl:idxr]),np.min(lhs1[idxl:idxr]),np.min(rhs0[idxl:idxr]),\
			np.min(rhs1[idxl:idxr]),np.min(rhs2[idxl:idxr]),np.min(rhs3[idxl:idxr]),np.min(res[idxl:idxr])])
            maxx = np.max([np.max(lhs0[idxl:idxr]),np.max(lhs1[idxl:idxr]),np.max(rhs0[idxl:idxr]),\
			np.max(rhs1[idxl:idxr]),np.max(rhs2[idxl:idxr]),np.max(rhs3[idxl:idxr]),np.max(res[idxl:idxr])])
            plt.axis([xbl,xbr,minx,maxx])
        else:
            minx = np.min([np.min(lhs0[0:-1]),np.min(lhs1[0:-1]),np.min(rhs0[0:-1]),np.min(rhs1[0:-1]),\
			np.min(rhs2[0:-1]),np.min(rhs3[0:-1]),np.min(res[0:-1])])
            maxx = np.max([np.max(lhs0[0:-1]),np.max(lhs1[0:-1]),np.max(rhs0[0:-1]),np.max(rhs1[0:-1]),\
			np.max(rhs2[0:-1]),np.max(rhs3[0:-1]),np.max(res[0:-1])])			
            plt.axis([grd1[0],grd1[-1],minx,maxx])
		
        # plot DATA 
        plt.title('entropy equation')
        plt.plot(grd1,lhs0,color='#FF6EB4',label = r"$-\partial_t (\overline{\rho} \widetilde{s} )$")
        plt.plot(grd1,lhs1,color='k',label = r"$-\nabla_r (\overline{\rho}\widetilde{u}_r \widetilde{s}$)")	
		
        plt.plot(grd1,rhs0,color='#FF8C00',label = r"$-\nabla_r f_s $")     
        plt.plot(grd1,rhs1,color='c',label = r"$-\overline{\nabla_r f_T /T}$ (not incl.)") 
        plt.plot(grd1,rhs2,color='b',label = r"$+\overline{\rho}\widetilde{\epsilon}_{nuc}/\overline{T}$")
        plt.plot(grd1,rhs3,color='m',label = r"$+\varepsilon_k/T$")
		
        plt.plot(grd1,res,color='k',linestyle='--',label=r"res $\sim N_s$")
 
        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"erg cm$^{-3}$ s$^{-1}$ K$^{-1}$"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=1,prop={'size':12})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+data_prefix+'ss_eq.png')
		
    def properties(self,xbl,xbr,data_prefix):
        """ Print properties of your simulation""" 

        # PROPERTIES #

        rc = self.xzn0		
		
        # get inner and outer boundary of computational domain  

        rin = self.xzn0[0]
        rout = self.xzn0[self.nx-1]

        # load TKE dissipation
        diss = self.resTkeEquation

        # calculate INDICES for grid boundaries 
        idxl, idxr = self.idx_bndry(xbl,xbr)		
		
        # Get rid of the numerical mess at inner boundary 
        diss[0:idxl] = 0.
        # Get rid of the numerical mess at outer boundary 
        diss[idxr:self.nx] = 0.
 
        diss_max = diss.max()
        ind = np.where( (diss > 0.02*diss_max) )[0]

        rinc  = rc[ind[0]]
        routc = rc[ind[-1]]

        ibot = ind[0]
        itop = ind[-1]

        Vol = 4./3.*np.pi*(self.xznr**3-self.xznl**3)

        # Calculate full dissipation rate and timescale
        TKE = (self.dd*self.tke*Vol)[ind].sum()
        epsD = (diss*Vol)[ind].sum()
        tD = TKE/epsD

        # RMS velocities
        M=(self.dd*Vol)[ind].sum()
        urms = np.sqrt(2.*TKE/M)

        # Turnover timescale
        tc = 2.*(routc-rinc)/urms

        # Dissipation length-scale
        ld = M*urms**3./epsD

        # Total nuclear luminosity
        tenuc = ((self.dd*(self.enuc1+self.enuc2))*Vol)[ind].sum()

        # Pturb over Pgas (work in progress, no gam1)
        #cs2 = (self.gam1*self.pp)/self.dd
        #ur2 = self.uxux
        #pturb_o_pgas = (self.gam1*ur2/cs2)[ind].mean()
    
        # Calculate size of convection zone in pressure scale heights

        hp = -self.pp/self.Grad(self.pp,self.xzn0)
        pbot = self.pp[ibot]
        lcz_vs_hp = np.log(pbot/self.pp[ibot:itop])

        # Reynolds number
		# todo
		
		
        print '---------------'
        print 'Resolution: %i' % self.nx,self.ny,self.nz
        print 'Radial size of computational domain (in cm): %.2e %.2e' % (rin,rout)
        print 'Radial size of convection zone (in cm):  %.2e %.2e' % (rinc,routc)
        print 'Extent of convection zone (in Hp): %f' % lcz_vs_hp[itop-ibot-1]
        print 'Averaging time window (in s): %f' % self.tavg
        print 'RMS velocities in convection zone (in cm/s):  %.2e' % urms
        print 'Convective turnover timescale (in s)  %.2e' % tc
        #print 'P_turb o P_gas %.2e' % pturb_o_pgas
        print 'Dissipation length scale (in cm): %.2e' % ld
        print 'Total nuclear luminosity (in erg/s): %.2e' % tenuc
        print 'Rate of TKE dissipation (in erg/s): %.2e' % epsD
        print 'Dissipation timescale for TKE (in s): %f' % tD
        #print 'Reynolds number: %f' % self.Re
        #print 'Dissipation timescale for radial TKE (in s): %f' % self.tD_rad
        #print 'Dissipation timescale for horizontal TKE (in s): %f' % self.tD_hor
		
		
    def gauss(x, *p): 
    # Define model function to be used to fit to the data above:
        A, mu, sigma = p
        return A*np.exp(-(x-mu)**2/(2.*sigma**2))		
	
    def idx_bndry(self,xbl,xbr):
    # calculate indices of grid boundaries 
        rr = np.asarray(self.xzn0)
        xlm = np.abs(rr-xbl)
        xrm = np.abs(rr-xbr)
        idxl = int(np.where(xlm==xlm.min())[0])
        idxr = int(np.where(xrm==xrm.min())[0])	
        return idxl,idxr
		
		
