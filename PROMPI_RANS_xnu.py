import numpy as np
import matplotlib.pyplot as plt
import CALCULUS as calc

# Theoretical background https://arxiv.org/abs/1401.5176

# Mocak, Meakin, Viallet, Arnett, 2014, Compressible Hydrodynamic Mean-Field #
# Equations in Spherical Geometry and their Application to Turbulent Stellar #
# Convection Data #

# https://github.com/mmicromegas/PROMPI_DATA/blob/master/ransXtoPROMPI.pdf

class PROMPI_xnu(calc.CALCULUS,object):

    def __init__(self,filename,ig,inuc,intc,LGRID,xbl,xbr):
        super(PROMPI_xnu,self).__init__(ig) 
	
        # load data to structured array
        eht = np.load(filename)	
		
        self.lgrid = LGRID
        self.intc = intc
        self.xbl = xbl
        self.xbr = xbr
        self.inuc = inuc
		
        # assign data and if needed convert to numpy array
        self.nx      = eht.item().get('nx')
        self.timec     = eht.item().get('timec')[intc] 
        self.tavg      = np.asarray(eht.item().get('tavg')) 
        self.trange    = np.asarray(eht.item().get('trange')) 		
        self.xzn0      = np.asarray(eht.item().get('rr')) 
        self.dd        = np.asarray(eht.item().get('dd')[intc])
        self.ux        = np.asarray(eht.item().get('ux')[intc])	
        self.pp        = np.asarray(eht.item().get('pp')[intc])
        self.ddux      = np.asarray(eht.item().get('ddux')[intc])
        self.dduy      = np.asarray(eht.item().get('dduy')[intc])
        self.dduz      = np.asarray(eht.item().get('dduz')[intc])
        self.ddhh      = np.asarray(eht.item().get('ddhh')[intc])	
        self.ddcp      = np.asarray(eht.item().get('ddcp')[intc])
        self.ddhhux    = np.asarray(eht.item().get('ddhhux')[intc])
        self.ddttsq    = np.asarray(eht.item().get('ddttsq')[intc])
        self.ddtt      = np.asarray(eht.item().get('ddtt')[intc])		
        self.dduxux    = np.asarray(eht.item().get('dduxux')[intc])
        self.dduyuy    = np.asarray(eht.item().get('dduyuy')[intc])
        self.dduzuz    = np.asarray(eht.item().get('dduzuz')[intc])
        self.xi        = np.asarray(eht.item().get('x'+inuc)[intc])		
        self.ddxi      = np.asarray(eht.item().get('ddx'+inuc)[intc])
        self.ddxiux    = np.asarray(eht.item().get('ddx'+inuc+'ux')[intc])
        self.ddxidot   = np.asarray(eht.item().get('ddx'+inuc+'dot')[intc])	
        self.ddxisq    = np.asarray(eht.item().get('ddx'+inuc+'sq')[intc])
        self.ddxisqux  = np.asarray(eht.item().get('ddx'+inuc+'squx')[intc])
        self.ddxixidot = np.asarray(eht.item().get('ddx'+inuc+'x'+inuc+'dot')[intc])
        self.xigradxpp = np.asarray(eht.item().get('x'+inuc+'gradxpp')[intc]) 	 		
        self.ddxidotux = np.asarray(eht.item().get('ddx'+inuc+'dotux')[intc]) 	
        self.ddxiuxux  = np.asarray(eht.item().get('ddx'+inuc+'uxux')[intc])		
        self.ddxiuyuy  = np.asarray(eht.item().get('ddx'+inuc+'uyuy')[intc])
        self.ddxiuzuz  = np.asarray(eht.item().get('ddx'+inuc+'uzuz')[intc])		
		
        self.dduyuy    = np.asarray(eht.item().get('dduyuy')[intc])
        self.dduzuz    = np.asarray(eht.item().get('dduzuz')[intc])
        self.dduxux    = np.asarray(eht.item().get('dduxux')[intc])
        self.dduxuy    = np.asarray(eht.item().get('dduxuy')[intc])
        self.dduxuz    = np.asarray(eht.item().get('dduxuz')[intc])		
		
        self.ppdivu    = np.asarray(eht.item().get('ppdivu')[intc])
        self.divu      = np.asarray(eht.item().get('divu')[intc])
        self.ppdivu    = np.asarray(eht.item().get('ppdivu')[intc])
        self.ddekux    = np.asarray(eht.item().get('ddekux')[intc])	
        self.ddek      = np.asarray(eht.item().get('ddek')[intc])
        self.ppux      = np.asarray(eht.item().get('ppux')[intc])	
				
		
        # store time series for time derivatives
        self.t_timec   = np.asarray(eht.item().get('timec')) 
        self.t_dd      = np.asarray(eht.item().get('dd')) 
        self.t_ddux    = np.asarray(eht.item().get('ddux')) 
        self.t_dduy    = np.asarray(eht.item().get('dduy'))
        self.t_dduz    = np.asarray(eht.item().get('dduz'))		
        self.t_ddxi    = np.asarray(eht.item().get('ddx'+inuc))		
        self.t_ddxisq  = np.asarray(eht.item().get('ddx'+inuc+'sq'))
        self.t_ddxiux  = np.asarray(eht.item().get('ddx'+inuc+'ux'))
		
        self.t_dduxux = np.asarray(eht.item().get('dduxux'))
        self.t_dduyuy = np.asarray(eht.item().get('dduyuy'))
        self.t_dduzuz = np.asarray(eht.item().get('dduzuz'))
		
        self.t_uxfuxf = self.t_dduxux/self.t_dd - self.t_ddux*self.t_ddux/(self.t_dd*self.t_dd)
        self.t_uyfuyf = self.t_dduyuy/self.t_dd - self.t_dduy*self.t_dduy/(self.t_dd*self.t_dd)
        self.t_uzfuzf = self.t_dduzuz/self.t_dd - self.t_dduz*self.t_dduz/(self.t_dd*self.t_dd)
		
        self.t_tke = 0.5*(self.t_uxfuxf+self.t_uyfuyf+self.t_uzfuzf)

        print('####################################')
        print('Plotting RANS for central time (in s): ',round(self.timec,1))
        print('####################################')	
        print('Averaging windows (in s): ',self.tavg)
        print('Time range (in s from-to): ',round(self.trange[0],1),round(self.trange[1],1))		
        print('####################################')
		
        #######################
        # Xi TRANSPORT EQUATION 
        #######################
		
        # LHS dq/dt 		
        self.dtddxi = self.dt(self.t_ddxi,self.xzn0,self.t_timec,intc)
        # LHS div(ddXiux)
        self.divddxiux = self.Div(self.ddxi*self.ddux/self.dd,self.xzn0)
		
        # RHS div(fxi) 
        self.divfxi = self.Div(self.ddxiux - self.ddxi*self.ddux/self.dd,self.xzn0) 
        # RHS ddXidot 
        self.ddXidot = self.ddxidot 
        # res
        self.resXiTransport = - self.dtddxi - self.divddxiux - self.divfxi + self.ddXidot
		
        ###########################		
        # END Xi TRANSPORT EQUATION
        ###########################
		
        ######################
        # Xi VARIANCE EQUATION 
        ######################		
 
        # LHS dq/dt 
        self.t_ddsigmai = self.t_ddxisq -self.t_ddxi*self.t_ddxi/self.t_dd
        self.dtddsigmai = self.dt(self.t_ddsigmai,self.xzn0,self.t_timec,intc)
        # LHS div(dduxsigmai)
        self.divdduxsigmai = self.Div((self.ddxisq -self.ddxi*self.ddxi/self.dd)*(self.ddux/self.dd),self.xzn0)

        # RHS div f^sigma
        self.divfxir = self.Div(self.ddxisqux - \
                       2.*self.ddxiux*self.ddxi/self.dd  - \
                       self.ddxisq*self.ddux/self.dd + \
                       2.*self.ddxi*self.ddxi*self.ddux/(self.dd*self.dd),self.xzn0)
        # RHS 2*f d_r Xi
#        self.fxigradi = 2.*self.dd*(self.ddxiux/self.dd - self.ddxi/(self.dd*self.dd))*self.Grad(self.ddxi/self.dd,self.xzn0)
        self.fxigradi = 2.*(self.ddxiux - self.ddxi*self.ddux/self.dd)*self.Grad(self.ddxi/self.dd,self.xzn0)


        # RHS 2*dd xif xdoti
        self.xifddxidot = 2.*(self.ddxixidot - (self.ddxi/self.dd)*self.ddxidot)	
	
        # res
        self.resXiVariance = - self.dtddsigmai - self.divdduxsigmai - \
                             self.divfxir - self.fxigradi + self.xifddxidot
	
        ##########################
        # END Xi VARIANCE EQUATION 		
        ##########################		
		
        ##################
        # Xi FLUX EQUATION 
        ##################		
 
        # LHS dq/dt 
        self.t_ddfluxi = self.t_ddxiux -self.t_ddxi*self.t_ddux/self.t_dd
        self.dtfluxi = self.dt(self.t_ddfluxi,self.xzn0,self.t_timec,intc)
        # LHS div(dduxfluxi)
        self.divuxfluxi = self.Div((self.ddxiux -self.ddxi*self.ddux/self.dd)*(self.ddux/self.dd),self.xzn0)		

        # RHS div(frxi) 
        self.divfrxi = self.Div(self.ddxiuxux - (self.ddxi/self.dd)*self.dduxux - \
                      2.*(self.ddux/self.dd)*self.ddxiux + \
                      2.*self.ddxi*self.ddux*self.ddux/(self.dd*self.dd),self.xzn0)

        # RHS fi d_r fu_r
        self.figradxfur = (self.ddxiux - self.ddxi*self.ddux/self.dd)*self.Grad(self.ddux/self.dd,self.xzn0)
		
        # RHS Rxx d_r Xi
        self.rxxgradxxi = (self.dduxux - self.ddux*self.ddux/self.dd)*self.Grad(self.ddxi/self.dd,self.xzn0)

        # RHS X''i d_r P - X''_i d_r P'
        self.xigradp = (self.xi*self.Grad(self.pp,self.xzn0) - (self.ddxi/self.dd)*self.Grad(self.pp,self.xzn0)) + \
                        (self.xigradxpp - self.xi*self.Grad(self.pp,self.xzn0)) 		

        # RHS uxfddXidot
        self.uxfddXidot = self.ddxidotux - (self.ddux/self.dd)*self.ddxidot  		
		
        # RHS geometry term
        self.gi = -(self.ddxiuyuy - (self.ddxi/self.dd)*self.dduyuy - 2.*(self.dduy/self.dd) + 2.*self.ddxi*self.dduy*self.dduy/(self.dd*self.dd))/self.xzn0 - \
                   (self.ddxiuzuz - (self.ddxi/self.dd)*self.dduzuz - 2.*(self.dduz/self.dd) + 2.*self.ddxi*self.dduz*self.dduz/(self.dd*self.dd))/self.xzn0 + \
                   (self.ddxiuyuy - (self.ddxi/self.dd)*self.dduyuy)/self.xzn0 + \
                   (self.ddxiuzuz - (self.ddxi/self.dd)*self.dduzuz)/self.xzn0
		
        self.resXiFlux = -self.dtfluxi - self.divuxfluxi - self.divfrxi - self.figradxfur - \
                          self.rxxgradxxi - self.xigradp + self.uxfddXidot + self.gi     
		
        ######################
        # END Xi FLUX EQUATION 
        ######################	
		
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
        self.ax = - self.ux + self.ddux/self.dd		
		
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
		
		
        ##############################		
        # GET SIZE OF CONVECTION ZONE 
        ##############################

        rc = self.xzn0		
        # load TKE dissipation
        self.diss = self.resTkeEquation

        # calculate INDICES for grid boundaries 
        if self.lgrid == 1:
            idxl, idxr = self.idx_bndry(self.xbl,self.xbr)
        else:
            idxl = 0
            idxr = self.nx-1			
		
        # Get rid of the numerical mess at inner boundary 
        self.diss[0:idxl] = 0.
        # Get rid of the numerical mess at outer boundary 
        self.diss[idxr:self.nx] = 0.
 
        self.diss_max = self.diss.max()
        self.ind = np.where( (self.diss > 0.02*self.diss_max) )[0]
		
        self.rinc  = rc[self.ind[0]]
        self.routc = rc[self.ind[-1]]
        self.lc = self.routc - self.rinc		
		
        #######################################		
        # Kolmogorov damping timescale
        #######################################		
		
        self.kolmog_damp_timescale = self.tke/(self.resTkeEquation/self.dd)
        self.variancediss = (self.ddxisq -self.ddxi*self.ddxi/self.dd)/self.kolmog_damp_timescale
		
        self.uconv = (2.*self.tke)**0.5
        self.kolmrate = self.uconv**3/self.lc
        self.tauL = self.tke/self.kolmrate
        #self.variancediss = (self.ddxisq -self.ddxi*self.ddxi/self.dd)/self.tauL

        ################
        # Xi DIFFUSIVITY 
        ################

        #self.k = (1./2.)*((self.dduxux - self.ddux*self.ddux/self.dd) + \
        #                  (self.dduyuy - self.dduy*self.dduy/self.dd) + \
        #                  (self.dduzuz - self.dduz*self.dduz/self.dd))/self.dd 
        self.uconv = (2.*self.tke)**0.5
 
        self.fi = self.ddxiux - self.ddxi*self.ddux/self.dd
        self.xi = self.ddxi/self.dd	
		
        self.Deff = -self.fi/(self.dd*self.Grad(self.xi,self.xzn0))
        self.Durms      = (1./3.)*self.uconv*self.lc

        alphae = 1.
        self.hp = - self.pp/self.Grad(self.pp,self.xzn0)
        self.u_mlt = (self.ddhhux - self.ddhh*self.ddux/self.dd)/(alphae*(self.ddcp/self.dd)*(self.ddttsq-self.ddtt*self.ddtt/self.dd)/self.dd)
		
        self.Dumlt1     = (1./3.)*self.u_mlt*self.lc

        alpha = 1.5
        self.Dumlt2 = (1./3.)*self.u_mlt*alpha*self.hp        

        alpha = 1.6
        self.Dumlt3 = (1./3.)*self.u_mlt*alpha*self.hp        

        self.lagr = (4.*np.pi*(self.xzn0**2.)*self.dd)**2.				

        ####################
        # END Xi DIFFUSIVITY 
        ####################		
		
		
    def plot_Xrho(self,data_prefix):
        """Plot Xrho stratification in the model""" 

        # convert nuc ID to string
        xnucid = str(self.inuc)
		
        # load x GRID
        grd1 = self.xzn0
	
        # load DATA to plot
        plt1 = self.ddxi
		
        # calculate INDICES for grid boundaries 
        if self.lgrid == 1:
            idxl, idxr = self.idx_bndry(self.xbl,self.xbr)
		
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		
		
        # limit x/y axis
        if self.lgrid == 1:
            plt.axis([self.xbl,self.xbr,np.min(plt1[idxl:idxr]),np.max(plt1[idxl:idxr])])
        else:
            plt.axis([grd1[0],grd1[-1],np.min(plt1[0:-1]),np.max(plt1[0:-1])])	
		
        # plot DATA 
        plt.title('rhoX xnucid '+str(xnucid))
        plt.plot(grd1,plt1,color='brown',label = r'$\overline{\rho} \widetilde{X}$')

        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"$\overline{\rho} \widetilde{X}$ (g cm$^{-3}$)"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=7,prop={'size':18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+data_prefix+'mean_rhoX_'+xnucid+'.png')
	
    def plot_Xtransport_equation(self,data_prefix):
        """Plot Xrho transport equation in the model""" 

        # convert nuc ID to string
        xnucid = str(self.inuc)
		
        # load x GRID
        grd1 = self.xzn0

        lhs0 = - self.dtddxi
        lhs1 = - self.divddxiux 
		
        rhs0 = - self.divfxi
        rhs1 = + self.ddXidot
		
        print(self.ddXidot)
		
        res = - self.resXiTransport
		
        # calculate INDICES for grid boundaries 
        if self.lgrid == 1:
            idxl, idxr = self.idx_bndry(self.xbl,self.xbr)
		
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		
		
        # limit x/y axis by global min/max from all terms
        if self.lgrid == 1:
            minx = np.min([np.min(lhs0[idxl:idxr]),np.min(lhs1[idxl:idxr]),np.min(rhs0[idxl:idxr]),np.min(rhs1[idxl:idxr]),np.min(res[idxl:idxr])])
            maxx = np.max([np.max(lhs0[idxl:idxr]),np.max(lhs1[idxl:idxr]),np.max(rhs0[idxl:idxr]),np.max(rhs1[idxl:idxr]),np.max(res[idxl:idxr])])
            plt.axis([self.xbl,self.xbr,minx,maxx])
        else:
            minx = np.min([np.min(lhs0[0:-1]),np.min(lhs1[0:-1]),np.min(rhs0[0:-1]),np.min(rhs1[0:-1]),np.min(res[0:-1])])
            maxx = np.max([np.max(lhs0[0:-1]),np.max(lhs1[0:-1]),np.max(rhs0[0:-1]),np.max(rhs1[0:-1]),np.max(res[0:-1])])			
            plt.axis([grd1[0],grd1[-1],minx,maxx])
		
        # plot DATA 
        plt.title('rhoX transport xnucid '+str(xnucid))
        plt.plot(grd1,lhs0,color='r',label = r'$-\partial_t (\overline{\rho} \widetilde{X})$')
        plt.plot(grd1,lhs1,color='cyan',label = r'$-\nabla_r (\overline{\rho} \widetilde{X} \widetilde{u}_r)$')		
        plt.plot(grd1,rhs0,color='b',label=r'$-\nabla_r f$')
        plt.plot(grd1,rhs1,color='g',label=r'$+\overline{\rho} \widetilde{\dot{X}}^{\rm nuc}$')
        plt.plot(grd1,res,color='k',linestyle='--',label='res')

        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"g cm$^{-3}$ s$^{-1}$"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=9,prop={'size':12})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+data_prefix+'mean_Xtransport_'+xnucid+'.png')

		
    def plot_Xflux(self,data_prefix):
        """Plot Xflux stratification in the model""" 

        # convert nuc ID to string
        xnucid = str(self.inuc)
		
        # load x GRID
        grd1 = self.xzn0		
		
        # load and calculate DATA to plot
        plt1 = self.ddxiux - self.ddxi*self.ddux/self.dd
		
        # calculate INDICES for grid boundaries 
        if self.lgrid == 1:
            idxl, idxr = self.idx_bndry(self.xbl,self.xbr)
		
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format Y AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))
		
        # limit x/y axis
        if self.lgrid == 1:
            plt.axis([self.xbl,self.xbr,np.min(plt1[idxl:idxr]),np.max(plt1[idxl:idxr])])
        else:
            plt.axis([grd1[0],grd1[-1],np.min(plt1[0:-1]),np.max(plt1[0:-1])])
			
        # plot DATA 
        plt.title('Xflux xnucid '+str(xnucid))
        plt.plot(grd1,plt1,color='k',label = r'f'+str(self.inuc))

        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"$\overline{\rho} \widetilde{X''_i u''_r}$ (g cm$^{-2}$ s$^{-1}$)"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=7,prop={'size':18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+data_prefix+'mean_Xflux_'+xnucid+'.png')
		
    def plot_Xflux_equation(self,data_prefix):
        """Plot Xi flux equation in the model""" 

        # convert nuc ID to string
        xnucid = str(self.inuc)
		
        # load x GRID
        grd1 = self.xzn0

        lhs0 = - self.dtfluxi
        lhs1 = - self.divuxfluxi
		
        rhs0 = - self.divfrxi
        rhs1 = - self.figradxfur
        rhs2 = - self.rxxgradxxi
        rhs3 = - self.xigradp
        rhs4 = + self.uxfddXidot 
        rhs5 = + self.gi
		
        res =  -self.resXiFlux
		
        # calculate INDICES for grid boundaries 
        if self.lgrid == 1:
            idxl, idxr = self.idx_bndry(self.xbl,self.xbr)
		
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		
		
        # limit x/y axis by global min/max from all terms
        if self.lgrid == 1:
            minx = np.min([np.min(lhs0[idxl:idxr]),np.min(lhs1[idxl:idxr]),\
	                       np.min(rhs0[idxl:idxr]),np.min(rhs1[idxl:idxr]),\
	                       np.min(rhs2[idxl:idxr]),np.min(rhs3[idxl:idxr]),\
	                       np.min(rhs4[idxl:idxr]),np.min(rhs5[idxl:idxr]),\
	                       np.min(res[idxl:idxr])])
            maxx = np.max([np.max(lhs0[idxl:idxr]),np.max(lhs1[idxl:idxr]),\
	                       np.max(rhs0[idxl:idxr]),np.max(rhs1[idxl:idxr]),\
	                       np.max(rhs2[idxl:idxr]),np.max(rhs3[idxl:idxr]),\
	                       np.max(rhs4[idxl:idxr]),np.max(rhs5[idxl:idxr]),\
	                       np.max(res[idxl:idxr])])
            plt.axis([self.xbl,self.xbr,minx,maxx])
        else:
            minx = np.min([np.min(lhs0[0:-1]),np.min(lhs1[0:-1]),\
	                       np.min(rhs0[0:-1]),np.min(rhs1[0:-1]),\
	                       np.min(rhs2[0:-1]),np.min(rhs3[0:-1]),\
	                       np.min(rhs4[0:-1]),np.min(rhs5[0:-1]),\
	                       np.min(res[0:-1])])
            maxx = np.max([np.max(lhs0[0:-1]),np.max(lhs1[0:-1]),\
	                       np.max(rhs0[0:-1]),np.max(rhs1[0:-1]),\
	                       np.max(rhs2[0:-1]),np.max(rhs3[0:-1]),\
	                       np.max(rhs4[0:-1]),np.max(rhs5[0:-1]),\
	                       np.max(res[0:-1])])
            plt.axis([grd1[0],grd1[-1],minx,maxx])		
		
        # plot DATA 
        plt.title('Xflux equation xnucid '+str(xnucid))
        plt.plot(grd1,lhs0,color='#8B3626',label = r'$-\partial_t f_i$')
        plt.plot(grd1,lhs1,color='#FF7256',label = r'$-\nabla_r (\widetilde{u}_r f)$')		
        plt.plot(grd1,rhs0,color='b',label=r'$-\nabla_r f^r_i$')
        plt.plot(grd1,rhs1,color='g',label=r'$-f \partial_r \widetilde{u}_r$')
        plt.plot(grd1,rhs2,color='r',label=r'$-R_{rr} \partial_r \widetilde{X}$')	
        plt.plot(grd1,rhs3,color='cyan',label=r'$-\overline{X^{,,}} \partial_r \overline{P} - \overline{X^{,,} \partial_r P^{,}}$')
        plt.plot(grd1,rhs4,color='purple',label=r'$+\overline{u^{,,}_r \rho \dot{X}}$')
        plt.plot(grd1,rhs5,color='yellow',label=r'$+G$')		
        plt.plot(grd1,res,color='k',linestyle='--',label='res')

        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"g cm$^{-2}$ s$^{-2}$"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=8,prop={'size':8})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+data_prefix+'mean_Xflux_'+xnucid+'.png')		
		

    def plot_Xvariance(self,data_prefix):
        """Plot Xvariance stratification in the model""" 

        # convert nuc ID to string
        xnucid = str(self.inuc)
		
        # load x GRID
        grd1 = self.xzn0
		
        # load and calculate DATA to plot
        plt1 = (self.ddxisq - self.ddxi*self.ddxi/self.dd)/self.dd
		
        # calculate INDICES for grid boundaries
        if self.lgrid == 1:		
            idxl, idxr = self.idx_bndry(self.xbl,self.xbr)
		
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format Y AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))
		
        # limit x/y axis
        if self.lgrid == 1:
            plt.axis([self.xbl,self.xbr,1.e-35,np.max(plt1[idxl:idxr])])
        else:
            plt.axis([grd1[0],grd1[-1],1.e-35,np.max(plt1[0:-1])])
		
        # plot DATA 
        plt.title('Xvariance xnucid '+str(xnucid))
        plt.semilogy(grd1,plt1,color='b',label = r'$\sigma$'+str(self.inuc))

        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"$\widetilde{X''_i X''_i}$"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=7,prop={'size':18})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+data_prefix+'mean_Xvariance_'+xnucid+'.png')
		
	
    def plot_Xvariance_equation(self,data_prefix):
        """Plot Xi variance equation in the model""" 

        # convert nuc ID to string
        xnucid = str(self.inuc)
		
        # load x GRID
        grd1 = self.xzn0

        lhs0 = - self.dtddsigmai
        lhs1 = - self.divdduxsigmai
		
        rhs0 = - self.fxigradi
        rhs1 = - self.divfxir
        rhs2 = + self.xifddxidot
		
        res = - self.resXiVariance
		
        rhs3 = - self.variancediss 
		
        # calculate INDICES for grid boundaries 
        if self.lgrid == 1:
            idxl, idxr = self.idx_bndry(self.xbl,self.xbr)
		
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		
		
        # limit x/y axis by global min/max from all terms
        if self.lgrid == 1:
            minx = np.min([np.min(lhs0[idxl:idxr]),np.min(lhs1[idxl:idxr]),np.min(rhs0[idxl:idxr]),np.min(rhs1[idxl:idxr]),np.min(rhs2[idxl:idxr]),np.min(rhs3[idxl:idxr]),np.min(res[idxl:idxr])])
            maxx = np.max([np.max(lhs0[idxl:idxr]),np.max(lhs1[idxl:idxr]),np.max(rhs0[idxl:idxr]),np.max(rhs1[idxl:idxr]),np.max(rhs2[idxl:idxr]),np.max(rhs3[idxl:idxr]),np.max(res[idxl:idxr])])
            plt.axis([self.xbl,self.xbr,minx,maxx])
        else:
            minx = np.min([np.min(lhs0[0:-1]),np.min(lhs1[0:-1]),np.min(rhs0[0:-1]),np.min(rhs1[0:-1]),np.min(rhs2[0:-1]),np.min(rhs3[0:-1]),np.min(res[0:-1])])
            maxx = np.max([np.max(lhs0[0:-1]),np.max(lhs1[0:-1]),np.max(rhs0[0:-1]),np.max(rhs1[0:-1]),np.min(rhs2[0:-1]),np.min(rhs3[0:-1]),np.max(res[0:-1])])
            plt.axis([grd1[0],grd1[-1],minx,maxx])		
		
		
        # plot DATA 
        plt.title('Xvariance equation xnucid '+str(xnucid))
        plt.plot(grd1,lhs0,color='cyan',label = r'$-\partial_t (\overline{\rho} \sigma)$')
        plt.plot(grd1,lhs1,color='purple',label = r'$-\nabla_r (\overline{\rho} \widetilde{u}_r \sigma)$')		
        plt.plot(grd1,rhs1,color='b',label=r'$-\nabla_r f^\sigma$')
        plt.plot(grd1,rhs0,color='g',label=r'$-2 f \partial_r \widetilde{X}$')
        plt.plot(grd1,rhs2,color='r',label=r'$+2 \overline{\rho X^{,,} \dot{X}}$')		
        plt.plot(grd1,rhs3,color='k',label=r'$+ \overline{\rho} \sigma / \tau_L$')		
        plt.plot(grd1,res,color='k',linestyle='--',label='res')

        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"g cm$^{-3}$ s$^{-1}$"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=9,prop={'size':10})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+data_prefix+'mean_Xvariance_'+xnucid+'.png')		
		
		
    def plot_X_Ediffusivity(self,data_prefix):
    # Eulerian diffusivity
        # convert nuc ID to string
        xnucid = str(self.inuc)
		
        # load x GRID
        grd1 = self.xzn0
		
        term0 = self.Deff
        term1 = self.Durms
        term2 = self.Dumlt1
        term3 = self.Dumlt2
        term4 = self.Dumlt3		
		
        # calculate INDICES for grid boundaries
        if self.lgrid == 1:		
            idxl, idxr = self.idx_bndry(self.xbl,self.xbr)
		
        # create FIGURE
        plt.figure(figsize=(7,6))
		
        # format AXIS, make sure it is exponential
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0,0))		
		
        # limit x/y axis by global min/max from all terms
        if self.lgrid == 1:
            minx = np.min([np.min(term0[idxl:idxr]),np.min(term1[idxl:idxr]),np.min(term2[idxl:idxr]),np.min(term3[idxl:idxr]),np.min(term4[idxl:idxr])])
            maxx = np.max([np.max(term0[idxl:idxr]),np.max(term1[idxl:idxr]),np.max(term2[idxl:idxr]),np.max(term3[idxl:idxr]),np.min(term4[idxl:idxr])])
            plt.axis([self.xbl,self.xbr,minx,maxx])
        else:
            minx = np.min([np.min(term0[0:-1]),np.min(term1[0:-1]),np.min(term2[0:-1]),np.min(term3[0:-1]),np.min(term4[0:-1])])
            maxx = np.max([np.max(term0[0:-1]),np.max(term1[0:-1]),np.max(term2[0:-1]),np.max(term3[0:-1]),np.min(term4[0:-1])])
            plt.axis([grd1[0],grd1[-1],minx,maxx])		

        # plot DATA 		
        plt.title(r'Eulerian Diff xnucid '+str(xnucid))
        plt.plot(grd1,term0,label=r"$\sigma_{eff} = - f_i/(\overline{\rho} \ \partial_r \widetilde{X}_i)$")
        plt.plot(grd1,term1,label=r"$\sigma_{urms} = (1/3) \ u_{rms} \ l_c $")
        plt.plot(grd1,term2,label=r"$\sigma_{umlt} = + u_{mlt} \ l_c $")        
        plt.plot(grd1,term3,label=r"$\sigma_{umlt} = + u_{mlt} \ \alpha_{mlt} \ H_P \ (\alpha_{mlt}$ = 1.5)")
        plt.plot(grd1,term4,label=r"$\sigma_{umlt} = +(u_{mlt} \ \alpha_{mlt} \ H_P \ (\alpha_{mlt}$ = 1.6)") 

        # define and show x/y LABELS
        setxlabel = r"r (cm)"
        setylabel = r"cm$^{-2}$ s$^{-1}$"
        plt.xlabel(setxlabel)
        plt.ylabel(setylabel)
		
        # show LEGEND
        plt.legend(loc=8,prop={'size':12})

        # display PLOT
        plt.show(block=False)

        # save PLOT
        plt.savefig('RESULTS/'+data_prefix+'Ediff_'+xnucid+'.png')			
			
				
    def gauss(x, *p): 
    # Define model function to be used to fit to the data above:
        A, mu, sigma = p
        return A*np.exp(-(x-mu)**2/(2.*sigma**2))		
	
    def idx_bndry(self,xbl,xbr):
    # calculate indices of grid boundaries 
        rr = np.asarray(self.xzn0)
        xlm = np.abs(rr-xbl)
        xrm = np.abs(rr-xbr)
        idxl = int(np.where(xlm==xlm.min())[0][0])
        idxr = int(np.where(xrm==xrm.min())[0][0])	
        return idxl,idxr
	
	
