import numpy as np

class PROMPI_rans:

    def __init__(self,filename):

        fhead = open(filename.replace("ransdat","ranshead"),'r') 

        header_line1 = fhead.readline().split()
        header_line2 = fhead.readline().split()
        header_line3 = fhead.readline().split()
        header_line4 = fhead.readline().split()

        qqx    = int(header_line2[0])
        qqy    = int(header_line2[1])
        qqz    = int(header_line2[2])
        nnuc   = int(header_line2[3])
        nrans  = int(header_line2[4])
        ndims = [4,nrans,qqx]

        self.ransl = []		
        for line in range(nrans):
            line = fhead.readline().strip()
            self.ransl.append(line)  

        xznl = [] 
        xzn0 = []
        xznr = []
        yznl = []
        yzn0 = []
        yznr = [] 
        zznl = []
        zzn0 = []
        zznr = []
			
        for line in range(qqx):
            line = fhead.readline().strip()        
            xznl.append(float(line[8:22].strip()))
            xzn0.append(float(line[23:38].strip()))
            xznr.append(float(line[39:54].strip()))
			
        for line in range(qqy):
            line = fhead.readline().strip()        
            yznl.append(float(line[8:22].strip()))
            yzn0.append(float(line[23:38].strip()))
            yznr.append(float(line[39:54].strip()))	

        for line in range(qqz):
            line = fhead.readline().strip()
            zznl.append(float(line[8:22].strip()))
            zzn0.append(float(line[23:38].strip()))
            zznr.append(float(line[39:54].strip()))	

        frans = open(filename,'rb')
        self.data = np.fromfile(frans)		
#        self.data = np.fromfile(frans,dtype='>f',count=ndims[0]*ndims[1]*ndims[2])
        self.data = np.reshape(self.data,(ndims[0],ndims[1],ndims[2]),order='F')	

        self.eh = {}
		
        eh_xzn0 = {"xzn0" : xzn0}
        self.eh.update(eh_xzn0)
		
        i = 0
        print(self.ransl)
		
        for s in self.ransl:
            eh_field = 'eh_'+str(s)
            eh_entry = {eh_field : self.data[2,i,:]}
            self.eh.update(eh_entry)
            i += 1
		
        frans.close()
        fhead.close()		

    def rans(self):	
        return self.eh
	
		
    def ranslist(self):
        return np.asarray(self.xzn0)

    def ransdict(self):
        print self.eh.keys()
		
    def sterad(self):
        pass
		
    def grad(self):
        pass
		
    def div(self):
        pass
		
     
class PROMPI_blck:

    def __init__(self,filename):
        pass
