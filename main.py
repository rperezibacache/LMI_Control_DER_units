import numpy as np
import sys as s
s.path.append("/Users/ricardo/Dropbox/RP_PYTHON_PACKAGE/")
import rp_pck as rp
import rp_performance as rp_per
import rp_ctr as rp_ctr
from scipy.linalg import solve_discrete_are,expm,eigvals,expm,svd,pinv,eig,solve_continuous_are,sqrtm, solve_sylvester,solve_discrete_lyapunov
import cvxpy as cp
#import mosek

#----- Continuous System ------#
class Model_Sys():
	#{{{
	# MODEL FOR DER1, CIGRE PSCAD
	Sb = 2.e6
	wb = 2.*np.pi*60.
	#Vb = 520.
	Vb = 636.*np.sqrt(2./3.)
	# Trafo Tr
	# These parameters can be changed
	#Rf = 1.e-3*(0.85)
	#Lf = 20.e-6
	#Cf = 1.125e-3
	#Cf = 2.812e-3
	#
	#Rg = 1.887e-3
	#Lg = 13.661e-6
	#dtd = 1./5000.
	# New design for full controller
	#Rf = 1.e-3*(0.755)
	Rf = 1.e-3*(1.62+5.)
	Lf = 43.e-6
	#Cf = 1.125e-3
	Cf = 1.308e-3
	#Rf = 1.e-3*(0.85+5.)
	#Lf = 17.51e-6
	#Cf = 1.125e-3
	#
	VT = 636./np.sqrt(3.)
	PT = Sb
	IT = 2.*Sb/(3.*VT)
	ZT = VT/(IT)
	Rg = ZT*0.02  #0.943e-3
	Lg = 0.0346*ZT/wb  #10.e-6
	#print 'Rg = ', Rg
	#print 'Lg = ', Lg
	print 'ib = ', IT
	dtd = 1./5000.
	# State S4ace
	A0 = np.matrix([[-(Rf/Lf),wb,-1./Lf,0.,0.,0.,0.],
			[-wb,-Rf/Lf,0.,-1./Lf,0.,0.,0.],
			[1./Cf,0.,0.,wb,-1./Cf,0.,0.],
			[0.,1./Cf,-wb,0.,0.,-1./Cf,0.],
			[0.,0.,1./Lg,0.,-Rg/Lg,wb,0.],
			[0.,0.,0.,1./Lg,-wb,-Rg/Lg, -Vb/Lg],
			[0.,0.,0.,0.,0.,0.,0.]])
	B0 = np.matrix([[1./Lf,0.,0.],[0.,1./Lf,0.],[0.,0.,0.],[0.,0.,-0.*Vb],[0.,0.,0.],[0.,0.,0.],[0.,0.,-1]])
	C0 = np.asmatrix(np.append(np.eye(6),np.zeros((6,1)),axis=1))
	D0w = np.append(np.zeros((6,3)),np.append(np.zeros((6,3)),np.eye(6),axis=1),axis=1)
	#
	B0w = np.matrix([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.],[0.,0.,0.],[-1./Lg,0.,0.],[0.,-1./Lg,0.],[0.,0.,1.]])
	# Sys continuous
	SS_cnt = (A0,B0,C0,D0w,B0w)
	# Sys Discrete
	Ad,Bd,Cd,Ddw,Bdw = rp.C2D(A0,B0,C0,D0w,B0w,dtd,'ZOH')
	Dzw = np.zeros((6,3))
	SS_dis = (Ad,Bd,Cd,Ddw,Bdw,Dzw)
	Bdw = np.append(Bdw,Bd, axis=1)
	Bdw = np.append(Bdw,np.zeros((7,6)), axis=1)
	# chosen output z = [vsd,vsq,wt]
	Cz = np.matrix([[0.,0.,1.,0.,0.,0.,0.],[0.,0.,0.,1.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.]],dtype=float)#np.append(Cd,np.zeros((Bd.shape[1],Ad.shape[0])),axis=0)
	Dz = np.matrix([[0.,0.,0.],[0.,0.,0.],[0.,0.,1.]],dtype=float)#np.append(np.zeros((Cd.shape[0],Bd.shape[1])),np.eye(Bd.shape[1]),axis=0)
	#aux = np.append(np.zeros((6,3)),np.append(np.zeros((6,3)),np.eye(6),axis=1),axis=1)
	Dzw = np.matrix([[0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]],dtype=float)#np.append(aux, np.zeros((Bd.shape[1],aux.shape[1])),axis=0)
	SS_ext_dis = (Ad,Bd,Cd,Ddw,Bdw,Cz,Dz,Dzw)
	#
	#
	Ib = 2.*Sb/(3.*Vb)
 	#print Ib
	phi = np.linspace(0., 2.*np.pi, num=10)
 	Io_nom = np.zeros((2,phi.shape[0]))
	#delta_nom = np.zeros(phi.shape[0])
	#x_nom = np.asmatrix(np.zeros((3,phi.shape[0])) )
	y_nom = np.asmatrix(np.zeros((6,phi.shape[0])) )
	x_nom = np.asmatrix(np.zeros((7,phi.shape[0])) )
	Io_nom[0,:] = 1.0*(Ib)*np.cos(phi)
	Io_nom[1,:] = 1.0*(Ib)*np.sin(phi)
	#print Io_nom
	for i in xrange(len(phi)):
		vg = Vb
		io_ = Io_nom[0,i] + 1J*Io_nom[1,i]
		delta = (Lg/(vg))*(-wb*io_.real-(Rg/Lg)*io_.imag)
		vs_ = (Rg + 1J*wb*Lg)*io_ + vg#vg*1J*delta + vg
		if_ = 1J*wb*Cf*vs_ + io_
		#delta = (1./(1J*vg))*(vs_-vg-Rg*io_-1J*wb*Lg*io_)
		#
		y_nom[:,i] = np.matrix([[if_.real, if_.imag,vs_.real,vs_.imag,io_.real,io_.imag]],dtype=float).T
		x_nom[:,i] = np.matrix([[if_.real, if_.imag,vs_.real,vs_.imag,io_.real,io_.imag,delta.real]],dtype=float).T
	#}}}

def Pole_Placement_Discrete(M,Dsgn):
	#{{{
	# Asumming a quadratic region on the complex plane
	Pi1T_P_Pi1 = M[0]
	Pi1T_PA_Pi1 = M[1]
	#
	R11 = Dsgn[0]
	R12 = Dsgn[1]
	R22 = Dsgn[2]
	L = sqrtm(R22)
	#
	aux11 = cp.kron(R11,Pi1T_P_Pi1) + cp.kron(R12,Pi1T_PA_Pi1) + cp.kron(R12.T,Pi1T_PA_Pi1.T) 
	aux12 = cp.kron(L,Pi1T_PA_Pi1.T)
	aux22 = cp.kron(-np.eye(L.shape[0]), Pi1T_P_Pi1)
	LMI = cp.bmat([[aux11,aux12],[aux12.T,aux22]])
	return LMI
	#}}}


# ========================================== #
# ====== LINEAR QUADRATIC GAUSSIAN ========= #
def LQG_dsgn():	
	#{{{	
	print 'SOLVING LQG CONTROL ....' 
	sys = Model_Sys()
	# state-space
	ss = (sys.Ad,sys.Bd,sys.Cd)
	# observer design
	Q_o = sys.Bdw*sys.Bdw.T
	#print Q_o.shape
	gamma = 3.e6
	R_o = gamma*np.eye(sys.Cd.shape[0])
	obs = (sys.Ad,sys.Cd,Q_o,R_o)
	# control design
	Q_x = sys.Cd.T*sys.Cd
	#print Q_x.shape
	R_u = 1.e0*np.matrix([[1.e0,0.,0.],[0.,1.e0,0.],[0.,0.,2.e2]])
	ctr = (sys.Ad,sys.Bd,sys.Cd,sys.Bd,sys.Cd,Q_x,R_u)
	# calling the solver
	(L_obs,K_ctr,aux,S_ctr), LQG_ctr = rp_ctr.LQG_LTR(ss,obs,ctr)
	#print K_ctr

	ss_ext = sys.SS_ext_dis
	ss_red = sys.SS_dis
	np.savez('data_dsg_LQGLTR', ss_ctr = LQG_ctr, ss_ext =ss_ext,dtd=sys.dtd, Inom = sys.Ib, ss_red = ss_red)
	#------#-----#-------#
	s.stdout.write("\033[F") #back to previous line
	s.stdout.write("\033[K") #clear line
	print 'SOLVING LQG CONTROL .... DONE!'
	# ------ #-------#------#
	#}}}
# ========================================== #


# ========================================== #
# ====== LINEAR MATRIX INEQUALITY ========= #
def LMI_dsgn():
	print 'SOLVING LMI CONTROL ....'
	sys = Model_Sys()
	dtd = sys.dtd
	def LMI_H2(sys):
		#{{{
		Bj=sys[0]
		Cj=sys[1]
		Dj=sys[2]
		Ej=sys[3]
		Fj=sys[4]
		Q = sys[5]
		aux11 = -cp.bmat([[X, np.eye(nx)],[np.eye(nx), Y]]) 
		aux12 = cp.bmat([[Bj + B*D_h*Fj],[Y*Bj + B_h*Fj]]) 
		aux13 = cp.bmat([[A*X + B*C_h, A+B*D_h*C],[A_h, Y*A+B_h*C]]) 
		aux22 = -np.eye(aux12.shape[1])
		aux23 = np.zeros((aux22.shape[0],aux13.shape[1]))
		aux33 = aux11
		aux = cp.bmat([[aux11,aux12,aux13],[aux12.T,aux22,aux23],[aux13.T,aux23.T,aux33]])
		LMI1 = aux << 0
		# LMI 2- H performance associated transformation
		aux11 = Q
		aux12 = cp.bmat([[Cj*X+Ej*C_h, Cj+Ej*D_h*C]])
		aux22 = cp.bmat([[X, np.eye(nx)],[np.eye(nx), Y]])
		aux = cp.bmat([[aux11,aux12],[aux12.T,aux22]])
		LMI2 = aux >> 0
		# LMI 3- D closed loop minimization, 
		aux = Ej*D_h*Fj + Dj
		LMI3 = aux==0
		return (LMI1,LMI2,LMI3)	
		#}}}
	
	def LMI_Gep_H2(sys):# Applies for energy to peak, or the generalized H2 performance
		#{{{
		Bj=sys[0]
		Cj=sys[1]
		Dj=sys[2]
		Ej=sys[3]
		Fj=sys[4]
		gamma = sys[5]
		#
		Pi1T_P_Pi1 = cp.bmat([[X, np.eye(nx)],[np.eye(nx), Y]])
		Pi1T_PA_Pi1 = cp.bmat([[A*X + B*C_h, A+B*D_h*C],[A_h, Y*A+B_h*C]]) 
		Cj_Pi1 = cp.bmat([[Cj*X+Ej*C_h, Cj+Ej*D_h*C]])
		PiT_P_Bj = cp.bmat([[Bj+B*D_h*Fj],[ Y*Bj+B_h*Fj]])
		#
		aux11 = Pi1T_P_Pi1
		aux13 = Pi1T_PA_Pi1.T
		aux23 = PiT_P_Bj.T
		aux22 = np.eye(aux23.shape[0])
		aux12 = np.zeros((aux11.shape[0], aux22.shape[1]))
		aux33 = Pi1T_P_Pi1
		aux = cp.bmat([[aux11,aux12,aux13],[aux12.T,aux22,aux23],[aux13.T,aux23.T,aux33]])
		LMI1 = aux >> 0
		# LMI 2- H performance associated transformation
		aux11 = Pi1T_P_Pi1
		aux13 = Cj_Pi1.T
		aux23 = (Dj + Ej*D_h*Fj).T
		aux22 = np.zeros((aux23.shape[0],aux23.T.shape[1]))
		aux33 = gamma*np.eye(aux23.shape[1])
		aux12 = np.zeros((aux11.shape[0],aux22.shape[1]))
		
		aux = cp.bmat([[aux11,aux12,aux13],[aux12.T,aux22,aux23],[aux13.T,aux23.T,aux33]])
		LMI2 = aux >> 0
		#
		LMI3 = Dj+Ej*D_h*Fj == 0
		return (LMI1,LMI2,LMI3)	
		#}}}
	def LMI_H2_Static(sys):
		#{{{
		Bj=sys[0]
		Cj=sys[1]
		Dj=sys[2]
		Ej=sys[3]
		Fj=sys[4]
		gamma = sys[5]
		#
		Pi1T_P_Pi1 = cp.bmat([[X, np.eye(nx)],[np.eye(nx), Y]])
		Pi1T_PA_Pi1 = cp.bmat([[A*X + B*C_h, A+B*D_h*C],[A_h, Y*A+B_h*C]]) 
		Cj_Pi1 = cp.bmat([[Cj*X+Ej*C_h, Cj+Ej*D_h*C]])
		PiT_P_Bj = cp.bmat([[Bj+B*D_h*Fj],[ Y*Bj+B_h*Fj]])
		#
		# LMI 2- H performance associated transformation
		aux11 = Pi1T_P_Pi1
		aux13 = Cj_Pi1.T
		aux23 = (Dj + Ej*D_h*Fj).T
		aux22 = np.zeros((aux23.shape[0],aux23.T.shape[1]))
		aux33 = gamma*np.eye(aux23.shape[1])
		aux12 = np.zeros((aux11.shape[0],aux22.shape[1]))
		
		aux = cp.bmat([[aux11,aux12,aux13],[aux12.T,aux22,aux23],[aux13.T,aux23.T,aux33]])
		LMI1 = aux >> 0
		#
		return LMI1	
		#}}}

	def decay_time(M,Dsgn):
	#{{{
		# Asumming a quadratic region on the complex plane
		Pi1T_P_Pi1 = M[0]
		Pi1T_PA_Pi1 = M[1]
		#
		R11 = Dsgn[0]
		R12 = Dsgn[1]
		R22 = Dsgn[2]
		L = sqrtm(R22)
		#
		aux11 = R11[0,0]*Pi1T_P_Pi1 
		aux12 = Pi1T_PA_Pi1.T
		aux22 = -Pi1T_P_Pi1
		LMI = cp.bmat([[aux11,aux12],[aux12.T,aux22]])
		return LMI
	#}}}

	#--- Constraints ----
	consts = []
 	A = sys.Ad
	B = sys.Bd
	C = sys.Cd
 	Dw = sys.Ddw
	Bw = sys.Bdw
	Cz = sys.Cz
	Dzw = sys.Dzw
	Dz = sys.Dz
	#
	nx = A.shape[0]
	nu = B.shape[1]
	ny = C.shape[0]
	# LMI variables
	X = cp.Variable((nx,nx),symmetric=True)
	Y = cp.Variable((nx,nx),symmetric=True)
	A_h = cp.Variable((nx,nx) )
	B_h = cp.Variable((nx,ny) )
	C_h = cp.Variable((nu,nx) )
	D_h = cp.Variable((nu,ny) )
	#===
	# LMIs for H2
	#{{{
	#Dzw = np.zeros((Cz.shape[0],B.shape[1]))
	# Problems variables definition 
	#
	Rj = np.matrix(np.eye(12),dtype=float)#1.e0*np.matrix(np.append(np.zeros((3,9)),np.eye(9),axis=0 ),dtype=float)
	Lj = np.matrix(np.eye(3),dtype=float)#1.*np.matrix([[0.,0.,1.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,1.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,1.]],dtype=float)
	#
	Bj = Bw*Rj
	Cj = Lj*Cz
	Dj =Lj*Dzw*Rj
	Ej = Lj*Dz
	Fj = Dw*Rj
	q = cp.Variable(Cj.shape[0])
	#print Cj.shape[0]
	Q_var = cp.diag( q)
	sys_LMI = (Bj,Cj,Dj,Ej,Fj,Q_var)
	(Cons1, Cons2 ,Cons3) = LMI_H2(sys_LMI)
	#
	# LMI 4- Trace  minimization, 
	LMI4 = cp.trace(Q_var)
	#}}}
	consts += [Cons1,Cons2]



	# LMIs for pole constraints
	#{{{
	Pi1T_P_Pi1 = cp.bmat([[X, np.eye(nx)],[np.eye(nx), Y]])
	Pi1T_PA_Pi1 = cp.bmat([[A*X + B*C_h, A+B*D_h*C],[A_h, Y*A+B_h*C]]) 
	Matrix_LMI = (Pi1T_P_Pi1, Pi1T_PA_Pi1)
	# for speed of the poles
	alfa = 30. # s+alfa  (poles) 
	#nd = 2*nx # dimension of the space where the poles will be
	R11 = np.matrix([[-np.exp(-2*alfa*dtd), 0.],[0.,-1.]])#-np.eye(nd)*np.exp(-2.*alfa*dtd)
	R12 = np.matrix([[0.,1.],[0.,0.]])#np.zeros((nd,nd))
	R22 = np.matrix([[0.,0.],[0.,0.]])#np.eye(nd)
	M_DSGN = (R11,R12,R22)
	#LMI5 = Pole_Placement_Discrete(Matrix_LMI, M_DSGN)
	LMI5 = decay_time(Matrix_LMI, M_DSGN)
	Cons5 = LMI5 << 0
	# damping constraints 
	xi_min = 0.6
	thet_ang = np.arccos(xi_min)
	case = 5 # approximation by: '0' for inner circle, '1' for inner ellipse, '2' half right-plane and ellipse, '3' ellipse cone, '4' ellipse+cone+right half-plane, '5' RPP+speed
	if case == 0:
		# {{{for a circle approximation
		x_m = np.exp(-thet_ang/np.tan(thet_ang))*np.cos(-thet_ang)
		y_m = -np.exp(-thet_ang/np.tan(thet_ang))*np.sin(-thet_ang)
		x_0 = -np.exp(-np.pi/np.tan(thet_ang))
		ak = x_m-x_0
		bk = y_m
		r_k = min(ak,bk)
		# LMI
		R11 = np.matrix([[x_m**2 - r_k**2]])
		R12 = -np.matrix([[x_m]])
		R22 = np.matrix([[1.]])
		M_DSGN = (R11,R12,R22)
		LMI6 = Pole_Placement_Discrete(Matrix_LMI, M_DSGN)
		#------- Draw constraints -------#
		Damp = (xi_min, dtd)
		Circ_Approx = (x_m,r_k)
		Constraints_Draw(Damp,Circ_Approx,'Circle')
		#}}}
		#
	if case == 1:
		#{{{ for a ellipse approximation
		x_m = np.exp(-thet_ang/np.tan(thet_ang))*np.cos(-thet_ang)
		y_m = -np.exp(-thet_ang/np.tan(thet_ang))*np.sin(-thet_ang)
		x_0 = -np.exp(-np.pi/np.tan(thet_ang))
		ak = x_m-x_0
		bk = y_m
		# LMI
		R11 = np.matrix([[-1.,-x_m/ak],[-x_m/ak,-1.]])
		R12 = np.matrix([[0.,(1./ak)*0.5 - (1./bk)*0.5],[(1./ak)*0.5 + (1./bk)*0.5,0.]])
		R22 = np.matrix([[0.,0.],[0.,0.]])
		M_DSGN = (R11,R12,R22)
		LMI6 = Pole_Placement_Discrete(Matrix_LMI, M_DSGN)
		#------- Draw constraints -------#
		Damp = (xi_min, dtd)
		Ellip_Approx = (ak,bk,x_m)
		Constraints_Draw(Damp,Ellip_Approx,'Ellipse')
		#}}}
		#
	if case == 2:
		#{{{ for a half plane and ellipse approximation
		x_m = np.exp(-thet_ang/np.tan(thet_ang))*np.cos(-thet_ang)
		y_m = -np.exp(-thet_ang/np.tan(thet_ang))*np.sin(-thet_ang)
		x_0 = -np.exp(-np.pi/np.tan(thet_ang))
		y_3 = np.exp(-np.pi/(2.*np.tan(thet_ang)) ) 
		ak = x_m*y_m/np.sqrt(y_m**2-y_3**2)
		bk = y_m
		# LMI
		R11 = np.matrix([[0.,0.,0.],[0.,-1.,-x_m/ak],[0.,-x_m/ak,-1.]])
		R12 = np.matrix([[-1.,0.,0.],[0.,0.,(1./ak)*0.5 - (1./bk)*0.5],[0.,(1./ak)*0.5 + (1./bk)*0.5,0.]])
		R22 = np.matrix([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]])
		M_DSGN = (R11,R12,R22)
		LMI6 = Pole_Placement_Discrete(Matrix_LMI, M_DSGN)
		#------- Draw constraints -------#
		Damp = (xi_min, dtd)
		Ellip_Approx = (ak,bk,x_m,y_3,x_0)
		Constraints_Draw(Damp,Ellip_Approx,'HP_Ellipse')
		#}}}
		#
	if case == 3:
		#{{{ for a ellipse-cone approximation
		x_m = np.exp(-thet_ang/np.tan(thet_ang))*np.cos(-thet_ang)
		y_m = -np.exp(-thet_ang/np.tan(thet_ang))*np.sin(-thet_ang)
		x_0 = -np.exp(-np.pi/np.tan(thet_ang))
		y_3 = np.exp(-np.pi/(2.*np.tan(thet_ang)) ) 
		a_e = x_m*y_m/np.sqrt(y_m**2-y_3**2)
		b_e = y_m
		#------ ellipse centered inner cordiode
		xe = 0.5 # interseccion cardiode and cone
		ye = b_e*np.sin( np.arccos( (xe-x_m)/a_e ) ) # interseccion con cardioide
		xse = (1.+x_0)/2. # new center of the ellipse
		ak = (1.-x_0)/2. # new 'a' 
		bk = ye*ak/np.sqrt(ak**2 - (xe-xse)**2) # new 'b'
		# matices
		R11e = np.matrix([[-1.,-xse/ak],[-xse/ak,-1.]])
		R12e = np.matrix([[0.,(1./ak)*0.5 - (1./bk)*0.5],[(1./ak)*0.5 + (1./bk)*0.5,0.]])
		R22e = np.matrix([[0.,0.],[0.,0.]])
		#-------- cone with vetex at (xv,0)
		gama = np.arctan(ye/(1.-xe))
		xv = 1.
		R11v = np.matrix([[-xv*np.sin(gama)*2, 0.],[0., -xv*np.sin(gama)*2]])
		R12v = np.matrix([[np.sin(gama), np.cos(gama)],[-np.cos(gama), np.sin(gama)]])
		R22v = np.matrix([[0.,0.],[0.,0.]])
		#--------- Intersection ellipse and cone
		Z = np.zeros((2,2))
		aux1 = np.append(R11e,Z,axis=1)
		aux2 = np.append(Z, R11v,axis=1)
		R11 = np.append(aux1,aux2, axis=0)
		aux1 = np.append(R12e,Z,axis=1)
		aux2 = np.append(Z, R12v,axis=1)
		R12 = np.append(aux1,aux2, axis=0)
		aux1 = np.append(R22e,Z,axis=1)
		aux2 = np.append(Z, R22v,axis=1)
		R22 = np.append(aux1,aux2, axis=0)
		#-------------------------------
		M_DSGN = (R11,R12,R22)
		LMI6 = Pole_Placement_Discrete(Matrix_LMI, M_DSGN)
		#------- Draw constraints -------#
		Damp = (xi_min, dtd)
		Ellip_Approx = (ak,bk,xse,xe, gama,xv )
		Constraints_Draw(Damp,Ellip_Approx,'Ellipse_Cone')
		#}}}
		#
	if case == 4:
		#{{{ for a ellipse-cone approximation and right half plane
		x_m = np.exp(-thet_ang/np.tan(thet_ang))*np.cos(-thet_ang)
		y_m = -np.exp(-thet_ang/np.tan(thet_ang))*np.sin(-thet_ang)
		x_0 = -np.exp(-np.pi/np.tan(thet_ang))
		y_3 = np.exp(-np.pi/(2.*np.tan(thet_ang)) ) 
		a_e = x_m*y_m/np.sqrt(y_m**2-y_3**2)
		b_e = y_m
		#
		k = np.tan(thet_ang)
		t = np.arange(-np.pi/k,0, 0.01)
		#------ ellipse centered inner cordiode
		xe = 0.6 # interseccion cardiode and cone
		for i in xrange(t.shape[0]):
			error = xe-np.exp(t[i])*np.cos(k*t[i])
			if error < 0.01:
				t_aux = t[i]
				break
		ye = abs(np.exp(t_aux)*np.sin(k*t_aux))
		#print ye
		xse = (1.+x_0)/2. # new center of the ellipse
		ak = (1.-x_0)/2. # new 'a' 
		bk = ye*ak/np.sqrt(ak**2 - (xe-xse)**2) # new 'b'
		# matices
		R11e = np.matrix([[-1.,-xse/ak],[-xse/ak,-1.]])
		R12e = np.matrix([[0.,(1./ak)*0.5 - (1./bk)*0.5],[(1./ak)*0.5 + (1./bk)*0.5,0.]])
		R22e = np.matrix([[0.,0.],[0.,0.]])
		#-------- cone with vetex at (xv,0)
		gama = np.arctan(ye/(1.-xe))
		xv = 1.
		R11v = np.matrix([[-xv*np.sin(gama)*2, 0.],[0., -xv*np.sin(gama)*2]])
		R12v = np.matrix([[np.sin(gama), np.cos(gama)],[-np.cos(gama), np.sin(gama)]])
		R22v = np.matrix([[0.,0.],[0.,0.]])
		#---------- right half plane
		R11h = np.matrix([[0.]])
		R12h = np.matrix([[-1.]])
		R22h = np.matrix([[0.]])
		#--------- Intersection ellipse and cone
		Z = np.zeros((2,2))
		Z2 = np.zeros((2,1))
		aux_a = np.append(R11e,Z,axis=1)
		aux1 = np.append(aux_a,Z2,axis=1)
		aux_b = np.append(Z, R11v,axis=1)
		aux2 = np.append(aux_b, Z2,axis=1)
		aux_c = np.append(Z2.T, Z2.T,axis=1)
		aux3 = np.append(aux_c, R11h,axis=1)
		aux_R = np.append(aux1,aux2, axis=0)
		R11 = np.append(aux_R,aux3, axis=0)
		#
		aux_a = np.append(R12e,Z,axis=1)
		aux1 = np.append(aux_a,Z2,axis=1)
		aux_b = np.append(Z, R12v,axis=1)
		aux2 = np.append(aux_b, Z2,axis=1)
		aux_c = np.append(Z2.T, Z2.T,axis=1)
		aux3 = np.append(aux_c, R12h,axis=1)
		aux_R = np.append(aux1,aux2, axis=0)
		R12 = np.append(aux_R,aux3, axis=0)
		#
		aux_a = np.append(R22e,Z,axis=1)
		aux1 = np.append(aux_a,Z2,axis=1)
		aux_b = np.append(Z, R22v,axis=1)
		aux2 = np.append(aux_b, Z2,axis=1)
		aux_c = np.append(Z2.T, Z2.T,axis=1)
		aux3 = np.append(aux_c, R22h,axis=1)
		aux_R = np.append(aux1,aux2, axis=0)
		R22 = np.append(aux_R,aux3, axis=0)
		#-------------------------------
		M_DSGN = (R11,R12,R22)
		LMI6 = Pole_Placement_Discrete(Matrix_LMI, M_DSGN)
		#------- Draw constraints -------#
		Damp = (xi_min, dtd)
		Ellip_Approx = (ak,bk,xse,xe, gama,xv )
		Constraints_Draw(Damp,Ellip_Approx,'Ellipse_Cone_RHP')
		#}}}
		#
	if case == 5:
		#{{{ right half plane
		#---------- right half plane
		Alfa_L = 0.
		R11h = np.matrix([[0.]])
		R12h = np.matrix([[-1.]])
		R22h = np.matrix([[0.]])
		#--------- Intersection ellipse and cone
		R11 = R11h
		R12 = R12h
		R22 = R22h
		#-------------------------------
		M_DSGN = (R11,R12,R22)
		LMI6 = Pole_Placement_Discrete(Matrix_LMI, M_DSGN)
		#}}}
	Cons6 = LMI6 << 0
	#}}}

	consts += [Cons5]

	# static H2, for voltage and frequency regulation 


	# LMI for H_inf performance 
	#{{{
	def Hinf_LMI(aux1,aux2):
		#{{{
		Lj = aux1[0]
		Rj = aux1[1]
		gamma = aux1[2]
		#
		Cz = aux2[0]
		Dz = aux2[1]
		Bw = aux2[2]
		Dw = aux2[3]
		Dzw = aux2[4]
		#
		Cj = Lj*Cz
		Ej = Lj*Dz
		Bj = Bw*Rj
		Fj = Dw*Rj
		Dj = Lj*Dzw*Rj
		#
		Pi1T_P_Pi1 = cp.bmat([[X, np.eye(nx)],[np.eye(nx), Y]])
		Pi1T_PA_Pi1 = cp.bmat([[A*X + B*C_h, A+B*D_h*C],[A_h, Y*A+B_h*C]]) 
		Cj_Pi1 = cp.bmat([[Cj*X+Ej*C_h, Cj+Ej*D_h*C]])
		PiT_P_Bj = cp.bmat([[Bj+B*D_h*Fj],[ Y*Bj+B_h*Fj]])
		#
		aux11 = -Pi1T_P_Pi1
		aux13 = Pi1T_PA_Pi1.T
		aux14 = Cj_Pi1.T
		aux23 = PiT_P_Bj.T
		aux24 = (Dj+Ej*D_h*Fj).T
		aux33 = aux11
		aux44 = -np.eye(aux24.T.shape[0])
		aux12 = np.zeros((aux11.shape[0],aux23.shape[0] ))
		aux22 = -np.eye(aux23.shape[0])*gamma
		aux34 = np.zeros((aux33.shape[0],aux24.shape[1]))
		#
		LMI7 = cp.bmat([[aux11,aux12,aux13,aux14],[aux12.T,aux22,aux23,aux24],[aux13.T,aux23.T,aux33, aux34],[aux14.T,aux24.T,aux34.T,aux44]])
		Cons_a = LMI7 << 0
		Cons_b = Dj+Ej*D_h*Fj == 0
		#return Cons_a, Cons_b
		return Cons_a
		#}}}
	
	def Hinf_LMI_Static(aux1,aux2):
		#{{{
		Lj = aux1[0]
		Rj = aux1[1]
		gamma = aux1[2]
		#
		Cz = aux2[0]
		Dz = aux2[1]
		Bw = aux2[2]
		Dw = aux2[3]
		Dzw = aux2[4]
		#
		Cj = Lj*Cz
		Ej = Lj*Dz
		Bj = Bw*Rj
		Fj = Dw*Rj
		Dj = Lj*Dzw*Rj
		#
		Cj_Pi1 = cp.bmat([[Cj*X+Ej*C_h, Cj+Ej*D_h*C]])
		#
		aux13 = Cj_Pi1.T
		aux23 = (Dj+Ej*D_h*Fj).T
		aux11 = np.zeros((aux13.shape[0],aux13.T.shape[1]))	
		aux22 = -np.eye(aux23.shape[0])
		aux12 = np.zeros((aux13.shape[0],aux22.shape[1]))
		aux33 = -gamma*np.eye(aux23.shape[1])
		#
		LMI7 = cp.bmat([[aux11,aux12,aux13],[aux12.T,aux22,aux23],[aux13.T,aux23.T,aux33]])
		Cons_a = LMI7 << 0
		return Cons_a
		#}}}

	#====================================
	# LMI for H_inf Static for wt / io
	gamma = (1.e0)**2#0.0001/5500.
	Lj = np.matrix([[0.,0.,1.]])
	Rj = np.matrix([[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.]]).T
	aux1 = (Lj,Rj,gamma)
	aux2 = (Cz,Dz,Bw,Dw,Dzw)
	# ------------------------
	#Cons_a= Hinf_LMI_Static(aux1,aux2)
	#consts += [Cons_a]
	#====================================
	# LMI for H_inf performance  for wt / if
	gamma = (1.e-1)**2#0.0001/5500.
	Lj = np.matrix([[0.,0.,1.]])
	#Lj = np.matrix([[1.,0.,0.],[0.,1.,0.]])
	Rj = np.matrix([[0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.]]).T
	aux1 = (Lj,Rj,gamma)
	aux2 = (Cz,Dz,Bw,Dw,Dzw)
	# ------------------------
	#Cons_a= Hinf_LMI(aux1,aux2)
	#consts += [Cons_a]
	#====================================
	# LMI for H_inf performance  for wt / if
	gamma = (1.e-4)**2#0.0001/5500.
	#Lj = np.matrix([[1.,0.,0.],[0.,1.,0.]])
	Lj = np.matrix([[0.,0.,1.]])
	Rj = np.matrix([[0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.]]).T
	aux1 = (Lj,Rj,gamma)
	aux2 = (Cz,Dz,Bw,Dw,Dzw)
	# ------------------------
	#Cons_a= Hinf_LMI(aux1,aux2)
	#consts += [Cons_a]
	
	#====================================
	# LMI for H_inf performance  for wt / [if,vs]
	gamma = (1.e-2/sys.IT)**2#0.0001/5500.
	Lj = np.matrix([[0.,0.,1.]])
	#Lj = np.matrix([[1.,0.,0.],[0.,1.,0.]])
	Rj = np.matrix([[0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.],
					[0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.],
					#[0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.],
					#[0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.],
 					[0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.],
					[0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.],	 ]).T
	aux1 = (Lj,Rj,gamma)
	aux2 = (Cz,Dz,Bw,Dw,Dzw)
	# ------------------------
	Cons_a= Hinf_LMI(aux1,aux2)
	consts += [Cons_a]
	#====================================
	# LMI for H_inf performance  for wt / [vs]
	gamma = (1.e-3/sys.IT)**2#0.0001/5500.
	Lj = np.matrix([[0.,0.,1.]])
	#Lj = np.matrix([[1.,0.,0.],[0.,1.,0.]])
	Rj = np.matrix([[0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.],
					[0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.],	 ]).T
	aux1 = (Lj,Rj,gamma)
	aux2 = (Cz,Dz,Bw,Dw,Dzw)
	# ------------------------
	#Cons_a= Hinf_LMI(aux1,aux2)
	#consts += [Cons_a]
	
	#====================================
	# LMI for H_inf performance  for wt / [io]
	gamma = (2.*np.pi/sys.IT)**2#0.0001/5500.
	#print np.sqrt(gamma)
	Lj = np.matrix([[0.,0.,1.]])
	#Lj = np.matrix([[1.,0.,0.],[0.,1.,0.]])
	Rj = np.matrix([[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.],
					[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.],	 ]).T
	aux1 = (Lj,Rj,gamma)
	aux2 = (Cz,Dz,Bw,Dw,Dzw)
	# ------------------------
	Cons_a= Hinf_LMI(aux1,aux2)
	consts += [Cons_a]

	#====================================
	# LMI for H_inf performance  for vs / [io], infeasible
	#gamma = (520.*0.4/sys.IT)**2#0.0001/5500.
	gamma = (1.e2)**2#0.0001/5500.
	#Lj = np.matrix([[0.,0.,1.]])
	#Lj = np.matrix([[1.,0.,0.],[0.,1.,0.]])
	Lj = np.matrix([[0.,1.,0.]])
	#Rj = np.matrix([[1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
	#				[0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],	 ]).T
	Rj = np.matrix([[1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]]).T
	aux1 = (Lj,Rj,gamma)
	aux2 = (Cz,Dz,Bw,Dw,Dzw)
	# ------------------------
	#Cons_a= Hinf_LMI(aux1,aux2)
	#consts += [Cons_a]

	#====================================
	# LMI for H_inf performance  for vs / [if], infeasible
	gamma = (1.e0)**2#0.0001/5500.
	#Lj = np.matrix([[0.,0.,1.]])
	Lj = np.matrix([[1.,0.,0.],[0.,1.,0.]])
	Rj = np.matrix([[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.],
					[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.],	 ]).T
	aux1 = (Lj,Rj,gamma)
	aux2 = (Cz,Dz,Bw,Dw,Dzw)
	# ------------------------
	#Cons_a= Hinf_LMI(aux1,aux2)
	#consts += [Cons_a]


	# ======== * ====ENSURE ROBUSTNESS vs/vg  
	gamma = (520*0.1)**2# the energy of the signal, 520*Porc*t_on, Por: porcentual amplitude of the disturbance, t_on = 0.2
	Lj = np.matrix([[1.,0.,0.],[0.,1.,0.]])
	Rj = np.matrix([[1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]]).T
	#Rj = np.matrix([[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.]]).T
	aux1 = (Lj,Rj,gamma)
	aux2 = (Cz,Dz,Bw,Dw,Dzw)
	#=====================================
	#Cons_a = Hinf_LMI(aux1,aux2)
	#consts += [Cons_a]
	#====================================
	# ==== *=== ENSURE ROBUSTNESS vs/wg  
	gamma = (5.*np.pi)**2
	#Lj = np.matrix([[1.,0.,0.],[0.,1.,0.]])
	#Lj = np.matrix([[0.,0.,0.,0.,1.],[0.,0.,0.,1.,0.]])
	Lj = np.matrix([[1.,0.,0.],[0.,1.,0.]])
	#Lj = np.matrix([[1.,0.,0.]])
	Rj = np.matrix([[0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.]]).T
	#Rj = np.matrix([[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.]]).T
	aux1 = (Lj,Rj,gamma)
	aux2 = (Cz,Dz,Bw,Dw,Dzw)
	#=====================================
	#Cons_a = Hinf_LMI(aux1,aux2)
	#consts += [Cons_a]
	#====================================
	#====================================
	# ==== *=== ENSURE ROBUSTNESS ws/vg  
	gamma = (520.*0.1)**2
	#Lj = np.matrix([[1.,0.,0.],[0.,1.,0.]])
	#Lj = np.matrix([[0.,0.,0.,0.,1.],[0.,0.,0.,1.,0.]])
	Lj = np.matrix([[0.,0.,1.]])
	Rj = np.matrix([[1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]]).T
	#Rj = np.matrix([[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.]]).T
	aux1 = (Lj,Rj,gamma)
	aux2 = (Cz,Dz,Bw,Dw,Dzw)
	#=====================================
	Cons_a = Hinf_LMI(aux1,aux2)
	consts += [Cons_a]
	#====================================
	# ==== *=== ENSURE ROBUSTNESS ws/wg  
	gamma = (2.*np.pi)**2
	#Lj = np.matrix([[1.,0.,0.],[0.,1.,0.]])
	#Lj = np.matrix([[0.,0.,0.,0.,1.],[0.,0.,0.,1.,0.]])
	Lj = np.matrix([[0.,0.,1.]])
	Rj = np.matrix([[0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.]]).T
	#Rj = np.matrix([[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.]]).T
	aux1 = (Lj,Rj,gamma)
	aux2 = (Cz,Dz,Bw,Dw,Dzw)
	#=====================================
	Cons_a = Hinf_LMI(aux1,aux2)
	consts += [Cons_a]
	#====================================
	#}}}

	#--------------------------------------------#
	#------- SOLVING THE OPTMIZATION PROBLEM-----#
	#{{{za
	optprob = cp.Problem(cp.Minimize(LMI4), constraints=consts)
	#optprob = cp.Problem(cp.Minimize(0.), constraints=consts)
	print("prob is DCP:", optprob.is_dcp())
	result = optprob.solve(solver=cp.CVXOPT,verbose=True, max_iters=10000, abstol=5.e-6, reltol=5.e-6,feastol=5.e-6,refinement = 30,kktsolver='robust')
	#result = optprob.solve(solver=cp.GUROBI)#,verbose=True, max_iters=10000, abstol=1.e-6, reltol=1.e-6,feastol=1.e-6,refinement = 50,kktsolver='robust')
	#result = optprob.solve(solver=cp.MOSEK,verbose=True,warm_start=True, mosek_params = {mosek.dparam.optimizer_max_time:  1000.0,
                                    #mosek.iparam.intpnt_solve_form:   mosek.solveform.primal, mosek.dparam.basis_rel_tol_s: 1e-12, mosek.dparam.basis_tol_x: 1e-8 , mosek.dparam.ana_sol_infeas_tol : 1.e-10 })
	print '#------ LMI-SOLUTION-CVXPY(solver) ------#'
	print 'Dinamic output feedback'
	X_LMI = np.matrix(X.value,dtype=float)
	Y_LMI = np.matrix(Y.value,dtype=float)
	Q_LMI = np.matrix(Q_var.value,dtype=float)
	Bh_LMI = np.matrix(B_h.value,dtype=float)
	Ah_LMI = np.matrix(A_h.value,dtype=float)
	Dh_LMI = np.matrix(D_h.value,dtype=float)
	Ch_LMI = np.matrix(C_h.value,dtype=float)
	#----------- Solving dynamic output feedback
	MNT = np.asmatrix(np.eye(nx)) - X_LMI*Y_LMI
	u,s,vT = svd(MNT)
	M = np.asmatrix(u,dtype=float)*sqrtm(np.asmatrix(np.diag(s),dtype=float) )
	N = np.asmatrix(vT,dtype=float).T*sqrtm(np.asmatrix(np.diag(s),dtype=float) )
	#-----------------
	Dk = Dh_LMI
	Ck = (Ch_LMI - Dk*C*X_LMI)*(M.I).T
	Bk = N.I*(Bh_LMI - Y_LMI*B*Dk)
	Ak = N.I*(Ah_LMI - N*Bk*C*X_LMI - Y_LMI*B*Ck*M.T -Y_LMI*A*X_LMI - Y_LMI*B*Dk*C*X_LMI)*(M.I).T
	#}}}
	LMI_ctr = (Ak,Bk,Ck,Dk)
	
	ss_ext = sys.SS_ext_dis
	ss_red = sys.SS_dis
	np.savez('data_dsg_LMI', ss_ctr = LMI_ctr, ss_ext =ss_ext,dtd=sys.dtd, Inom = sys.Ib, ss_red = ss_red)
	#------#-----#-------#
	#s.stdout.write("\033[F") #back to previous line
	#s.stdout.write("\033[K") #clear line
	print 'SOLVING LMI CONTROL .... DONE!'
	

	print 'tr(Q*) = ', Q_LMI.trace()
	print 'gamma_2 = ', 2.*np.pi/sys.IT
	print 'gamma_3 = ', 1.e-2/sys.IT
	print 'gamma_4 = ', 520*0.1
	print 'gamma_5 = ', 2.*np.pi


# ========================================== #
#-----+----- calling LQG -----+ --------#
#LQG_dsgn()# execute the design and save it!.

#-----+----- calling LMI -----+ --------#
LMI_dsgn()# execute the design and save it!.

# ========================================================= #
s.exit('# ======= Im DONE bonitinho!!. ======== #') #====== #
# ========================================================= #


b = rp_per.Perf_Dsct_time()
A = sys.Ad
B = sys.Bd
C = sys.Cd
Dw = sys.Ddw
Bw = sys.Bdw
Cz = sys.Cz
Dzw = sys.Dzw
Dz = sys.Dz
ss = (A,B,C,Dw,Bw,Cz,Dz,Dzw)
# ==== norm infty (2) ==== # 
Lj = np.matrix([[0.,0.,1.]])
Rj = np.matrix([[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.]]).T
ss_out = (Rj,Lj)
(Acl,Bcl,Ccl,Dcl) = rp_ctr.SS_closed_loop(ss,LMI_ctr,ss_out)
Lya_ee, Norm_ee = b.T_ee(Acl,Bcl,Ccl,Dcl)
print 'gamma_2opt = ', Norm_ee
print 'gamma_2 = ', 2.*np.pi/sys.IT
# ==== norm infty (3) ==== # 
Lj = np.matrix([[0.,0.,1.]])
Rj = np.matrix([[0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.],
				[0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.],
				#[0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.],
				#[0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.],
				[0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.],
				[0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.],	 ]).T
ss_out = (Rj,Lj)
(Acl,Bcl,Ccl,Dcl) = rp_ctr.SS_closed_loop(ss,LMI_ctr,ss_out)
Lya_ee, Norm_ee = b.T_ee(Acl,Bcl,Ccl,Dcl)
print 'gamma_2opt = ', Norm_ee
print 'gamma_2 = ', 1.e-2/sys.IT

# -------- CLOSING THE LOOP ------------ #
print 'SOLVING LQG NORMS....'
ss = sys.SS_ext_dis
# output chose z = [y,u]
Cz = np.append(sys.Cd,np.zeros((sys.Bd.shape[1],sys.Ad.shape[0])),axis=0)
Dz = np.append(np.zeros((sys.Cd.shape[0],sys.Bd.shape[1])),np.eye(sys.Bd.shape[1]),axis=0)
aux = np.append(np.zeros((6,3)),np.append(np.zeros((6,3)),np.eye(6),axis=1),axis=1)
Dzw = np.append(aux, np.zeros((sys.Bd.shape[1],aux.shape[1])),axis=0)
#--chosing the analysis
# w_t vs v_iod, w =Rj*wj, zj=Lj*z
Rj = np.matrix([[0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.]],dtype=float).T
Lj = np.matrix([[0.,0.,1.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,1.,0.,0.,0.,0.,0.]],dtype=float)
ss_out = (Cz,Dz,Dzw,Rj,Lj)

(Acl,Bcl,Ccl,Dcl) = rp_ctr.SS_closed_loop(ss,LQG_ctr,ss_out)

# ========================================== #
# ========================================== #
#
#

# ============ SYSTEM PERFORMANCE ANALYSIS DISCRETE TIME =========== #
b = rp_per.Perf_Dsct_time()
Lya_ie, Norm_ie = b.T_ie(Acl,Bcl,Ccl,Dcl)
Lya_ep, Norm_ep = b.T_ep(Acl,Bcl,Ccl,Dcl)
#Lya_ee, Norm_ee = b.T_ee(Acl,Bcl,Ccl,Dcl)
#------#-----#-------#
s.stdout.write("\033[F") #back to previous line
s.stdout.write("\033[K") #clear line
print 'SOLVING LQG NORMS.... DONE!'
# =================================================================== #
print Norm_ie
print Norm_ep
#print Norm_ee


#a = Perf_Cnt_time()
#Lya_ie, Norm_ie = a.T_ie(A,B,C,0)
#Lya_ep, Norm_ep = a.T_ep(A,B,C,0)
#Lya_ee, Norm_ee = a.T_ee(A,B,C,D)

#b = Perf_Dsct_time()
#Lya_ie, Norm_ie = b.T_ie(Ad,Bd,Cd,Dd)
#Lya_ep, Norm_ep = b.T_ep(Ad,Bd,Cd,Dd)
#Lya_ee, Norm_ee = b.T_ee(Ad,Bd,Cd,Dd)
#print 'Y = ', Lya_ie
#print 'Gam_ie = ', Norm_ie
#print 'X = ', Lya_ep
#print 'Gam_ep = ', Norm_ep
#print 'P = ', Lya_ee
#print 'Gam_ee = ', Norm_ee

# ===== UNIFIED SKELTON ANALYSIS DISCRETE-TIME ====== #
# - from QMI to LMI, from (QMI) (Om+Gam*G*Lam)*R*().T << Q to (LMI) Gam_d*G*Lam_d +().T << Om_d 
def QMI_to_LMI(Om,Gam,G,Lam,R,Q):
	Gam_d = cp.bmat([[Gam],[ np.zeros((Om.T.shape[0],Gam.shape[1]))]])
	Lam_d = cp.bmat([[np.zeros((Lam.shape[0],Q.shape[1])) , Lam]])
	Om_d = cp.bmat([[-Q, Om],[Om.T,-R]] )
	LMI = (Om + Gam*G*Lam)*R*(Om + Gam*G*Lam).T - Q << 0
	#LMI = cp.quad_form(Om + Gam*G*Lam,R) - Q << 0
	LMI = (Gam_d*G*Lam_d) + (Gam_d*G*Lam_d).T + Om_d << 0
	return LMI



# ------ Solving Output Dynamic Feedback ----#
#---- H2
#Ac,Bc,Cc,Dc, = Simp_Disc_H2_Out_FB1(sys_ctr)
#---- H2 and pole constraints
#Ac,Bc,Cc,Dc = Simp_Disc_H2_Out_FB2(sys_ctr,sys.dtd)
#---- H2, pole constraints  and H_inf
#Ac,Bc,Cc,Dc = Simp_Disc_H2_Out_FB3(sys_ctr,sys.dtd)
#---- H2, pole constraints, H_inf
sys_ctr =(sys.Ad,sys.Bd,sys.Cd,sys.Bdw,sys.Ddw)
#Ac,Bc,Cc,Dc,z_nom = Simp_Disc_H2_Out_FB5(sys_ctr,sys.dtd,sys.y_nom,sys.x_nom)
# Controller
#Ac = np.asmatrix(Ac)
#Bc = np.asmatrix(Bc)
#Cc = np.asmatrix(Cc)
#Dc = np.asmatrix(Dc)

# --- Discrete System ----#
#(As_d, Bs_d, Cs_d, Ds_d, Ps_d, Dw_d) = sys.SS_dis

#np.savez('data_dsg_LMI', Ac = Ac , Bc=Bc, Cc=Cc, Dc=Dc, As = As_d, Bs = Bs_d, Cs = Cs_d, Ps = Ps_d, Dw = Dw_d,Ds = Ds_d,dtd=sys.dtd, z_nom=z_nom)


# Consider a system [A,B,C,D]
# model for cnt test
xi = 0.1
ws = 1.
k = 1.
A = np.matrix([[-2.*xi*ws,-ws**2],[1.,0.]])
B = np.matrix([[1.],[0.]])
C = np.matrix([[0.,k]])
D = 0
# model discrte time test
Ac = np.matrix([[-1.,0.],[1.,-2.]])
Bc = np.matrix([[1.,0.],[0.,2.]])
#print expm(Ac*0.5)
Ad = expm(Ac*0.5)###np.matrix([[0.9048,0.],[0.0861,0.8187]])
Bd = Ac.I*(Ad-np.eye(2))*Bc#np.matrix([[0.0952,0.],[0.0045,0.1813]])
Cd = np.asmatrix(np.eye(2))
Dd = np.asmatrix(np.zeros((2,2)) )

def Constraints_Draw(Damp,data,method):
	#{{{
	xi = Damp[0]
	dtd = Damp[1]
	# cardiode
	thet = np.arccos(xi)
	k = np.tan(thet)
	t = np.arange(-np.pi/k,0, 0.01)
	z1 = np.exp(t)*(np.cos(k*t) + 1J*np.sin(k*t))
	z2 = np.exp(t)*(np.cos(k*t) - 1J*np.sin(k*t))
	# circle
	if method=='Circle':
		x_0 = data[0]
		r_c = data[1]
		t = np.arange(0.,2.*np.pi, 0.01)
		cx = x_0 + r_c*np.cos(t)
		cy =  r_c*np.sin(t)
	if method=='Ellipse':
		a = data[0]
		b = data[1]
		x_m = data[2]
		t = np.arange(0.,2.*np.pi, 0.01)
		cx = x_m + a*np.cos(t)
		cy =  b*np.sin(t)
	if method=='HP_Ellipse':
		a = data[0]
		b = data[1]
		x_m = data[2]
		y_3 = data[3]
		x_0 = data[4]
		alfa = np.arctan(float(y_3/(x_m)) )
		t = np.arange(-np.pi, np.pi, 0.001)
		cx = np.zeros(t.shape[0])
		cy = np.zeros(t.shape[0])
		for i in xrange(t.shape[0]):
			#if (t[i] > -np.pi+alfa) and (t[i] < np.pi-alfa):
			if x_m + a*np.cos(t[i])>0:
				cx[i] = x_m + a*np.cos(t[i])
				cy[i] =  b*np.sin(t[i])
	if method=='Ellipse_Cone':
		a = data[0]
		b = data[1]
		x_m = data[2]
		xe = data[3] # valor x in the intesction of areas
		gamma = data[4] # valor gamma angulo interseccion
		xv = data[5] # valor gamma angulo interseccion
		t = np.arange(-np.pi, np.pi, 0.001)
		cx = np.zeros(t.shape[0])
		cy = np.zeros(t.shape[0])
		for i in xrange(t.shape[0]):
			#if (t[i] > -np.pi+alfa) and (t[i] < np.pi-alfa):
			if x_m + a*np.cos(t[i]) <= xe:
				cx[i] = x_m + a*np.cos(t[i])
				cy[i] =  b*np.sin(t[i])
			if (x_m + a*np.cos(t[i]) > xe) and (t[i]<0):
				cx[i] = x_m + a*np.cos(t[i])
				cy[i] = (xv-cx[i])*np.tan(-gamma)
			if (x_m + a*np.cos(t[i]) > xe) and (t[i]>0):
				cx[i] = x_m + a*np.cos(t[i])
				cy[i] = (xv-cx[i])*np.tan(gamma)
	if method=='Ellipse_Cone_RHP':
		a = data[0]
		b = data[1]
		x_m = data[2]
		xe = data[3] # valor x in the intesction of areas
		gamma = data[4] # valor gamma angulo interseccion
		xv = data[5] # valor gamma angulo interseccion
		t = np.arange(-np.pi, np.pi, 0.001)
		cx = np.zeros(t.shape[0])
		cy = np.zeros(t.shape[0])
		for i in xrange(t.shape[0]):
			if x_m + a*np.cos(t[i]) >= 0:
				if x_m + a*np.cos(t[i]) <= xe:
					cx[i] = x_m + a*np.cos(t[i])
					cy[i] =  b*np.sin(t[i])
				if (x_m + a*np.cos(t[i]) > xe) and (t[i]<0):
					cx[i] = x_m + a*np.cos(t[i])
					cy[i] = (xv-cx[i])*np.tan(-gamma)
				if (x_m + a*np.cos(t[i]) > xe) and (t[i]>0):
					cx[i] = x_m + a*np.cos(t[i])
					cy[i] = (xv-cx[i])*np.tan(gamma)
	#
	np.savez('data_Const', z1 = z1, z2=z2, cx=cx, cy=cy)
	#}}}

def Pole_Placement_Discrete(M,Dsgn):
	#{{{
	# Asumming a quadratic region on the complex plane
	Pi1T_P_Pi1 = M[0]
	Pi1T_PA_Pi1 = M[1]
	#
	R11 = Dsgn[0]
	R12 = Dsgn[1]
	R22 = Dsgn[2]
	L = sqrtm(R22)
	#
	aux11 = cp.kron(R11,Pi1T_P_Pi1) + cp.kron(R12,Pi1T_PA_Pi1) + cp.kron(R12.T,Pi1T_PA_Pi1.T) 
	aux12 = cp.kron(L,Pi1T_PA_Pi1.T)
	aux22 = cp.kron(-np.eye(L.shape[0]), Pi1T_P_Pi1)
	LMI = cp.bmat([[aux11,aux12],[aux12.T,aux22]])
	return LMI
	#}}}

#------ LMI-H2 output feedback
def Simp_Disc_H2_Out_FB1(Sys):
	#{{{
 	A = Sys[0]
	B = Sys[1]
	C = Sys[2]
 	P = Sys[3]
	Q = C.T*C # xT*Q*x
	R = np.matrix([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
	#
	R_sqrt = 1.e0*sqrtm(R)
	Q_sqrt = sqrtm(Q)
	nx = A.shape[0]
	nu = B.shape[1]
	ny = C.shape[0]
	#
	Cz = np.append(Q_sqrt,np.zeros((nu,nx)),axis=0)
	Dz = np.append(np.zeros((nx,nu)),R_sqrt,axis=0)
	Bw = P
	Dw = np.asmatrix(np.zeros((ny,nu)) )
	Dzw = np.zeros((Cz.shape[0],B.shape[1]))
	# Problems variables definition
        X = cp.Variable((nx,nx),symmetric=True)
	Y = cp.Variable((nx,nx),symmetric=True)
	A_h = cp.Variable((nx,nx) )
	B_h = cp.Variable((nx,ny) )
	C_h = cp.Variable((nu,nx) )
	D_h = cp.Variable((nu,ny) )
	q = cp.Variable(Cz.shape[0])
	Q_var = cp.diag( q)
	nuq = cp.Variable(1)
	# LMI 1- Stability of the closed loop system, lyapunov H2
	aux11 = -cp.bmat([[X.T, np.eye(nx)],[np.eye(nx), Y.T]]) 
	aux12 = cp.bmat([[Bw + B*D_h*Dw],[Y*Bw + B_h*Dw]]) 
	aux13 = cp.bmat([[A*X + B*C_h, A+B*D_h*C],[A_h, Y*A+B_h*C]]) 
	aux22 = -np.eye(aux12.shape[1])
	aux23 = np.zeros((aux22.shape[0],aux13.shape[1]))
	aux33 = aux11
	LMI1 = cp.bmat([[aux11,aux12,aux13],[aux12.T,aux22,aux23],[aux13.T,aux23.T,aux33]])
	# LMI 2- H performance associated transformation
	aux11 = Q_var
	aux12 = cp.bmat([[Cz*X+Dz*C_h, Cz+Dz*D_h*C]])
	aux22 = cp.bmat([[X.T, np.eye(nx)],[np.eye(nx), Y.T]])
	LMI2 = cp.bmat([[aux11,aux12],[aux12.T,aux22]])
	# LMI 3- D closed loop minimization, 
	LMI3 = Dz*D_h*Dw + Dzw 
	# LMI 4- Trace  minimization, 
	LMI4 = cp.trace(Q_var)
	#--- Constraints ----
	Cons1 = LMI1 << 0
	Cons2 = LMI2 >> 0
	Cons3 = LMI3 == 0
	#Cons4 = LMI4 << nuq
	#Cons5 = nuq >> 0
	consts = [Cons1,Cons2,Cons3]
	#------- SOLVING THE OPTMIZATION PROBLEM
	#{{{
	optprob = cp.Problem(1.e-4*cp.Minimize(LMI4), constraints=consts)
	print("prob is DCP:", optprob.is_dcp())
	result = optprob.solve(solver=cp.CVXOPT,verbose=True, max_iters=10000, abstol=1.e-7, reltol=1.e-7,feastol=1.e-7,refinement = 20,kktsolver='robust')
	print '#------ LMI-SOLUTION-CVXPY(solver) ------#'
	print 'Dinamic output feedback'
	X_LMI = np.matrix(X.value)
	Y_LMI = np.matrix(Y.value)
	Bh_LMI = np.matrix(B_h.value)
	Ah_LMI = np.matrix(A_h.value)
	Dh_LMI = np.matrix(D_h.value)
	Ch_LMI = np.matrix(C_h.value)
	#----------- Solving dynamic output feedback
	MNT = np.asmatrix(np.eye(nx)) - X_LMI*Y_LMI
	u,s,vT = svd(MNT)
	M = np.asmatrix(u)*sqrtm(np.asmatrix(np.diag(s)) )
	N = np.asmatrix(vT).T*sqrtm(np.asmatrix(np.diag(s)) )
	#-----------------
	Dk = Dh_LMI
	Ck = (Ch_LMI - Dk*C*X_LMI)*(M.I).T
	Bk = N.I*(Bh_LMI - Y_LMI*B*Dk)
	Ak = N.I*(Ah_LMI - N*Bk*C*X_LMI - Y_LMI*B*Ck*M.T -Y_LMI*A*X_LMI - Y_LMI*B*Dk*C*      X_LMI)*(M.I).T
	#}}}
	#
	return Ak,Bk,Ck,Dk
	#}}}


#------ LMI-H2, and regional pole COnstraints, output feedback
def Simp_Disc_H2_Out_FB2(Sys,dtd):
	#{{{
 	A = Sys[0]
	B = Sys[1]
	C = Sys[2]
 	P = Sys[3]
	Q = C.T*C # xT*Q*x
	R = 1.e0*np.matrix([[1.,0.,0.],[0.,1.,0.],[0.,0.,2.e2]])
	#
	R_sqrt = 1.e0*sqrtm(R)
	Q_sqrt = sqrtm(Q)
	nx = A.shape[0]
	nu = B.shape[1]
	ny = C.shape[0]
	# LMIs for H2
	#{{{
	Cz = np.append(Q_sqrt,np.zeros((nu,nx)),axis=0)
	Dz = np.append(np.zeros((nx,nu)),R_sqrt,axis=0)
	Bw = P
	Dw = np.asmatrix(np.zeros((ny,nu)) )
	Dzw = np.zeros((Cz.shape[0],B.shape[1]))
	# Problems variables definition
        X = cp.Variable((nx,nx),symmetric=True)
	Y = cp.Variable((nx,nx),symmetric=True)
	A_h = cp.Variable((nx,nx) )
	B_h = cp.Variable((nx,ny) )
	C_h = cp.Variable((nu,nx) )
	D_h = cp.Variable((nu,ny) )
	q = cp.Variable(Cz.shape[0])
	Q_var = cp.diag( q)
	nuq = cp.Variable(1)
	# LMI 1- Stability of the closed loop system, lyapunov H2
	aux11 = -cp.bmat([[X, np.eye(nx)],[np.eye(nx), Y]]) 
	aux12 = cp.bmat([[Bw + B*D_h*Dw],[Y*Bw + B_h*Dw]]) 
	aux13 = cp.bmat([[A*X + B*C_h, A+B*D_h*C],[A_h, Y*A+B_h*C]]) 
	aux22 = -np.eye(aux12.shape[1])
	aux23 = np.zeros((aux22.shape[0],aux13.shape[1]))
	aux33 = aux11
	LMI1 = cp.bmat([[aux11,aux12,aux13],[aux12.T,aux22,aux23],[aux13.T,aux23.T,aux33]])
	# LMI 2- H performance associated transformation
	aux11 = Q_var
	aux12 = cp.bmat([[Cz*X+Dz*C_h, Cz+Dz*D_h*C]])
	aux22 = cp.bmat([[X, np.eye(nx)],[np.eye(nx), Y]])
	LMI2 = cp.bmat([[aux11,aux12],[aux12.T,aux22]])
	# LMI 3- D closed loop minimization, 
	LMI3 = Dz*D_h*Dw + Dzw 
	# LMI 4- Trace  minimization, 
	LMI4 = cp.trace(Q_var)
	#--- Constraints ----
	Cons1 = LMI1 << 0
	Cons2 = LMI2 >> 0
	Cons3 = LMI3 == 0
	#
	#}}}
	# LMIs for pole constraints
	#{{{
	Pi1T_P_Pi1 = cp.bmat([[X, np.eye(nx)],[np.eye(nx), Y]])
	Pi1T_PA_Pi1 = cp.bmat([[A*X + B*C_h, A+B*D_h*C],[A_h, Y*A+B_h*C]]) 
	Matrix_LMI = (Pi1T_P_Pi1, Pi1T_PA_Pi1)
	# for speed of the poles
	alfa = 30. # s+alfa  (poles) 
	#nd = 2*nx # dimension of the space where the poles will be
	R11 = np.matrix([[-np.exp(-2*alfa*dtd), 0.],[0.,-1.]])#-np.eye(nd)*np.exp(-2.*alfa*dtd)
	R12 = np.matrix([[0.,1.],[0.,0.]])#np.zeros((nd,nd))
	R22 = np.matrix([[0.,0.],[0.,0.]])#np.eye(nd)
	M_DSGN = (R11,R12,R22)
	LMI5 = Pole_Placement_Discrete(Matrix_LMI, M_DSGN)
	Cons5 = LMI5 << 0
	# damping constraints 
	xi_min = 0.48
	thet_ang = np.arccos(xi_min)
	case = 5 # approximation by: '0' for inner circle, '1' for inner ellipse, '2' half right-plane and ellipse, '3' ellipse cone, '4' ellipse+cone+right half-plane, '5' RHP
	if case == 0:
		# {{{for a circle approximation
		x_m = np.exp(-thet_ang/np.tan(thet_ang))*np.cos(-thet_ang)
		y_m = -np.exp(-thet_ang/np.tan(thet_ang))*np.sin(-thet_ang)
		x_0 = -np.exp(-np.pi/np.tan(thet_ang))
		ak = x_m-x_0
		bk = y_m
		r_k = min(ak,bk)
		# LMI
		R11 = np.matrix([[x_m**2 - r_k**2]])
		R12 = -np.matrix([[x_m]])
		R22 = np.matrix([[1.]])
		M_DSGN = (R11,R12,R22)
		LMI6 = Pole_Placement_Discrete(Matrix_LMI, M_DSGN)
		#------- Draw constraints -------#
		Damp = (xi_min, dtd)
		Circ_Approx = (x_m,r_k)
		Constraints_Draw(Damp,Circ_Approx,'Circle')
		#}}}
		#
	if case == 1:
		#{{{ for a ellipse approximation
		x_m = np.exp(-thet_ang/np.tan(thet_ang))*np.cos(-thet_ang)
		y_m = -np.exp(-thet_ang/np.tan(thet_ang))*np.sin(-thet_ang)
		x_0 = -np.exp(-np.pi/np.tan(thet_ang))
		ak = x_m-x_0
		bk = y_m
		# LMI
		R11 = np.matrix([[-1.,-x_m/ak],[-x_m/ak,-1.]])
		R12 = np.matrix([[0.,(1./ak)*0.5 - (1./bk)*0.5],[(1./ak)*0.5 + (1./bk)*0.5,0.]])
		R22 = np.matrix([[0.,0.],[0.,0.]])
		M_DSGN = (R11,R12,R22)
		LMI6 = Pole_Placement_Discrete(Matrix_LMI, M_DSGN)
		#------- Draw constraints -------#
		Damp = (xi_min, dtd)
		Ellip_Approx = (ak,bk,x_m)
		Constraints_Draw(Damp,Ellip_Approx,'Ellipse')
		#}}}
		#
	if case == 2:
		#{{{ for a half plane and ellipse approximation
		x_m = np.exp(-thet_ang/np.tan(thet_ang))*np.cos(-thet_ang)
		y_m = -np.exp(-thet_ang/np.tan(thet_ang))*np.sin(-thet_ang)
		x_0 = -np.exp(-np.pi/np.tan(thet_ang))
		y_3 = np.exp(-np.pi/(2.*np.tan(thet_ang)) ) 
		ak = x_m*y_m/np.sqrt(y_m**2-y_3**2)
		bk = y_m
		# LMI
		R11 = np.matrix([[0.,0.,0.],[0.,-1.,-x_m/ak],[0.,-x_m/ak,-1.]])
		R12 = np.matrix([[-1.,0.,0.],[0.,0.,(1./ak)*0.5 - (1./bk)*0.5],[0.,(1./ak)*0.5 + (1./bk)*0.5,0.]])
		R22 = np.matrix([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]])
		M_DSGN = (R11,R12,R22)
		LMI6 = Pole_Placement_Discrete(Matrix_LMI, M_DSGN)
		#------- Draw constraints -------#
		Damp = (xi_min, dtd)
		Ellip_Approx = (ak,bk,x_m,y_3,x_0)
		Constraints_Draw(Damp,Ellip_Approx,'HP_Ellipse')
		#}}}
		#
	if case == 3:
		#{{{ for a ellipse-cone approximation
		x_m = np.exp(-thet_ang/np.tan(thet_ang))*np.cos(-thet_ang)
		y_m = -np.exp(-thet_ang/np.tan(thet_ang))*np.sin(-thet_ang)
		x_0 = -np.exp(-np.pi/np.tan(thet_ang))
		y_3 = np.exp(-np.pi/(2.*np.tan(thet_ang)) ) 
		a_e = x_m*y_m/np.sqrt(y_m**2-y_3**2)
		b_e = y_m
		#------ ellipse centered inner cordiode
		xe = 0.6 # interseccion cardiode and cone
		ye = b_e*np.sin( np.arccos( (xe-x_m)/a_e ) ) # interseccion con cardioide
		xse = (1.+x_0)/2. # new center of the ellipse
		ak = (1.-x_0)/2. # new 'a' 
		bk = ye*ak/np.sqrt(ak**2 - (xe-xse)**2) # new 'b'
		# matices
		R11e = np.matrix([[-1.,-xse/ak],[-xse/ak,-1.]])
		R12e = np.matrix([[0.,(1./ak)*0.5 - (1./bk)*0.5],[(1./ak)*0.5 + (1./bk)*0.5,0.]])
		R22e = np.matrix([[0.,0.],[0.,0.]])
		#-------- cone with vetex at (xv,0)
		gama = np.arctan(ye/(1.-xe))
		xv = 1.
		R11v = np.matrix([[-xv*np.sin(gama)*2, 0.],[0., -xv*np.sin(gama)*2]])
		R12v = np.matrix([[np.sin(gama), np.cos(gama)],[-np.cos(gama), np.sin(gama)]])
		R22v = np.matrix([[0.,0.],[0.,0.]])
		#--------- Intersection ellipse and cone
		Z = np.zeros((2,2))
		aux1 = np.append(R11e,Z,axis=1)
		aux2 = np.append(Z, R11v,axis=1)
		R11 = np.append(aux1,aux2, axis=0)
		aux1 = np.append(R12e,Z,axis=1)
		aux2 = np.append(Z, R12v,axis=1)
		R12 = np.append(aux1,aux2, axis=0)
		aux1 = np.append(R22e,Z,axis=1)
		aux2 = np.append(Z, R22v,axis=1)
		R22 = np.append(aux1,aux2, axis=0)
		#-------------------------------
		M_DSGN = (R11,R12,R22)
		LMI6 = Pole_Placement_Discrete(Matrix_LMI, M_DSGN)
		#------- Draw constraints -------#
		Damp = (xi_min, dtd)
		Ellip_Approx = (ak,bk,xse,xe, gama,xv )
		Constraints_Draw(Damp,Ellip_Approx,'Ellipse_Cone')
		#}}}
		#
	if case == 4:
		#{{{ for a ellipse-cone approximation and right half plane
		x_m = np.exp(-thet_ang/np.tan(thet_ang))*np.cos(-thet_ang)
		y_m = -np.exp(-thet_ang/np.tan(thet_ang))*np.sin(-thet_ang)
		x_0 = -np.exp(-np.pi/np.tan(thet_ang))
		y_3 = np.exp(-np.pi/(2.*np.tan(thet_ang)) ) 
		a_e = x_m*y_m/np.sqrt(y_m**2-y_3**2)
		b_e = y_m
		#
		k = np.tan(thet_ang)
		t = np.arange(-np.pi/k,0, 0.01)
		#------ ellipse centered inner cordiode
		xe = 0.6 # interseccion cardiode and cone
		for i in xrange(t.shape[0]):
			error = xe-np.exp(t[i])*np.cos(k*t[i])
			if error < 0.01:
				t_aux = t[i]
				break
		ye = abs(np.exp(t_aux)*np.sin(k*t_aux))
		#print ye
		xse = (1.+x_0)/2. # new center of the ellipse
		ak = (1.-x_0)/2. # new 'a' 
		bk = ye*ak/np.sqrt(ak**2 - (xe-xse)**2) # new 'b'
		# matices
		R11e = np.matrix([[-1.,-xse/ak],[-xse/ak,-1.]])
		R12e = np.matrix([[0.,(1./ak)*0.5 - (1./bk)*0.5],[(1./ak)*0.5 + (1./bk)*0.5,0.]])
		R22e = np.matrix([[0.,0.],[0.,0.]])
		#-------- cone with vetex at (xv,0)
		gama = np.arctan(ye/(1.-xe))
		xv = 1.
		R11v = np.matrix([[-xv*np.sin(gama)*2, 0.],[0., -xv*np.sin(gama)*2]])
		R12v = np.matrix([[np.sin(gama), np.cos(gama)],[-np.cos(gama), np.sin(gama)]])
		R22v = np.matrix([[0.,0.],[0.,0.]])
		#---------- right half plane
		R11h = np.matrix([[0.]])
		R12h = np.matrix([[-1.]])
		R22h = np.matrix([[0.]])
		#--------- Intersection ellipse and cone
		Z = np.zeros((2,2))
		Z2 = np.zeros((2,1))
		aux_a = np.append(R11e,Z,axis=1)
		aux1 = np.append(aux_a,Z2,axis=1)
		aux_b = np.append(Z, R11v,axis=1)
		aux2 = np.append(aux_b, Z2,axis=1)
		aux_c = np.append(Z2.T, Z2.T,axis=1)
		aux3 = np.append(aux_c, R11h,axis=1)
		aux_R = np.append(aux1,aux2, axis=0)
		R11 = np.append(aux_R,aux3, axis=0)
		#
		aux_a = np.append(R12e,Z,axis=1)
		aux1 = np.append(aux_a,Z2,axis=1)
		aux_b = np.append(Z, R12v,axis=1)
		aux2 = np.append(aux_b, Z2,axis=1)
		aux_c = np.append(Z2.T, Z2.T,axis=1)
		aux3 = np.append(aux_c, R12h,axis=1)
		aux_R = np.append(aux1,aux2, axis=0)
		R12 = np.append(aux_R,aux3, axis=0)
		#
		aux_a = np.append(R22e,Z,axis=1)
		aux1 = np.append(aux_a,Z2,axis=1)
		aux_b = np.append(Z, R22v,axis=1)
		aux2 = np.append(aux_b, Z2,axis=1)
		aux_c = np.append(Z2.T, Z2.T,axis=1)
		aux3 = np.append(aux_c, R22h,axis=1)
		aux_R = np.append(aux1,aux2, axis=0)
		R22 = np.append(aux_R,aux3, axis=0)
		#-------------------------------
		M_DSGN = (R11,R12,R22)
		LMI6 = Pole_Placement_Discrete(Matrix_LMI, M_DSGN)
		#------- Draw constraints -------#
		Damp = (xi_min, dtd)
		Ellip_Approx = (ak,bk,xse,xe, gama,xv )
		Constraints_Draw(Damp,Ellip_Approx,'Ellipse_Cone_RHP')
		#}}}

	if case == 5:
		#{{{ right half plane
		#---------- right half plane
		R11h = np.matrix([[0.]])
		R12h = np.matrix([[-1.]])
		R22h = np.matrix([[0.]])
		#--------- Intersection ellipse and cone
		R11 = R11h
		R12 = R12h
		R22 = R22h
		#-------------------------------
		M_DSGN = (R11,R12,R22)
		LMI6 = Pole_Placement_Discrete(Matrix_LMI, M_DSGN)
		#}}}
	Cons6 = LMI6 << 0
	#}}}
	

	####################################################
	consts = [Cons1,Cons2,Cons3,Cons5,Cons6]
	####################################################



	#Bj = Bw*Rj
	#Pi1T_P_Pi1 = cp.bmat([[X, np.eye(nx)],[np.eye(nx), Y]])
	#Pi2T_Bj = cp.bmat([[Bj + B*D_h*Fj],[Y*Bj+B_h*Fj]]) 
	#Cj_Pi1 = cp.bmat([[Cj*X + Ej*C_h ,Cj+Ej*D_h*C]]) 
	#Matrix_LMI = (Pi1T_P_Pi1, Pi2T_Bj,Cj_Pi1)

	#--------------------------------------------#
	#------- SOLVING THE OPTMIZATION PROBLEM-----#
	#{{{
	optprob = cp.Problem(1.e-5*cp.Minimize(LMI4), constraints=consts)
	#optprob = cp.Problem(cp.Minimize(0.), constraints=consts)
	print("prob is DCP:", optprob.is_dcp())
	result = optprob.solve(solver=cp.CVXOPT,verbose=True, max_iters=10000, abstol=1.e-7, reltol=1.e-7,feastol=1.e-7,refinement = 40,kktsolver='robust')
	print '#------ LMI-SOLUTION-CVXPY(solver) ------#'
	print 'Dinamic output feedback'
	X_LMI = np.matrix(X.value)
	Y_LMI = np.matrix(Y.value)
	Bh_LMI = np.matrix(B_h.value)
	Ah_LMI = np.matrix(A_h.value)
	Dh_LMI = np.matrix(D_h.value)
	Ch_LMI = np.matrix(C_h.value)
	#----------- Solving dynamic output feedback
	MNT = np.asmatrix(np.eye(nx)) - X_LMI*Y_LMI
	u,s,vT = svd(MNT)
	M = np.asmatrix(u)*sqrtm(np.asmatrix(np.diag(s)) )
	N = np.asmatrix(vT).T*sqrtm(np.asmatrix(np.diag(s)) )
	#-----------------
	Dk = Dh_LMI
	Ck = (Ch_LMI - Dk*C*X_LMI)*(M.I).T
	Bk = N.I*(Bh_LMI - Y_LMI*B*Dk)
	Ak = N.I*(Ah_LMI - N*Bk*C*X_LMI - Y_LMI*B*Ck*M.T -Y_LMI*A*X_LMI - Y_LMI*B*Dk*C*      X_LMI)*(M.I).T
	#}}}
	#u_nom = np.matrix(np.zeros((2,x_nom.shape[1])) )
	#K_ss = Ck*(np.eye(Ak.shape[0])-Ak).I*Bk + Dk
	#print K_ss
	#for i in xrange(x_nom.shape[1]):
	#	y_nom = np.matrix([[x_nom[0,i],x_nom[1,i] ]]).T
		#print y_nom
	#	u_nom[:,i] = K_ss*y_nom

	#--------------------------------------------#
	return Ak,Bk,Ck,Dk
	#}}}


#------ LMI: H2, RPP and H_inf output feedback
def Simp_Disc_H2_Out_FB3(Sys,dtd):
	#{{{
 	A = Sys[0]
	B = Sys[1]
	C = Sys[2]
 	P = Sys[3]
	C_aux = C#np.matrix([[0.,0.,0.,0.,1.,0.,0.],[0.,0.,0.,0.,0.,1.,0.]])
	Q = C_aux.T*C_aux # xT*Q*x
	R = 1.e0*np.matrix([[1.,0.,0],[0.,1.,0],[0.,0.,1.]])
	#
	R_sqrt = 1.e0*sqrtm(R)
	Q_sqrt = sqrtm(Q)
	nx = A.shape[0]
	nu = B.shape[1]
	ny = C.shape[0]
	# LMIs for H2
	#{{{
	Cz = np.append(Q_sqrt,np.zeros((nu,nx)),axis=0)
	Dz = np.append(np.zeros((nx,nu)),R_sqrt,axis=0)
	Bw = P
	Dw = np.asmatrix(np.zeros((ny,nu)) )
	Dzw = np.zeros((Cz.shape[0],B.shape[1]))
	# Problems variables definition
        X = cp.Variable((nx,nx),symmetric=True)
	Y = cp.Variable((nx,nx),symmetric=True)
	A_h = cp.Variable((nx,nx) )
	B_h = cp.Variable((nx,ny) )
	C_h = cp.Variable((nu,nx) )
	D_h = cp.Variable((nu,ny) )
	q = cp.Variable(Cz.shape[0])
	Q_var = cp.diag( q)
	nuq = cp.Variable(1)
	# LMI 1- Stability of the closed loop system, lyapunov H2
	aux11 = -cp.bmat([[X, np.eye(nx)],[np.eye(nx), Y]]) 
	aux12 = cp.bmat([[Bw + B*D_h*Dw],[Y*Bw + B_h*Dw]]) 
	aux13 = cp.bmat([[A*X + B*C_h, A+B*D_h*C],[A_h, Y*A+B_h*C]]) 
	aux22 = -np.eye(aux12.shape[1])
	aux23 = np.zeros((aux22.shape[0],aux13.shape[1]))
	aux33 = aux11
	LMI1 = cp.bmat([[aux11,aux12,aux13],[aux12.T,aux22,aux23],[aux13.T,aux23.T,aux33]])
	# LMI 2- H performance associated transformation
	aux11 = Q_var
	aux12 = cp.bmat([[Cz*X+Dz*C_h, Cz+Dz*D_h*C]])
	aux22 = cp.bmat([[X, np.eye(nx)],[np.eye(nx), Y]])
	LMI2 = cp.bmat([[aux11,aux12],[aux12.T,aux22]])
	# LMI 3- D closed loop minimization, 
	LMI3 = Dz*D_h*Dw + Dzw 
	# LMI 4- Trace  minimization, 
	LMI4 = cp.trace(Q_var)
	#--- Constraints ----
	Cons1 = LMI1 << 0
	Cons2 = LMI2 >> 0
	Cons3 = LMI3 == 0
	#
	#}}}
	# LMIs for pole constraints
	#{{{
	Pi1T_P_Pi1 = cp.bmat([[X, np.eye(nx)],[np.eye(nx), Y]])
	Pi1T_PA_Pi1 = cp.bmat([[A*X + B*C_h, A+B*D_h*C],[A_h, Y*A+B_h*C]]) 
	Matrix_LMI = (Pi1T_P_Pi1, Pi1T_PA_Pi1)
	# for speed of the poles
	alfa = 25. # s+alfa  (poles) 
	#nd = 2*nx # dimension of the space where the poles will be
	R11 = np.matrix([[-np.exp(-2*alfa*dtd), 0.],[0.,-1.]])#-np.eye(nd)*np.exp(-2.*alfa*dtd)
	R12 = np.matrix([[0.,1.],[0.,0.]])#np.zeros((nd,nd))
	R22 = np.matrix([[0.,0.],[0.,0.]])#np.eye(nd)
	M_DSGN = (R11,R12,R22)
	LMI5 = Pole_Placement_Discrete(Matrix_LMI, M_DSGN)
	Cons5 = LMI5 << 0
	# damping constraints 
	xi_min = 0.68
	thet_ang = np.arccos(xi_min)
	case = 5 # approximation by: '0' for inner circle, '1' for inner ellipse, '2' half right-plane and ellipse, '3' ellipse cone, '4' ellipse+cone+right half-plane, '5' RPP+speed
	if case == 0:
		# {{{for a circle approximation
		x_m = np.exp(-thet_ang/np.tan(thet_ang))*np.cos(-thet_ang)
		y_m = -np.exp(-thet_ang/np.tan(thet_ang))*np.sin(-thet_ang)
		x_0 = -np.exp(-np.pi/np.tan(thet_ang))
		ak = x_m-x_0
		bk = y_m
		r_k = min(ak,bk)
		# LMI
		R11 = np.matrix([[x_m**2 - r_k**2]])
		R12 = -np.matrix([[x_m]])
		R22 = np.matrix([[1.]])
		M_DSGN = (R11,R12,R22)
		LMI6 = Pole_Placement_Discrete(Matrix_LMI, M_DSGN)
		#------- Draw constraints -------#
		Damp = (xi_min, dtd)
		Circ_Approx = (x_m,r_k)
		Constraints_Draw(Damp,Circ_Approx,'Circle')
		#}}}
		#
	if case == 1:
		#{{{ for a ellipse approximation
		x_m = np.exp(-thet_ang/np.tan(thet_ang))*np.cos(-thet_ang)
		y_m = -np.exp(-thet_ang/np.tan(thet_ang))*np.sin(-thet_ang)
		x_0 = -np.exp(-np.pi/np.tan(thet_ang))
		ak = x_m-x_0
		bk = y_m
		# LMI
		R11 = np.matrix([[-1.,-x_m/ak],[-x_m/ak,-1.]])
		R12 = np.matrix([[0.,(1./ak)*0.5 - (1./bk)*0.5],[(1./ak)*0.5 + (1./bk)*0.5,0.]])
		R22 = np.matrix([[0.,0.],[0.,0.]])
		M_DSGN = (R11,R12,R22)
		LMI6 = Pole_Placement_Discrete(Matrix_LMI, M_DSGN)
		#------- Draw constraints -------#
		Damp = (xi_min, dtd)
		Ellip_Approx = (ak,bk,x_m)
		Constraints_Draw(Damp,Ellip_Approx,'Ellipse')
		#}}}
		#
	if case == 2:
		#{{{ for a half plane and ellipse approximation
		x_m = np.exp(-thet_ang/np.tan(thet_ang))*np.cos(-thet_ang)
		y_m = -np.exp(-thet_ang/np.tan(thet_ang))*np.sin(-thet_ang)
		x_0 = -np.exp(-np.pi/np.tan(thet_ang))
		y_3 = np.exp(-np.pi/(2.*np.tan(thet_ang)) ) 
		ak = x_m*y_m/np.sqrt(y_m**2-y_3**2)
		bk = y_m
		# LMI
		R11 = np.matrix([[0.,0.,0.],[0.,-1.,-x_m/ak],[0.,-x_m/ak,-1.]])
		R12 = np.matrix([[-1.,0.,0.],[0.,0.,(1./ak)*0.5 - (1./bk)*0.5],[0.,(1./ak)*0.5 + (1./bk)*0.5,0.]])
		R22 = np.matrix([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]])
		M_DSGN = (R11,R12,R22)
		LMI6 = Pole_Placement_Discrete(Matrix_LMI, M_DSGN)
		#------- Draw constraints -------#
		Damp = (xi_min, dtd)
		Ellip_Approx = (ak,bk,x_m,y_3,x_0)
		Constraints_Draw(Damp,Ellip_Approx,'HP_Ellipse')
		#}}}
		#
	if case == 3:
		#{{{ for a ellipse-cone approximation
		x_m = np.exp(-thet_ang/np.tan(thet_ang))*np.cos(-thet_ang)
		y_m = -np.exp(-thet_ang/np.tan(thet_ang))*np.sin(-thet_ang)
		x_0 = -np.exp(-np.pi/np.tan(thet_ang))
		y_3 = np.exp(-np.pi/(2.*np.tan(thet_ang)) ) 
		a_e = x_m*y_m/np.sqrt(y_m**2-y_3**2)
		b_e = y_m
		#------ ellipse centered inner cordiode
		xe = 0.5 # interseccion cardiode and cone
		ye = b_e*np.sin( np.arccos( (xe-x_m)/a_e ) ) # interseccion con cardioide
		xse = (1.+x_0)/2. # new center of the ellipse
		ak = (1.-x_0)/2. # new 'a' 
		bk = ye*ak/np.sqrt(ak**2 - (xe-xse)**2) # new 'b'
		# matices
		R11e = np.matrix([[-1.,-xse/ak],[-xse/ak,-1.]])
		R12e = np.matrix([[0.,(1./ak)*0.5 - (1./bk)*0.5],[(1./ak)*0.5 + (1./bk)*0.5,0.]])
		R22e = np.matrix([[0.,0.],[0.,0.]])
		#-------- cone with vetex at (xv,0)
		gama = np.arctan(ye/(1.-xe))
		xv = 1.
		R11v = np.matrix([[-xv*np.sin(gama)*2, 0.],[0., -xv*np.sin(gama)*2]])
		R12v = np.matrix([[np.sin(gama), np.cos(gama)],[-np.cos(gama), np.sin(gama)]])
		R22v = np.matrix([[0.,0.],[0.,0.]])
		#--------- Intersection ellipse and cone
		Z = np.zeros((2,2))
		aux1 = np.append(R11e,Z,axis=1)
		aux2 = np.append(Z, R11v,axis=1)
		R11 = np.append(aux1,aux2, axis=0)
		aux1 = np.append(R12e,Z,axis=1)
		aux2 = np.append(Z, R12v,axis=1)
		R12 = np.append(aux1,aux2, axis=0)
		aux1 = np.append(R22e,Z,axis=1)
		aux2 = np.append(Z, R22v,axis=1)
		R22 = np.append(aux1,aux2, axis=0)
		#-------------------------------
		M_DSGN = (R11,R12,R22)
		LMI6 = Pole_Placement_Discrete(Matrix_LMI, M_DSGN)
		#------- Draw constraints -------#
		Damp = (xi_min, dtd)
		Ellip_Approx = (ak,bk,xse,xe, gama,xv )
		Constraints_Draw(Damp,Ellip_Approx,'Ellipse_Cone')
		#}}}
		#
	if case == 4:
		#{{{ for a ellipse-cone approximation and right half plane
		x_m = np.exp(-thet_ang/np.tan(thet_ang))*np.cos(-thet_ang)
		y_m = -np.exp(-thet_ang/np.tan(thet_ang))*np.sin(-thet_ang)
		x_0 = -np.exp(-np.pi/np.tan(thet_ang))
		y_3 = np.exp(-np.pi/(2.*np.tan(thet_ang)) ) 
		a_e = x_m*y_m/np.sqrt(y_m**2-y_3**2)
		b_e = y_m
		#
		k = np.tan(thet_ang)
		t = np.arange(-np.pi/k,0, 0.01)
		#------ ellipse centered inner cordiode
		xe = 0.6 # interseccion cardiode and cone
		for i in xrange(t.shape[0]):
			error = xe-np.exp(t[i])*np.cos(k*t[i])
			if error < 0.01:
				t_aux = t[i]
				break
		ye = abs(np.exp(t_aux)*np.sin(k*t_aux))
		#print ye
		xse = (1.+x_0)/2. # new center of the ellipse
		ak = (1.-x_0)/2. # new 'a' 
		bk = ye*ak/np.sqrt(ak**2 - (xe-xse)**2) # new 'b'
		# matices
		R11e = np.matrix([[-1.,-xse/ak],[-xse/ak,-1.]])
		R12e = np.matrix([[0.,(1./ak)*0.5 - (1./bk)*0.5],[(1./ak)*0.5 + (1./bk)*0.5,0.]])
		R22e = np.matrix([[0.,0.],[0.,0.]])
		#-------- cone with vetex at (xv,0)
		gama = np.arctan(ye/(1.-xe))
		xv = 1.
		R11v = np.matrix([[-xv*np.sin(gama)*2, 0.],[0., -xv*np.sin(gama)*2]])
		R12v = np.matrix([[np.sin(gama), np.cos(gama)],[-np.cos(gama), np.sin(gama)]])
		R22v = np.matrix([[0.,0.],[0.,0.]])
		#---------- right half plane
		R11h = np.matrix([[0.]])
		R12h = np.matrix([[-1.]])
		R22h = np.matrix([[0.]])
		#--------- Intersection ellipse and cone
		Z = np.zeros((2,2))
		Z2 = np.zeros((2,1))
		aux_a = np.append(R11e,Z,axis=1)
		aux1 = np.append(aux_a,Z2,axis=1)
		aux_b = np.append(Z, R11v,axis=1)
		aux2 = np.append(aux_b, Z2,axis=1)
		aux_c = np.append(Z2.T, Z2.T,axis=1)
		aux3 = np.append(aux_c, R11h,axis=1)
		aux_R = np.append(aux1,aux2, axis=0)
		R11 = np.append(aux_R,aux3, axis=0)
		#
		aux_a = np.append(R12e,Z,axis=1)
		aux1 = np.append(aux_a,Z2,axis=1)
		aux_b = np.append(Z, R12v,axis=1)
		aux2 = np.append(aux_b, Z2,axis=1)
		aux_c = np.append(Z2.T, Z2.T,axis=1)
		aux3 = np.append(aux_c, R12h,axis=1)
		aux_R = np.append(aux1,aux2, axis=0)
		R12 = np.append(aux_R,aux3, axis=0)
		#
		aux_a = np.append(R22e,Z,axis=1)
		aux1 = np.append(aux_a,Z2,axis=1)
		aux_b = np.append(Z, R22v,axis=1)
		aux2 = np.append(aux_b, Z2,axis=1)
		aux_c = np.append(Z2.T, Z2.T,axis=1)
		aux3 = np.append(aux_c, R22h,axis=1)
		aux_R = np.append(aux1,aux2, axis=0)
		R22 = np.append(aux_R,aux3, axis=0)
		#-------------------------------
		M_DSGN = (R11,R12,R22)
		LMI6 = Pole_Placement_Discrete(Matrix_LMI, M_DSGN)
		#------- Draw constraints -------#
		Damp = (xi_min, dtd)
		Ellip_Approx = (ak,bk,xse,xe, gama,xv )
		Constraints_Draw(Damp,Ellip_Approx,'Ellipse_Cone_RHP')
		#}}}
		#
	if case == 5:
		#{{{ right half plane
		#---------- right half plane
		Alfa_L = 0.
		R11h = np.matrix([[0.]])
		R12h = np.matrix([[-1.]])
		R22h = np.matrix([[0.]])
		#--------- Intersection ellipse and cone
		R11 = R11h
		R12 = R12h
		R22 = R22h
		#-------------------------------
		M_DSGN = (R11,R12,R22)
		LMI6 = Pole_Placement_Discrete(Matrix_LMI, M_DSGN)
		#}}}
	Cons6 = LMI6 << Alfa_L
	#}}}

	
	# LMI for H_inf performance
	#{{{
	gamma = 1.2
	Lj = np.eye(3)#np.matrix([[1.,0.,0.]])
	Rj = np.eye(3)#np.matrix([[0.,0.,1.]]).T
	# Different variables to optimize
	case = 0
	if case == 0: # currents vsd, vsq and omega_t
		Cz = np.matrix([[0.,0.,1.,0.,0.,0.,0.],[0.,0.,0.,1.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.]]) 
		Dz = np.matrix([[0.,0.,0.],[0.,0.,0.],[0.,0.,1.]]) 
		Bw = P
		Dw = np.asmatrix(np.zeros((6,3)) )
		Dzw = np.asmatrix(np.zeros((3,3)) )
	if case == 1: # currents iod
		Cz = np.matrix([[0.,0.,0.,0.,1.,0.,0.]]) 
		Dz = np.matrix([[0.,0.,0.]]) 
		Bw = P
		Dw = np.asmatrix(np.zeros((6,3)) )
		Dzw = np.asmatrix(np.zeros((3,3)) )
	#
	Cj = Lj*Cz
	Ej = Lj*Dz
	Bj = Bw*Rj
	Fj = Dw*Rj
	Dj = Lj*Dzw*Rj
	#
	Pi1T_P_Pi1 = cp.bmat([[X, np.eye(nx)],[np.eye(nx), Y]])
	Pi1T_PA_Pi1 = cp.bmat([[A*X + B*C_h, A+B*D_h*C],[A_h, Y*A+B_h*C]]) 
	Cj_Pi1 = cp.bmat([[Cj*X+Ej*C_h, Cj+Ej*D_h*C]])
	PiT_P_Bj = cp.bmat([[Bj+B*D_h*Fj],[ Y*Bj+B_h*Fj]])
	#
	aux11 = -Pi1T_P_Pi1
	aux13 = Pi1T_PA_Pi1.T
	aux14 = Cj_Pi1.T
	aux23 = PiT_P_Bj.T
	aux24 = (Dj+Ej*D_h*Fj).T
	aux33 = aux11
	aux44 = -np.eye(aux24.shape[0])
	aux12 = np.zeros((aux11.shape[0],aux23.shape[0] ))
	aux22 = -np.eye(aux23.shape[0])*gamma**2
	aux34 = np.zeros((aux33.shape[0],aux24.shape[1]))
	#
	LMI7 = cp.bmat([[aux11,aux12,aux13,aux14],[aux12.T,aux22,aux23,aux24],[aux13.T,aux23.T,aux33, aux34],[aux14.T,aux24.T,aux34.T,aux44]])
	Cons7 = LMI7 << 0
	Cons8 = Dj+Ej*D_h*Fj == 0
	#}}}

	
	####################################################
	consts = [Cons1,Cons2,Cons3,Cons5,Cons7,Cons8]
	####################################################


	#--------------------------------------------#
	#------- SOLVING THE OPTMIZATION PROBLEM-----#
	#{{{
	optprob = cp.Problem(1.e-5*cp.Minimize(LMI4), constraints=consts)
	#optprob = cp.Problem(cp.Minimize(0.), constraints=consts)
	print("prob is DCP:", optprob.is_dcp())
	result = optprob.solve(solver=cp.CVXOPT,verbose=True, max_iters=10000, abstol=1.e-7, reltol=1.e-7,feastol=1.e-7,refinement = 100,kktsolver='robust')
	print '#------ LMI-SOLUTION-CVXPY(solver) ------#'
	print 'Dinamic output feedback'
	X_LMI = np.matrix(X.value)
	Y_LMI = np.matrix(Y.value)
	Bh_LMI = np.matrix(B_h.value)
	Ah_LMI = np.matrix(A_h.value)
	Dh_LMI = np.matrix(D_h.value)
	Ch_LMI = np.matrix(C_h.value)
	#----------- Solving dynamic output feedback
	MNT = np.asmatrix(np.eye(nx)) - X_LMI*Y_LMI
	u,s,vT = svd(MNT)
	M = np.asmatrix(u)*sqrtm(np.asmatrix(np.diag(s)) )
	N = np.asmatrix(vT).T*sqrtm(np.asmatrix(np.diag(s)) )
	#-----------------
	Dk = Dh_LMI
	Ck = (Ch_LMI - Dk*C*X_LMI)*(M.I).T
	Bk = N.I*(Bh_LMI - Y_LMI*B*Dk)
	Ak = N.I*(Ah_LMI - N*Bk*C*X_LMI - Y_LMI*B*Ck*M.T -Y_LMI*A*X_LMI - Y_LMI*B*Dk*C*      X_LMI)*(M.I).T
	#}}}
	#u_nom = np.matrix(np.zeros((2,x_nom.shape[1])) )
	#K_ss = Ck*(np.eye(Ak.shape[0])-Ak).I*Bk + Dk
	#print K_ss
	#for i in xrange(x_nom.shape[1]):
	#	y_nom = np.matrix([[x_nom[0,i],x_nom[1,i] ]]).T
		#print y_nom
	#	u_nom[:,i] = K_ss*y_nom

	#--------------------------------------------#
	return Ak,Bk,Ck,Dk
	#}}}


#------ LMI: H2, RPP, H_inf and P2P output feedback
def Simp_Disc_H2_Out_FB4(Sys,dtd,y_nom,x_nom):
	#{{{
	consts = []
 	A = Sys[0]
	B = Sys[1]
	C = Sys[2]
 	P = Sys[3]
 	Dw = Sys[4]
	C_aux = np.matrix([[1.,0.,0.,0.,0.,0.,0.],[0.,1.,0.,0.,0.,0.,0.],[0.,0.,1.,0.,0.,0.,0.],[0.,0.,0.,1.,0.,0.,0.],[0.,0,0.,0.,1.,0.,0.],[0.,0.,0.,0.,0.,1.,0.],[0.,0.,0.,0.,0.,0.,0.]])
	Q = C.T*C#C_aux.T*C_aux ## xT*Q*x
	#R =1.e7*np.matrix([[1./(377.**2),0.,0],[0.,1./(377.**2),0],[0.,0.,0e-1]])
	R =1.e5*np.matrix([[1.,0.,0],[0.,1.,0],[0.,0.,0e-4]])
	#
	R_sqrt = 1.e0*sqrtm(R)
	Q_sqrt = sqrtm(Q)
	nx = A.shape[0]
	nu = B.shape[1]
	ny = C.shape[0]
	# LMIs for H2
	#{{{
	Cz = np.append(Q_sqrt,np.zeros((nu,nx)),axis=0)
	Dz = np.append(np.zeros((nx,nu)),R_sqrt,axis=0)
	Bw = P
	Dw = Dw#np.asmatrix(np.zeros((ny,nu)) )
	#Dzw = np.zeros((Cz.shape[0],B.shape[1]))
	Dzw = np.zeros((Cz.shape[0],P.shape[1]))
	# Problems variables definition
        X = cp.Variable((nx,nx),symmetric=True)
	Y = cp.Variable((nx,nx),symmetric=True)
	A_h = cp.Variable((nx,nx) )
	B_h = cp.Variable((nx,ny) )
	C_h = cp.Variable((nu,nx) )
	D_h = cp.Variable((nu,ny) )
	q = cp.Variable(Cz.shape[0])
	Q_var = cp.diag( q)
	nuq = cp.Variable(1)
	# LMI 1- Stability of the closed loop system, lyapunov H2
	aux11 = -cp.bmat([[X, np.eye(nx)],[np.eye(nx), Y]]) 
	aux12 = cp.bmat([[Bw + B*D_h*Dw],[Y*Bw + B_h*Dw]]) 
	aux13 = cp.bmat([[A*X + B*C_h, A+B*D_h*C],[A_h, Y*A+B_h*C]]) 
	aux22 = -np.eye(aux12.shape[1])
	aux23 = np.zeros((aux22.shape[0],aux13.shape[1]))
	aux33 = aux11
	LMI1 = cp.bmat([[aux11,aux12,aux13],[aux12.T,aux22,aux23],[aux13.T,aux23.T,aux33]])
	# LMI 2- H performance associated transformation
	aux11 = Q_var
	aux12 = cp.bmat([[Cz*X+Dz*C_h, Cz+Dz*D_h*C]])
	aux22 = cp.bmat([[X, np.eye(nx)],[np.eye(nx), Y]])
	LMI2 = cp.bmat([[aux11,aux12],[aux12.T,aux22]])
	# LMI 3- D closed loop minimization, 
	LMI3 = Dz*D_h*Dw + Dzw 
	# LMI 4- Trace  minimization, 
	LMI4 = cp.trace(Q_var)
	#--- Constraints ----
	Cons1 = LMI1 << 0
	Cons2 = LMI2 >> 0
	Cons3 = LMI3 == 0
	#
	#}}}
	consts += [Cons1,Cons2,Cons3]
	# LMIs for pole constraints
	#{{{
	Pi1T_P_Pi1 = cp.bmat([[X, np.eye(nx)],[np.eye(nx), Y]])
	Pi1T_PA_Pi1 = cp.bmat([[A*X + B*C_h, A+B*D_h*C],[A_h, Y*A+B_h*C]]) 
	Matrix_LMI = (Pi1T_P_Pi1, Pi1T_PA_Pi1)
	# for speed of the poles
	alfa = 25. # s+alfa  (poles) 
	#nd = 2*nx # dimension of the space where the poles will be
	R11 = np.matrix([[-np.exp(-2*alfa*dtd), 0.],[0.,-1.]])#-np.eye(nd)*np.exp(-2.*alfa*dtd)
	R12 = np.matrix([[0.,1.],[0.,0.]])#np.zeros((nd,nd))
	R22 = np.matrix([[0.,0.],[0.,0.]])#np.eye(nd)
	M_DSGN = (R11,R12,R22)
	LMI5 = Pole_Placement_Discrete(Matrix_LMI, M_DSGN)
	Cons5 = LMI5 << 0
	# damping constraints 
	xi_min = 0.6
	thet_ang = np.arccos(xi_min)
	case = 5 # approximation by: '0' for inner circle, '1' for inner ellipse, '2' half right-plane and ellipse, '3' ellipse cone, '4' ellipse+cone+right half-plane, '5' RPP+speed
	if case == 0:
		# {{{for a circle approximation
		x_m = np.exp(-thet_ang/np.tan(thet_ang))*np.cos(-thet_ang)
		y_m = -np.exp(-thet_ang/np.tan(thet_ang))*np.sin(-thet_ang)
		x_0 = -np.exp(-np.pi/np.tan(thet_ang))
		ak = x_m-x_0
		bk = y_m
		r_k = min(ak,bk)
		# LMI
		R11 = np.matrix([[x_m**2 - r_k**2]])
		R12 = -np.matrix([[x_m]])
		R22 = np.matrix([[1.]])
		M_DSGN = (R11,R12,R22)
		LMI6 = Pole_Placement_Discrete(Matrix_LMI, M_DSGN)
		#------- Draw constraints -------#
		Damp = (xi_min, dtd)
		Circ_Approx = (x_m,r_k)
		Constraints_Draw(Damp,Circ_Approx,'Circle')
		#}}}
		#
	if case == 1:
		#{{{ for a ellipse approximation
		x_m = np.exp(-thet_ang/np.tan(thet_ang))*np.cos(-thet_ang)
		y_m = -np.exp(-thet_ang/np.tan(thet_ang))*np.sin(-thet_ang)
		x_0 = -np.exp(-np.pi/np.tan(thet_ang))
		ak = x_m-x_0
		bk = y_m
		# LMI
		R11 = np.matrix([[-1.,-x_m/ak],[-x_m/ak,-1.]])
		R12 = np.matrix([[0.,(1./ak)*0.5 - (1./bk)*0.5],[(1./ak)*0.5 + (1./bk)*0.5,0.]])
		R22 = np.matrix([[0.,0.],[0.,0.]])
		M_DSGN = (R11,R12,R22)
		LMI6 = Pole_Placement_Discrete(Matrix_LMI, M_DSGN)
		#------- Draw constraints -------#
		Damp = (xi_min, dtd)
		Ellip_Approx = (ak,bk,x_m)
		Constraints_Draw(Damp,Ellip_Approx,'Ellipse')
		#}}}
		#
	if case == 2:
		#{{{ for a half plane and ellipse approximation
		x_m = np.exp(-thet_ang/np.tan(thet_ang))*np.cos(-thet_ang)
		y_m = -np.exp(-thet_ang/np.tan(thet_ang))*np.sin(-thet_ang)
		x_0 = -np.exp(-np.pi/np.tan(thet_ang))
		y_3 = np.exp(-np.pi/(2.*np.tan(thet_ang)) ) 
		ak = x_m*y_m/np.sqrt(y_m**2-y_3**2)
		bk = y_m
		# LMI
		R11 = np.matrix([[0.,0.,0.],[0.,-1.,-x_m/ak],[0.,-x_m/ak,-1.]])
		R12 = np.matrix([[-1.,0.,0.],[0.,0.,(1./ak)*0.5 - (1./bk)*0.5],[0.,(1./ak)*0.5 + (1./bk)*0.5,0.]])
		R22 = np.matrix([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]])
		M_DSGN = (R11,R12,R22)
		LMI6 = Pole_Placement_Discrete(Matrix_LMI, M_DSGN)
		#------- Draw constraints -------#
		Damp = (xi_min, dtd)
		Ellip_Approx = (ak,bk,x_m,y_3,x_0)
		Constraints_Draw(Damp,Ellip_Approx,'HP_Ellipse')
		#}}}
		#
	if case == 3:
		#{{{ for a ellipse-cone approximation
		x_m = np.exp(-thet_ang/np.tan(thet_ang))*np.cos(-thet_ang)
		y_m = -np.exp(-thet_ang/np.tan(thet_ang))*np.sin(-thet_ang)
		x_0 = -np.exp(-np.pi/np.tan(thet_ang))
		y_3 = np.exp(-np.pi/(2.*np.tan(thet_ang)) ) 
		a_e = x_m*y_m/np.sqrt(y_m**2-y_3**2)
		b_e = y_m
		#------ ellipse centered inner cordiode
		xe = 0.5 # interseccion cardiode and cone
		ye = b_e*np.sin( np.arccos( (xe-x_m)/a_e ) ) # interseccion con cardioide
		xse = (1.+x_0)/2. # new center of the ellipse
		ak = (1.-x_0)/2. # new 'a' 
		bk = ye*ak/np.sqrt(ak**2 - (xe-xse)**2) # new 'b'
		# matices
		R11e = np.matrix([[-1.,-xse/ak],[-xse/ak,-1.]])
		R12e = np.matrix([[0.,(1./ak)*0.5 - (1./bk)*0.5],[(1./ak)*0.5 + (1./bk)*0.5,0.]])
		R22e = np.matrix([[0.,0.],[0.,0.]])
		#-------- cone with vetex at (xv,0)
		gama = np.arctan(ye/(1.-xe))
		xv = 1.
		R11v = np.matrix([[-xv*np.sin(gama)*2, 0.],[0., -xv*np.sin(gama)*2]])
		R12v = np.matrix([[np.sin(gama), np.cos(gama)],[-np.cos(gama), np.sin(gama)]])
		R22v = np.matrix([[0.,0.],[0.,0.]])
		#--------- Intersection ellipse and cone
		Z = np.zeros((2,2))
		aux1 = np.append(R11e,Z,axis=1)
		aux2 = np.append(Z, R11v,axis=1)
		R11 = np.append(aux1,aux2, axis=0)
		aux1 = np.append(R12e,Z,axis=1)
		aux2 = np.append(Z, R12v,axis=1)
		R12 = np.append(aux1,aux2, axis=0)
		aux1 = np.append(R22e,Z,axis=1)
		aux2 = np.append(Z, R22v,axis=1)
		R22 = np.append(aux1,aux2, axis=0)
		#-------------------------------
		M_DSGN = (R11,R12,R22)
		LMI6 = Pole_Placement_Discrete(Matrix_LMI, M_DSGN)
		#------- Draw constraints -------#
		Damp = (xi_min, dtd)
		Ellip_Approx = (ak,bk,xse,xe, gama,xv )
		Constraints_Draw(Damp,Ellip_Approx,'Ellipse_Cone')
		#}}}
		#
	if case == 4:
		#{{{ for a ellipse-cone approximation and right half plane
		x_m = np.exp(-thet_ang/np.tan(thet_ang))*np.cos(-thet_ang)
		y_m = -np.exp(-thet_ang/np.tan(thet_ang))*np.sin(-thet_ang)
		x_0 = -np.exp(-np.pi/np.tan(thet_ang))
		y_3 = np.exp(-np.pi/(2.*np.tan(thet_ang)) ) 
		a_e = x_m*y_m/np.sqrt(y_m**2-y_3**2)
		b_e = y_m
		#
		k = np.tan(thet_ang)
		t = np.arange(-np.pi/k,0, 0.01)
		#------ ellipse centered inner cordiode
		xe = 0.6 # interseccion cardiode and cone
		for i in xrange(t.shape[0]):
			error = xe-np.exp(t[i])*np.cos(k*t[i])
			if error < 0.01:
				t_aux = t[i]
				break
		ye = abs(np.exp(t_aux)*np.sin(k*t_aux))
		#print ye
		xse = (1.+x_0)/2. # new center of the ellipse
		ak = (1.-x_0)/2. # new 'a' 
		bk = ye*ak/np.sqrt(ak**2 - (xe-xse)**2) # new 'b'
		# matices
		R11e = np.matrix([[-1.,-xse/ak],[-xse/ak,-1.]])
		R12e = np.matrix([[0.,(1./ak)*0.5 - (1./bk)*0.5],[(1./ak)*0.5 + (1./bk)*0.5,0.]])
		R22e = np.matrix([[0.,0.],[0.,0.]])
		#-------- cone with vetex at (xv,0)
		gama = np.arctan(ye/(1.-xe))
		xv = 1.
		R11v = np.matrix([[-xv*np.sin(gama)*2, 0.],[0., -xv*np.sin(gama)*2]])
		R12v = np.matrix([[np.sin(gama), np.cos(gama)],[-np.cos(gama), np.sin(gama)]])
		R22v = np.matrix([[0.,0.],[0.,0.]])
		#---------- right half plane
		R11h = np.matrix([[0.]])
		R12h = np.matrix([[-1.]])
		R22h = np.matrix([[0.]])
		#--------- Intersection ellipse and cone
		Z = np.zeros((2,2))
		Z2 = np.zeros((2,1))
		aux_a = np.append(R11e,Z,axis=1)
		aux1 = np.append(aux_a,Z2,axis=1)
		aux_b = np.append(Z, R11v,axis=1)
		aux2 = np.append(aux_b, Z2,axis=1)
		aux_c = np.append(Z2.T, Z2.T,axis=1)
		aux3 = np.append(aux_c, R11h,axis=1)
		aux_R = np.append(aux1,aux2, axis=0)
		R11 = np.append(aux_R,aux3, axis=0)
		#
		aux_a = np.append(R12e,Z,axis=1)
		aux1 = np.append(aux_a,Z2,axis=1)
		aux_b = np.append(Z, R12v,axis=1)
		aux2 = np.append(aux_b, Z2,axis=1)
		aux_c = np.append(Z2.T, Z2.T,axis=1)
		aux3 = np.append(aux_c, R12h,axis=1)
		aux_R = np.append(aux1,aux2, axis=0)
		R12 = np.append(aux_R,aux3, axis=0)
		#
		aux_a = np.append(R22e,Z,axis=1)
		aux1 = np.append(aux_a,Z2,axis=1)
		aux_b = np.append(Z, R22v,axis=1)
		aux2 = np.append(aux_b, Z2,axis=1)
		aux_c = np.append(Z2.T, Z2.T,axis=1)
		aux3 = np.append(aux_c, R22h,axis=1)
		aux_R = np.append(aux1,aux2, axis=0)
		R22 = np.append(aux_R,aux3, axis=0)
		#-------------------------------
		M_DSGN = (R11,R12,R22)
		LMI6 = Pole_Placement_Discrete(Matrix_LMI, M_DSGN)
		#------- Draw constraints -------#
		Damp = (xi_min, dtd)
		Ellip_Approx = (ak,bk,xse,xe, gama,xv )
		Constraints_Draw(Damp,Ellip_Approx,'Ellipse_Cone_RHP')
		#}}}
		#
	if case == 5:
		#{{{ right half plane
		#---------- right half plane
		Alfa_L = 0.
		R11h = np.matrix([[0.]])
		R12h = np.matrix([[-1.]])
		R22h = np.matrix([[0.]])
		#--------- Intersection ellipse and cone
		R11 = R11h
		R12 = R12h
		R22 = R22h
		#-------------------------------
		M_DSGN = (R11,R12,R22)
		LMI6 = Pole_Placement_Discrete(Matrix_LMI, M_DSGN)
		#}}}
	Cons6 = LMI6 << 0
	#}}}

	consts += [Cons5]
	
	# LMI for H_inf performance 
	#{{{
	def Hinf_LMI(aux1,aux2):
		#{{{
		Lj = aux1[0]
		Rj = aux1[1]
		gamma = aux1[2]
		#
		Cz = aux2[0]
		Dz = aux2[1]
		Bw = aux2[2]
		Dw = aux2[3]
		Dzw = aux2[4]
		#
		Cj = Lj*Cz
		Ej = Lj*Dz
		Bj = Bw*Rj
		Fj = Dw*Rj
		Dj = Lj*Dzw*Rj
		#
		Pi1T_P_Pi1 = cp.bmat([[X, np.eye(nx)],[np.eye(nx), Y]])
		Pi1T_PA_Pi1 = cp.bmat([[A*X + B*C_h, A+B*D_h*C],[A_h, Y*A+B_h*C]]) 
		Cj_Pi1 = cp.bmat([[Cj*X+Ej*C_h, Cj+Ej*D_h*C]])
		PiT_P_Bj = cp.bmat([[Bj+B*D_h*Fj],[ Y*Bj+B_h*Fj]])
		#
		aux11 = -Pi1T_P_Pi1
		aux13 = Pi1T_PA_Pi1.T
		aux14 = Cj_Pi1.T
		aux23 = PiT_P_Bj.T
		aux24 = (Dj+Ej*D_h*Fj).T
		aux33 = aux11
		aux44 = -np.eye(aux24.shape[0])
		aux12 = np.zeros((aux11.shape[0],aux23.shape[0] ))
		aux22 = -np.eye(aux23.shape[0])*gamma**2
		aux34 = np.zeros((aux33.shape[0],aux24.shape[1]))
		#
		LMI7 = cp.bmat([[aux11,aux12,aux13,aux14],[aux12.T,aux22,aux23,aux24],[aux13.T,aux23.T,aux33, aux34],[aux14.T,aux24.T,aux34.T,aux44]])
		Cons_a = LMI7 << 0
		Cons_b = Dj+Ej*D_h*Fj == 0
		return Cons_a, Cons_b
		#}}}
	
	# Different variables to optimize
	case = 5
	if case == 0: # currents vsd, vsq and omega_t
		Cz = np.matrix([[0.,0.,1.,0.,0.,0.,0.],[0.,0.,0.,1.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.]]) 
		Dz = np.matrix([[0.,0.,0.],[0.,0.,0.],[0.,0.,1.]]) 
		Bw = P
		Dw = np.asmatrix(np.zeros((6,3)) )
		Dzw = np.asmatrix(np.zeros((3,3)) )
	if case == 1: # currents iod
		Cz = np.matrix([[0.,0.,0.,0.,1.,0.,0.],[0.,0.,0.,0.,0.,1.,0.],[0.,0.,0.,0.,0.,0.,0.]]) 
		Dz = np.matrix([[0.,0.,0.],[0.,0.,0.],[0.,0.,1.]]) 
		Bw = P
		Dw = np.asmatrix(np.zeros((6,3)) )
		Dzw = np.asmatrix(np.zeros((3,3)) )
	if case == 2: # omega_t versus noise in iod ioq
		Cz = np.matrix([[0.,0.,0.,0.,0.,0.,0.]]) 
		Dz = np.matrix([[0.,0.,1.]]) 
		Dzw = np.asmatrix(np.zeros((1,2)) )
		Bw = np.zeros((7,2))
		Dw = np.matrix([[0.,0.],[0.,0.],[0.,0.],[0.,0.],[1.,0.],[0.,1.]])
	if case == 3: # omega_t versus noise in iod ioq in the controlled variables
		Cz = np.matrix([[0.,0.,0.,1.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.]]) 
		Dz = np.matrix([[0.,0.,0.],[0.,0.,1.]]) 
		Dzw = np.zeros((2,2))#np.matrix([[1.,1.]])
		Bw = np.zeros((7,2))
		Dw = np.matrix([[0.,0.],[0.,0.],[0.,0.],[0.,0.],[1.,0.],[0.,1.]])
	if case == 4: # omega_t versus noise vg,wg
		Cz = np.matrix([[0.,0.,0.,0.,0.,0.,0.]]) 
		Dz = np.matrix([[0.,0.,1.]]) 
		Dzw = np.zeros((1,3))#np.matrix([[1.,1.]])
		Bw = P
		Dw = np.zeros((6,3))#np.matrix([[0.,0.],[0.,0.],[0.,0.],[0.,0.],[1.,0.],[0.,1.]])
	if case == 5: # omega_t versus noise vg,wg
		Cz = np.matrix([[0.,0.,1.,0.,0.,0.,0.],[0.,0.,0.,1.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,1.],[0.,0.,0.,0.,0.,0.,0.]]) 
		Dz = np.matrix([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.],[0.,0.,1.]]) 
		Dzw = np.zeros((4,5))#np.matrix([[1.,1.]])
		Bw = P#np.append(P,np.zeros((7,2)),axis=1)
		Dw = Dw#np.append(np.zeros((6,3)),np.append(np.zeros((4,2)),np.eye(2),axis=0),axis=1)

#np.matrix([[0.,0.],[0.,0.],[0.,0.],[0.,0.],[1.,0.],[0.,1.]])
	# w = Rj*wj, zj = Lj*z
	# LMI for H_inf performance for wt / ioq
	gamma =0.01**2
	Lj = np.matrix([[0.,0.,1.,0.]])
	Rj = np.matrix([[0.,0.,0.,1.,0.]]).T
	aux1 = (Lj,Rj,gamma)
	aux2 = (Cz,Dz,Bw,Dw,Dzw)
	#=====================================
	Cons_a, Cons_b = Hinf_LMI(aux1,aux2)
	#consts += [Cons_a,Cons_b]
	#====================================
	# LMI for H_inf performance  for wt / iod
	gamma = 0.01**2
	Lj = np.matrix([[0.,0.,1.,0.]])
	Rj = np.matrix([[0.,0.,0.,0.,1.]]).T
	aux1 = (Lj,Rj,gamma)
	aux2 = (Cz,Dz,Bw,Dw,Dzw)
	#=====================================
	Cons_a, Cons_b = Hinf_LMI(aux1,aux2)
	#consts += [Cons_a,Cons_b]
	#====================================
	# LMI for H_inf performance for voltage wt / wg
	gamma = 0.005**2
	Lj = np.matrix([[0.,0.,0.,1.]])
	Rj = np.matrix([[0.,0.,0.,0.,1.]]).T
	aux1 = (Lj,Rj,gamma)
	aux2 = (Cz,Dz,Bw,Dw,Dzw)
	#=====================================
	Cons_a, Cons_b = Hinf_LMI(aux1,aux2)
	#consts += [Cons_a,Cons_b]
	#====================================
	# LMI for H_inf performance for voltage vs / wg
	gamma = 0.001**2
	Lj = np.matrix([[0.,0.,1.,0.]])
	Rj = np.matrix([[0.,0.,0.,0.,1.]]).T
	aux1 = (Lj,Rj,gamma)
	aux2 = (Cz,Dz,Bw,Dw,Dzw)
	#=====================================
	Cons_a, Cons_b = Hinf_LMI(aux1,aux2)
	#consts += [Cons_a,Cons_b]
	#====================================
	

	#}}}

	####################################################
	
	# Peak to Peak
	#{{{
	# w(3), for voltage
	V_max = 520.*0.1
	Lj = 1.
	Cz = np.matrix([[0.,0.,1.,0.,0.,0.,0.],[0.,0.,0.,1.,0.,0.,0.]]) 
	Dz = np.matrix([[0.,0.,0.],[0.,0.,0.]]) 
	Cj = Lj*Cz
	Ej = Lj*Dz
	aux22 = np.matrix([[V_max**2,0.],[0.,V_max**2]]) # 20 %
	aux12 = cp.bmat([[Cj*X+Ej*C_h, Cj+Ej*D_h*C]])
	aux11 = cp.bmat([[X, np.eye(nx)],[np.eye(nx), Y]])
	LMI = cp.bmat([[aux11,aux12.T],[aux12,aux22]])
	Cons9 = LMI >> 0
	# === w(3), for frequency === #
	F_max = 1.*2.*np.pi
	Lj = 1.
	Cz = np.matrix([[0.,0.,0.,0.,0.,0.,0.]]) 
	Dz = np.matrix([[0.,0.,1.]]) 
	Cj = Lj*Cz
	Ej = Lj*Dz
	aux22 = np.matrix([[F_max**2]]) # 20 %
	aux12 = cp.bmat([[Cj*X+Ej*C_h, Cj+Ej*D_h*C]])
	aux11 = cp.bmat([[X, np.eye(nx)],[np.eye(nx), Y]])
	LMI = cp.bmat([[aux11,aux12.T],[aux12,aux22]])
	Cons10 = LMI >> 0
	# w(1), for both voltage and frequency
	aux11 = cp.bmat([[X, np.eye(nx)],[np.eye(nx), Y]]) 
	aux12 = cp.bmat([[A*X + B*C_h, A+B*D_h*C],[A_h, Y*A+B_h*C]]) 
	aux22 = aux11
	LMI = cp.bmat([[aux11,aux12],[aux12.T,aux22]])
	Cons11 = LMI >> 0
	# 
	Lj = 1.
	Rj = 1.
	Dz = np.matrix([[0.,0.,1.]]) 
	Dzw = np.matrix([[0.,0.,1.]]) 
	Dw = np.matrix([[1.,1.,1.,1.,1.,1.]]).T
	Ej = Lj*Dz
	Dj = Lj*Dzw*Rj
	Fj = Dw*Rj
	#Cons12 = Dj + Ej*D_h*Fj == 0
	####################################################
 	#consts += [Cons9,Cons11]
	
	# constraint initial condition
	for i in xrange(y_nom.shape[1]):		
	#for i in xrange(1):		
		#------- w(2). for Voltage, something wrong considers the controller on steady state condition
		Rj = 1
		Dw = np.zeros((6,1))
		Fj = Dw*Rj
		#Bw = np.matrix([[x_nom[0,i],0.],[0.,x_nom[1,i]],[0.,0.] ])#*(np.sqrt(x_nom[0,i]**2+x_nom[1,i]**2) )
		Bj = x_nom[:,i]
		#Bj = np.matrix([[x_nom[0,i]],[x_nom[1,i]],[x_nom[2,i]] ])#*(np.sqrt(x_nom[0,i]**2+x_nom[1,i]**2) )
		aux11 = cp.bmat([[X, np.eye(nx)],[np.eye(nx), Y]])
		#aux12 = cp.bmat([[Bj+B*D_h*Fj],[Y*Bj+B_h*Fj]]).T
		aux12 = cp.bmat([[Bj],[Y*Bj]])
		aux22 = np.matrix([[V_max**2]]) # 20 %
		aux_LMI =  cp.bmat([[aux22, aux12.T],[aux12, aux11]])
		#x0 = x_nom[:,i]
		#aux_LMI = cp.bmat([[ np.eye(1), x0.T] , [x0, F]])
		aux_cons = aux_LMI >> 0
		#consts += [aux_cons]
		#------- w(2). for Frequency, something wrong considers the controller on steady state condition
		Rj = 1
		Dw = np.zeros((6,1))
		Fj = Dw*Rj
		#Bw = np.matrix([[x_nom[0,i],0.],[0.,x_nom[1,i]],[0.,0.] ])#*(np.sqrt(x_nom[0,i]**2+x_nom[1,i]**2) )
		Bj = x_nom[:,i]
		#Bj = np.matrix([[x_nom[0,i]],[x_nom[1,i]],[x_nom[2,i]] ])#*(np.sqrt(x_nom[0,i]**2+x_nom[1,i]**2) )
		aux11 = cp.bmat([[X, np.eye(nx)],[np.eye(nx), Y]])
		#aux12 = cp.bmat([[Bj+B*D_h*Fj],[Y*Bj+B_h*Fj]]).T
		aux12 = cp.bmat([[Bj],[Y*Bj]])
		aux22 = np.matrix([[F_max**2]]) # 20 %
		aux_LMI =  cp.bmat([[aux22, aux12.T],[aux12, aux11]])
		#x0 = x_nom[:,i]
		#aux_LMI = cp.bmat([[ np.eye(1), x0.T] , [x0, F]])
		aux_cons = aux_LMI >> 0
		#consts += [aux_cons]

	#}}}

	#--------------------------------------------#
	#------- SOLVING THE OPTMIZATION PROBLEM-----#
	#{{{
	optprob = cp.Problem(1.*cp.Minimize(LMI4), constraints=consts)
	#optprob = cp.Problem(cp.Minimize(0.), constraints=consts)
	print("prob is DCP:", optprob.is_dcp())
	result = optprob.solve(solver=cp.CVXOPT,verbose=True, max_iters=10000, abstol=1.e-7, reltol=1.e-7,feastol=1.e-7,refinement = 15,kktsolver='robust')
	#result = optprob.solve(solver=cp.GUROBI)#,verbose=True, max_iters=10000, abstol=1.e-6, reltol=1.e-6,feastol=1.e-6,refinement = 50,kktsolver='robust')
	#result = optprob.solve(solver=cp.MOSEK,verbose=True,warm_start=True, mosek_params = {mosek.dparam.optimizer_max_time:  100.0,
        #                            mosek.iparam.intpnt_solve_form:   mosek.solveform.primal, mosek.dparam.basis_rel_tol_s: 1e-12, mosek.dparam.basis_tol_x: 1e-9 , mosek.dparam.ana_sol_infeas_tol : 1.e-8 })
	print '#------ LMI-SOLUTION-CVXPY(solver) ------#'
	print 'Dinamic output feedback'
	X_LMI = np.matrix(X.value)
	Y_LMI = np.matrix(Y.value)
	Bh_LMI = np.matrix(B_h.value)
	Ah_LMI = np.matrix(A_h.value)
	Dh_LMI = np.matrix(D_h.value)
	Ch_LMI = np.matrix(C_h.value)
	#----------- Solving dynamic output feedback
	MNT = np.asmatrix(np.eye(nx)) - X_LMI*Y_LMI
	u,s,vT = svd(MNT)
	M = np.asmatrix(u)*sqrtm(np.asmatrix(np.diag(s)) )
	N = np.asmatrix(vT).T*sqrtm(np.asmatrix(np.diag(s)) )
	#-----------------
	Dk = Dh_LMI
	Ck = (Ch_LMI - Dk*C*X_LMI)*(M.I).T
	Bk = N.I*(Bh_LMI - Y_LMI*B*Dk)
	Ak = N.I*(Ah_LMI - N*Bk*C*X_LMI - Y_LMI*B*Ck*M.T -Y_LMI*A*X_LMI - Y_LMI*B*Dk*C*X_LMI)*(M.I).T
	#}}}
	#
	#print Dc
	# Solving for operating point 
	Case = 1 # 0: voltage control ,1: output io current control with voltage reference
	#--- solving the steady state condition to include the voltage reference
	if Case == 0:
		#{{{
		Z0 = np.matrix([[0.,0.,1.,0.,0.,0.,0.],[0.,0.,0.,1.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.]])
		D0z = np.matrix([[0.,0.,0.],[0.,0.,0.],[0.,0.,1.]])
		aux1 = np.append(As+Bs*Dc*Cs, Bs*Cc,axis=1)
		aux2 = np.append(Bc*Cs, Ac,axis=1)
		A_aux = np.append(aux1,aux2,axis=0)
		B_aux = np.append(Bs, np.zeros((Ac.shape[0],Bs.shape[1])) , axis=0)
		P_aux = np.append(Ps, np.zeros((Ac.shape[0],Bs.shape[1])) , axis=0)
		#C_aux = np.append(Z0, np.zeros((Z0.shape[0],Ac.shape[0])), axis=1)
		C_aux = np.append(Z0+D0z*Dc*Cs, D0z*Cc, axis=1)
		D_aux = D0z
		A_bar = np.asmatrix(A_aux)
		B_bar = np.asmatrix(B_aux)
		C_bar = np.asmatrix(C_aux)
		D_bar = np.asmatrix(D_aux)
		P_bar = np.asmatrix(P_aux)
		#
		aux = C_bar*( (np.eye(A_bar.shape[0])-A_bar).I )*P_bar 
		Hcl1 = np.eye(aux.shape[0])-aux
		Hcl2 = C_bar*( (np.eye(A_bar.shape[0])-A_bar).I )*B_bar +D_bar
		Acl = Hcl2.I*Hcl1
		# --- Discrete System ----#
		(As_d, Bs_d, Cs_d, Ds_d, Ps_d) = sys.SS_dis
		
		np.savez('data_dsg_LQGLTR', Ac = Ac , Bc=Bc, Cc=Cc, Dc=Dc, As = As_d, Bs = Bs_d, Cs = Cs_d, Ps = Ps_d, Dw =Ds_d , dtd=sys.dtd, Acl=Acl, Case=Case)
		#}}}

	#
	if Case == 1:
		#{{{
		#Z0 = np.matrix([[0.,0.,1.,0.,0.,0.,0.],[0.,0.,0.,1.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.]])
		# For Current control 
		Z0 = np.matrix([[0.,0.,0.,0.,1.,0.,0.],[0.,0.,0.,0.,0.,1.,0.],[0.,0.,1.,0.,0.,0.,0.],[0.,0.,0.,1.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.]])
		D0z = np.matrix([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.],[0.,0.,0.],[0.,0.,1.]])
		##
		aux1 = np.append(A+B*Dk*C, B*Ck,axis=1)
		aux2 = np.append(Bk*C, Ak,axis=1)
		A_aux = np.append(aux1,aux2,axis=0)
		# 
		aux1 = np.append(-B*Dk, B, axis=1)
		aux2 = np.append(-Bk, np.zeros((Bk.shape[0],B.shape[1])), axis=1)
		B_aux = np.append(aux1,aux2 , axis=0)
		#
		P_aux = np.append(P, np.zeros((Ak.shape[0],B.shape[1])) , axis=0)
		#C_aux = np.append(Z0, np.zeros((Z0.shape[0],Ac.shape[0])), axis=1)
		C_aux = np.append(Z0+D0z*Dk*C, D0z*Ck, axis=1)
		D_aux = np.append(-D0z*Dk,D0z,axis=1)
		#
		A_bar = np.asmatrix(A_aux)
		B_bar = np.asmatrix(B_aux)
		C_bar = np.asmatrix(C_aux)
		D_bar = np.asmatrix(D_aux)
		P_bar = np.asmatrix(P_aux)
		#
		aux1 = C_bar*( (np.eye(A_bar.shape[0])-A_bar).I )*B_bar + D_bar
		aux2 = C_bar*( (np.eye(A_bar.shape[0])-A_bar).I )*P_bar 
		aux3 = np.eye(Z0.shape[0])- np.append(np.zeros((aux2.shape[0],2)),aux2,axis=1)
		Acl = aux1.I*aux3
		# [y0,u0].T = Acl*z_ref
		# --- Discrete System ----#
		#(As_d, Bs_d, Cs_d, Ds_d, Ps_d) = sys.SS_dis
		# --- Regulation Vsd, ws----#
		y_nom = Model_Sys.y_nom
		Cz = np.matrix([[0.,0.,1.,0.,0.,0.,0.],[0.,0.,0.,1.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.]])
		Dz = np.matrix([[0.,0.,0.],[0.,0.,0.],[0.,0.,1.]])
		d_nom = np.matrix([[Model_Sys.Vb,0.,Model_Sys.wb]]).T
		Ayx = (np.eye(A.shape[0]) - A - B*Dk*C).I*B*Ck*(np.eye(Ak.shape[0])-Ak).I*Bk #+ Ps*d_nom
		z_nom = np.matrix(np.zeros((3,y_nom.shape[1])) )
		u_nom = np.matrix(np.zeros((3,y_nom.shape[1])) )
		K_xy = Cz*Ayx + Dz*Dk + Dz*Ck*(np.eye(Ak.shape[0])-Ak).I*Bk
		K_uy = Ck*(np.eye(Ak.shape[0])-Ak).I*Bk + Dk
		#print K_xy
		for i in xrange(y_nom.shape[1]):
		#	y_nom = xnom[0:6,i]
			z_ref = np.matrix([[0.,0.,sys.Vb,0.,sys.wb]]).T
			z0 = (Acl*z_ref)
			y0 = z0[0:6,:]
			u0 = z0[6:9:]
			#print y_nom
			aux_nom = K_uy*(y_nom[:,i]-y0)+u0
			# to the input 
			u_nom[:,i] = np.matrix([[aux_nom[0,0]],[aux_nom[1,0]],[aux_nom[2,0]]])
			# to the voltage vs
			io_ = y_nom[4,i] + 1J*y_nom[5,i]
			vt_ = aux_nom[0,0] + 1J*aux_nom[1,0]
			wt_ = aux_nom[2,0]
			vs_ = (vt_ - (Model_Sys.Rf+1J*wt_*Model_Sys.Lf)*io_)/(1.+(Model_Sys.Rf+1J*wt_*Model_Sys.Lf)*1J*wt_*Model_Sys.Cf)
			#print vs_
			#z_nom[:,i] = np.matrix([[aux_nom[1,0]],[aux_nom[2,0]]])
			# obtaining z from u 
			z_nom[:,i] = np.matrix([[vs_.real],[vs_.imag],[aux_nom[2,0]]])
			# direct evaluation of z
			#z_nom[:,i] = K_xy*(y_nom[:,i]-y0)
	#}}}

	#--------------------------------------------#
	#print z_nom[0,:]
	return Ak,Bk,Ck,Dk,z_nom
	#}}}

#------ LMI: H2
def Simp_Disc_H2_Out_FB5(Sys,dtd,y_nom,x_nom):
	#{{{
	consts = []
 	A = Sys[0]
	B = Sys[1]
	C = Sys[2]
 	P = Sys[3]
 	Dw = Sys[4]
	C_aux = np.matrix([[1.,0.,0.,0.,0.,0.,0.],[0.,1.,0.,0.,0.,0.,0.],[0.,0.,1.,0.,0.,0.,0.],[0.,0.,0.,1.,0.,0.,0.],[0.,0,0.,0.,1.,0.,0.],[0.,0.,0.,0.,0.,1.,0.],[0.,0.,0.,0.,0.,0.,0.]])
	nx = A.shape[0]
	nu = B.shape[1]
	ny = C.shape[0]
	# LMIs for H2
	#{{{
	#Dzw = np.zeros((Cz.shape[0],B.shape[1]))
	# Problems variables definition 
	X = cp.Variable((nx,nx),symmetric=True)
	Y = cp.Variable((nx,nx),symmetric=True)
	A_h = cp.Variable((nx,nx) )
	B_h = cp.Variable((nx,ny) )
	C_h = cp.Variable((nu,nx) )
	D_h = cp.Variable((nu,ny) )
	nuq = cp.Variable(1)
	# LMI 1- Stability of the closed loop system, lyapunov H2
	def LMI_H2(sys):
		#{{{
		Bj=sys[0]
		Cj=sys[1]
		Dj=sys[2]
		Ej=sys[3]
		Fj=sys[4]
		Q = sys[5]
		aux11 = -cp.bmat([[X, np.eye(nx)],[np.eye(nx), Y]]) 
		aux12 = cp.bmat([[Bj + B*D_h*Fj],[Y*Bj + B_h*Fj]]) 
		aux13 = cp.bmat([[A*X + B*C_h, A+B*D_h*C],[A_h, Y*A+B_h*C]]) 
		aux22 = -np.eye(aux12.shape[1])
		aux23 = np.zeros((aux22.shape[0],aux13.shape[1]))
		aux33 = aux11
		aux = cp.bmat([[aux11,aux12,aux13],[aux12.T,aux22,aux23],[aux13.T,aux23.T,aux33]])
		LMI1 = aux << 0
		# LMI 2- H performance associated transformation
		aux11 = Q
		aux12 = cp.bmat([[Cj*X+Ej*C_h, Cj+Ej*D_h*C]])
		aux22 = cp.bmat([[X, np.eye(nx)],[np.eye(nx), Y]])
		aux = cp.bmat([[aux11,aux12],[aux12.T,aux22]])
		LMI2 = aux >> 0
		# LMI 3- D closed loop minimization, 
		aux = Ej*D_h*Fj + Dj
		LMI3 = aux==0
		return (LMI1,LMI2,LMI3)	
		#}}}
	#--- Constraints ----

	Q = C.T*C#C_aux.T*C_aux ## xT*Q*x
	#R =1.e7*np.matrix([[1./(377.**2),0.,0],[0.,1./(377.**2),0],[0.,0.,0e-1]])
	#R =1.e6*np.matrix([[1.,0.,0],[0.,1.,0],[0.,0.,2e-7]])
	R =1.e6*np.matrix([[1.,0.,0],[0.,1.,0],[0.,0.,1e-7]])
	#
	R_sqrt = sqrtm(R)
	Q_sqrt = sqrtm(Q)
	Cz = np.append(Q_sqrt,np.zeros((nu,nx)),axis=0)
	Dz = np.append(np.zeros((nx,nu)),R_sqrt,axis=0)
	Bw = P
	Dw = 1.e0*np.asmatrix(Dw)#np.asmatrix(np.zeros((ny,nu)) )
	Dzw = np.asmatrix(np.zeros((Cz.shape[0],P.shape[1])) )
	#======= [iod,ioq]_2 /[v,e]_2
	# w = Rj*wj , zj = Lj*z, wj = [],zj =[vtd,vtq,wt]
	Rj = 1.e0*np.append(np.zeros((12,3)),np.append(np.zeros((3,9)),np.eye(9),axis=0),axis=1 )
	Lj = 1.*np.matrix([[0.,0.,0.,0.,0.,0.,0.,1.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,1.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,1.]],dtype=float)
	# w = Rj*wj , zj = Lj*z, wj = [],zj =[vtd,vtq]
	#Rj = 1.e0*np.append(np.zeros((12,3)),np.append(np.zeros((3,9)),np.eye(9),axis=0),axis=1 )
	#Lj = 1.*np.matrix([[0.,0.,0.,0.,0.,0.,0.,1.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,1.,0.]])
	Bj = Bw*Rj
	Cj = Lj*Cz
	Dj = Lj*Dzw*Rj
	Ej = Lj*Dz
	Fj = Dw*Rj
	q = cp.Variable(Cj.shape[0])
	Q_var = cp.diag( q)
	sys = (Bj,Cj,Dj,Ej,Fj,Q_var)
	(Cons1, Cons2 ,Cons3) = LMI_H2(sys)
	#
	# LMI 4- Trace  minimization, 
	LMI4 = cp.trace(Q_var)
	#}}}
	consts += [Cons1,Cons2]
	# LMIs for pole constraints
	#{{{
	Pi1T_P_Pi1 = cp.bmat([[X, np.eye(nx)],[np.eye(nx), Y]])
	Pi1T_PA_Pi1 = cp.bmat([[A*X + B*C_h, A+B*D_h*C],[A_h, Y*A+B_h*C]]) 
	Matrix_LMI = (Pi1T_P_Pi1, Pi1T_PA_Pi1)
	# for speed of the poles
	alfa = 20. # s+alfa  (poles) 
	#nd = 2*nx # dimension of the space where the poles will be
	R11 = np.matrix([[-np.exp(-2*alfa*dtd), 0.],[0.,-1.]])#-np.eye(nd)*np.exp(-2.*alfa*dtd)
	R12 = np.matrix([[0.,1.],[0.,0.]])#np.zeros((nd,nd))
	R22 = np.matrix([[0.,0.],[0.,0.]])#np.eye(nd)
	M_DSGN = (R11,R12,R22)
	LMI5 = Pole_Placement_Discrete(Matrix_LMI, M_DSGN)
	Cons5 = LMI5 << 0
	# damping constraints 
	xi_min = 0.6
	thet_ang = np.arccos(xi_min)
	case = 5 # approximation by: '0' for inner circle, '1' for inner ellipse, '2' half right-plane and ellipse, '3' ellipse cone, '4' ellipse+cone+right half-plane, '5' RPP+speed
	if case == 0:
		# {{{for a circle approximation
		x_m = np.exp(-thet_ang/np.tan(thet_ang))*np.cos(-thet_ang)
		y_m = -np.exp(-thet_ang/np.tan(thet_ang))*np.sin(-thet_ang)
		x_0 = -np.exp(-np.pi/np.tan(thet_ang))
		ak = x_m-x_0
		bk = y_m
		r_k = min(ak,bk)
		# LMI
		R11 = np.matrix([[x_m**2 - r_k**2]])
		R12 = -np.matrix([[x_m]])
		R22 = np.matrix([[1.]])
		M_DSGN = (R11,R12,R22)
		LMI6 = Pole_Placement_Discrete(Matrix_LMI, M_DSGN)
		#------- Draw constraints -------#
		Damp = (xi_min, dtd)
		Circ_Approx = (x_m,r_k)
		Constraints_Draw(Damp,Circ_Approx,'Circle')
		#}}}
		#
	if case == 1:
		#{{{ for a ellipse approximation
		x_m = np.exp(-thet_ang/np.tan(thet_ang))*np.cos(-thet_ang)
		y_m = -np.exp(-thet_ang/np.tan(thet_ang))*np.sin(-thet_ang)
		x_0 = -np.exp(-np.pi/np.tan(thet_ang))
		ak = x_m-x_0
		bk = y_m
		# LMI
		R11 = np.matrix([[-1.,-x_m/ak],[-x_m/ak,-1.]])
		R12 = np.matrix([[0.,(1./ak)*0.5 - (1./bk)*0.5],[(1./ak)*0.5 + (1./bk)*0.5,0.]])
		R22 = np.matrix([[0.,0.],[0.,0.]])
		M_DSGN = (R11,R12,R22)
		LMI6 = Pole_Placement_Discrete(Matrix_LMI, M_DSGN)
		#------- Draw constraints -------#
		Damp = (xi_min, dtd)
		Ellip_Approx = (ak,bk,x_m)
		Constraints_Draw(Damp,Ellip_Approx,'Ellipse')
		#}}}
		#
	if case == 2:
		#{{{ for a half plane and ellipse approximation
		x_m = np.exp(-thet_ang/np.tan(thet_ang))*np.cos(-thet_ang)
		y_m = -np.exp(-thet_ang/np.tan(thet_ang))*np.sin(-thet_ang)
		x_0 = -np.exp(-np.pi/np.tan(thet_ang))
		y_3 = np.exp(-np.pi/(2.*np.tan(thet_ang)) ) 
		ak = x_m*y_m/np.sqrt(y_m**2-y_3**2)
		bk = y_m
		# LMI
		R11 = np.matrix([[0.,0.,0.],[0.,-1.,-x_m/ak],[0.,-x_m/ak,-1.]])
		R12 = np.matrix([[-1.,0.,0.],[0.,0.,(1./ak)*0.5 - (1./bk)*0.5],[0.,(1./ak)*0.5 + (1./bk)*0.5,0.]])
		R22 = np.matrix([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]])
		M_DSGN = (R11,R12,R22)
		LMI6 = Pole_Placement_Discrete(Matrix_LMI, M_DSGN)
		#------- Draw constraints -------#
		Damp = (xi_min, dtd)
		Ellip_Approx = (ak,bk,x_m,y_3,x_0)
		Constraints_Draw(Damp,Ellip_Approx,'HP_Ellipse')
		#}}}
		#
	if case == 3:
		#{{{ for a ellipse-cone approximation
		x_m = np.exp(-thet_ang/np.tan(thet_ang))*np.cos(-thet_ang)
		y_m = -np.exp(-thet_ang/np.tan(thet_ang))*np.sin(-thet_ang)
		x_0 = -np.exp(-np.pi/np.tan(thet_ang))
		y_3 = np.exp(-np.pi/(2.*np.tan(thet_ang)) ) 
		a_e = x_m*y_m/np.sqrt(y_m**2-y_3**2)
		b_e = y_m
		#------ ellipse centered inner cordiode
		xe = 0.5 # interseccion cardiode and cone
		ye = b_e*np.sin( np.arccos( (xe-x_m)/a_e ) ) # interseccion con cardioide
		xse = (1.+x_0)/2. # new center of the ellipse
		ak = (1.-x_0)/2. # new 'a' 
		bk = ye*ak/np.sqrt(ak**2 - (xe-xse)**2) # new 'b'
		# matices
		R11e = np.matrix([[-1.,-xse/ak],[-xse/ak,-1.]])
		R12e = np.matrix([[0.,(1./ak)*0.5 - (1./bk)*0.5],[(1./ak)*0.5 + (1./bk)*0.5,0.]])
		R22e = np.matrix([[0.,0.],[0.,0.]])
		#-------- cone with vetex at (xv,0)
		gama = np.arctan(ye/(1.-xe))
		xv = 1.
		R11v = np.matrix([[-xv*np.sin(gama)*2, 0.],[0., -xv*np.sin(gama)*2]])
		R12v = np.matrix([[np.sin(gama), np.cos(gama)],[-np.cos(gama), np.sin(gama)]])
		R22v = np.matrix([[0.,0.],[0.,0.]])
		#--------- Intersection ellipse and cone
		Z = np.zeros((2,2))
		aux1 = np.append(R11e,Z,axis=1)
		aux2 = np.append(Z, R11v,axis=1)
		R11 = np.append(aux1,aux2, axis=0)
		aux1 = np.append(R12e,Z,axis=1)
		aux2 = np.append(Z, R12v,axis=1)
		R12 = np.append(aux1,aux2, axis=0)
		aux1 = np.append(R22e,Z,axis=1)
		aux2 = np.append(Z, R22v,axis=1)
		R22 = np.append(aux1,aux2, axis=0)
		#-------------------------------
		M_DSGN = (R11,R12,R22)
		LMI6 = Pole_Placement_Discrete(Matrix_LMI, M_DSGN)
		#------- Draw constraints -------#
		Damp = (xi_min, dtd)
		Ellip_Approx = (ak,bk,xse,xe, gama,xv )
		Constraints_Draw(Damp,Ellip_Approx,'Ellipse_Cone')
		#}}}
		#
	if case == 4:
		#{{{ for a ellipse-cone approximation and right half plane
		x_m = np.exp(-thet_ang/np.tan(thet_ang))*np.cos(-thet_ang)
		y_m = -np.exp(-thet_ang/np.tan(thet_ang))*np.sin(-thet_ang)
		x_0 = -np.exp(-np.pi/np.tan(thet_ang))
		y_3 = np.exp(-np.pi/(2.*np.tan(thet_ang)) ) 
		a_e = x_m*y_m/np.sqrt(y_m**2-y_3**2)
		b_e = y_m
		#
		k = np.tan(thet_ang)
		t = np.arange(-np.pi/k,0, 0.01)
		#------ ellipse centered inner cordiode
		xe = 0.6 # interseccion cardiode and cone
		for i in xrange(t.shape[0]):
			error = xe-np.exp(t[i])*np.cos(k*t[i])
			if error < 0.01:
				t_aux = t[i]
				break
		ye = abs(np.exp(t_aux)*np.sin(k*t_aux))
		#print ye
		xse = (1.+x_0)/2. # new center of the ellipse
		ak = (1.-x_0)/2. # new 'a' 
		bk = ye*ak/np.sqrt(ak**2 - (xe-xse)**2) # new 'b'
		# matices
		R11e = np.matrix([[-1.,-xse/ak],[-xse/ak,-1.]])
		R12e = np.matrix([[0.,(1./ak)*0.5 - (1./bk)*0.5],[(1./ak)*0.5 + (1./bk)*0.5,0.]])
		R22e = np.matrix([[0.,0.],[0.,0.]])
		#-------- cone with vetex at (xv,0)
		gama = np.arctan(ye/(1.-xe))
		xv = 1.
		R11v = np.matrix([[-xv*np.sin(gama)*2, 0.],[0., -xv*np.sin(gama)*2]])
		R12v = np.matrix([[np.sin(gama), np.cos(gama)],[-np.cos(gama), np.sin(gama)]])
		R22v = np.matrix([[0.,0.],[0.,0.]])
		#---------- right half plane
		R11h = np.matrix([[0.]])
		R12h = np.matrix([[-1.]])
		R22h = np.matrix([[0.]])
		#--------- Intersection ellipse and cone
		Z = np.zeros((2,2))
		Z2 = np.zeros((2,1))
		aux_a = np.append(R11e,Z,axis=1)
		aux1 = np.append(aux_a,Z2,axis=1)
		aux_b = np.append(Z, R11v,axis=1)
		aux2 = np.append(aux_b, Z2,axis=1)
		aux_c = np.append(Z2.T, Z2.T,axis=1)
		aux3 = np.append(aux_c, R11h,axis=1)
		aux_R = np.append(aux1,aux2, axis=0)
		R11 = np.append(aux_R,aux3, axis=0)
		#
		aux_a = np.append(R12e,Z,axis=1)
		aux1 = np.append(aux_a,Z2,axis=1)
		aux_b = np.append(Z, R12v,axis=1)
		aux2 = np.append(aux_b, Z2,axis=1)
		aux_c = np.append(Z2.T, Z2.T,axis=1)
		aux3 = np.append(aux_c, R12h,axis=1)
		aux_R = np.append(aux1,aux2, axis=0)
		R12 = np.append(aux_R,aux3, axis=0)
		#
		aux_a = np.append(R22e,Z,axis=1)
		aux1 = np.append(aux_a,Z2,axis=1)
		aux_b = np.append(Z, R22v,axis=1)
		aux2 = np.append(aux_b, Z2,axis=1)
		aux_c = np.append(Z2.T, Z2.T,axis=1)
		aux3 = np.append(aux_c, R22h,axis=1)
		aux_R = np.append(aux1,aux2, axis=0)
		R22 = np.append(aux_R,aux3, axis=0)
		#-------------------------------
		M_DSGN = (R11,R12,R22)
		LMI6 = Pole_Placement_Discrete(Matrix_LMI, M_DSGN)
		#------- Draw constraints -------#
		Damp = (xi_min, dtd)
		Ellip_Approx = (ak,bk,xse,xe, gama,xv )
		Constraints_Draw(Damp,Ellip_Approx,'Ellipse_Cone_RHP')
		#}}}
		#
	if case == 5:
		#{{{ right half plane
		#---------- right half plane
		Alfa_L = 0.
		R11h = np.matrix([[0.]])
		R12h = np.matrix([[-1.]])
		R22h = np.matrix([[0.]])
		#--------- Intersection ellipse and cone
		R11 = R11h
		R12 = R12h
		R22 = R22h
		#-------------------------------
		M_DSGN = (R11,R12,R22)
		LMI6 = Pole_Placement_Discrete(Matrix_LMI, M_DSGN)
		#}}}
	Cons6 = LMI6 << 0
	#}}}

	consts += [Cons5]
	
	# LMI for H_inf performance 
	#{{{
	def Hinf_LMI(aux1,aux2):
		#{{{
		Lj = aux1[0]
		Rj = aux1[1]
		gamma = aux1[2]
		#
		Cz = aux2[0]
		Dz = aux2[1]
		Bw = aux2[2]
		Dw = aux2[3]
		Dzw = aux2[4]
		#
		Cj = Lj*Cz
		Ej = Lj*Dz
		Bj = Bw*Rj
		Fj = Dw*Rj
		Dj = Lj*Dzw*Rj
		#
		Pi1T_P_Pi1 = cp.bmat([[X, np.eye(nx)],[np.eye(nx), Y]])
		Pi1T_PA_Pi1 = cp.bmat([[A*X + B*C_h, A+B*D_h*C],[A_h, Y*A+B_h*C]]) 
		Cj_Pi1 = cp.bmat([[Cj*X+Ej*C_h, Cj+Ej*D_h*C]])
		PiT_P_Bj = cp.bmat([[Bj+B*D_h*Fj],[ Y*Bj+B_h*Fj]])
		#
		aux11 = -Pi1T_P_Pi1
		aux13 = Pi1T_PA_Pi1.T
		aux14 = Cj_Pi1.T
		aux23 = PiT_P_Bj.T
		aux24 = (Dj+Ej*D_h*Fj).T
		aux33 = aux11
		aux44 = -np.eye(aux24.shape[0])
		aux12 = np.zeros((aux11.shape[0],aux23.shape[0] ))
		aux22 = -np.eye(aux23.shape[0])*gamma**2
		aux34 = np.zeros((aux33.shape[0],aux24.shape[1]))
		#
		LMI7 = cp.bmat([[aux11,aux12,aux13,aux14],[aux12.T,aux22,aux23,aux24],[aux13.T,aux23.T,aux33, aux34],[aux14.T,aux24.T,aux34.T,aux44]])
		Cons_a = LMI7 << 0
		Cons_b = Dj+Ej*D_h*Fj == 0
		#return Cons_a, Cons_b
		return Cons_a
		#}}}
	
	# Different variables to optimize
	case = 5
	if case == 0: # currents vsd, vsq and omega_t
		Cz = np.matrix([[0.,0.,1.,0.,0.,0.,0.],[0.,0.,0.,1.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.]]) 
		Dz = np.matrix([[0.,0.,0.],[0.,0.,0.],[0.,0.,1.]]) 
		Bw = P
		Dw = np.asmatrix(np.zeros((6,3)) )
		Dzw = np.asmatrix(np.zeros((3,3)) )
	if case == 1: # currents iod
		Cz = np.matrix([[0.,0.,0.,0.,1.,0.,0.],[0.,0.,0.,0.,0.,1.,0.],[0.,0.,0.,0.,0.,0.,0.]]) 
		Dz = np.matrix([[0.,0.,0.],[0.,0.,0.],[0.,0.,1.]]) 
		Bw = P
		Dw = np.asmatrix(np.zeros((6,3)) )
		Dzw = np.asmatrix(np.zeros((3,3)) )
	if case == 2: # omega_t versus noise in iod ioq
		Cz = np.matrix([[0.,0.,0.,0.,0.,0.,0.]]) 
		Dz = np.matrix([[0.,0.,1.]]) 
		Dzw = np.asmatrix(np.zeros((1,2)) )
		Bw = np.zeros((7,2))
		Dw = np.matrix([[0.,0.],[0.,0.],[0.,0.],[0.,0.],[1.,0.],[0.,1.]])
	if case == 3: # omega_t versus noise in iod ioq in the controlled variables
		Cz = np.matrix([[0.,0.,0.,1.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.]]) 
		Dz = np.matrix([[0.,0.,0.],[0.,0.,1.]]) 
		Dzw = np.zeros((2,2))#np.matrix([[1.,1.]])
		Bw = np.zeros((7,2))
		Dw = np.matrix([[0.,0.],[0.,0.],[0.,0.],[0.,0.],[1.,0.],[0.,1.]])
	if case == 4: # omega_t versus noise vg,wg
		Cz = np.matrix([[0.,0.,0.,0.,0.,0.,0.]]) 
		Dz = np.matrix([[0.,0.,1.]]) 
		Dzw = np.zeros((1,3))#np.matrix([[1.,1.]])
		Bw = P
		Dw = np.zeros((6,3))#np.matrix([[0.,0.],[0.,0.],[0.,0.],[0.,0.],[1.,0.],[0.,1.]])
	if case == 5: # [vsd,vsq,omega_t] versus noise vg,wg
		Cz = np.matrix([[0.,0.,0.,0.,1.,0.,0.],[0.,0.,0.,0.,0.,1.,0.],[0.,0.,1.,0.,0.,0.,0.],[0.,0.,0.,1.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.]]) 
		Dz = np.matrix([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.],[0.,0.,0.],[0.,0.,1.]]) 
		Bw = P#np.append(P,np.zeros((7,2)),axis=1)
		Dw = Dw#np.append(np.zeros((6,3)),np.append(np.zeros((4,2)),np.eye(2),axis=0),axis=1)
		Dzw = Cz[:,0:6]*Dw#np.zeros((3,12))#np.matrix([[1.,1.]])

#np.matrix([[0.,0.],[0.,0.],[0.,0.],[0.,0.],[1.,0.],[0.,1.]])
	# w = Rj*wj, zj = Lj*z, wj = [dw,du,dy]
	# LMI for H_inf performance for wt / ioq
	gamma = 0.1
	Lj = np.matrix([[0.,0.,0.,0.,1.]])
	Rj = np.matrix([[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.]]).T
	aux1 = (Lj,Rj,gamma)
	aux2 = (Cz,Dz,Bw,Dw,Dzw)
	#=====================================
	Cons_a = Hinf_LMI(aux1,aux2)
	consts += [Cons_a]
	#====================================
	# LMI for H_inf performance  for wt / iod
	gamma = 0.1
	Lj = np.matrix([[0.,0.,0.,0.,1.]])
	#Rj = np.matrix([[0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.]]).T
	Rj = np.matrix([[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.]]).T
	aux1 = (Lj,Rj,gamma)
	aux2 = (Cz,Dz,Bw,Dw,Dzw)
	# ------------------------
	Cons_a= Hinf_LMI(aux1,aux2)
	consts += [Cons_a]
	#====================================
	# LMI for H_inf performance for voltage wt / wg
	gamma = 0.01
	#Lj = np.matrix([[1.,0.,0.],[0.,1.,0.]])
	Lj = np.matrix([[0.,0.,0.,0.,1.]])
	#Rj = np.matrix([[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,10000.]]).T
	Rj = np.matrix([[0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.]]).T
	aux1 = (Lj,Rj,gamma)
	aux2 = (Cz,Dz,Bw,Dw,Dzw)
	#=====================================
	Cons_a = Hinf_LMI(aux1,aux2)
	consts += [Cons_a]
	#====================================
	# LMI for H_inf performance for voltage vs / wg
	gamma = 0.0001
	#Lj = np.matrix([[1.,0.,0.],[0.,1.,0.]])
	Lj = np.matrix([[0.,1.,0.]])
	#Rj = np.matrix([[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,10000.]]).T
	Rj = np.matrix([[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.]]).T
	aux1 = (Lj,Rj,gamma)
	aux2 = (Cz,Dz,Bw,Dw,Dzw)
	#=====================================
	#Cons_a= Hinf_LMI(aux1,aux2)
	#consts += [Cons_a]
	#====================================
	

	#}}}

	####################################################
	
	# Peak to Peak
	#{{{
	# w(3), for voltage
	V_max = 520.*0.1
	Lj = 1.
	Cz = np.matrix([[0.,0.,1.,0.,0.,0.,0.],[0.,0.,0.,1.,0.,0.,0.]]) 
	Dz = np.matrix([[0.,0.,0.],[0.,0.,0.]]) 
	Cj = Lj*Cz
	Ej = Lj*Dz
	aux22 = np.matrix([[V_max**2,0.],[0.,V_max**2]]) # 20 %
	aux12 = cp.bmat([[Cj*X+Ej*C_h, Cj+Ej*D_h*C]])
	aux11 = cp.bmat([[X, np.eye(nx)],[np.eye(nx), Y]])
	LMI = cp.bmat([[aux11,aux12.T],[aux12,aux22]])
	Cons9 = LMI >> 0
	# === w(3), for frequency === #
	F_max = 1.*1.*np.pi
	Lj = 1.
	Cz = np.matrix([[0.,0.,0.,0.,0.,0.,0.]]) 
	Dz = np.matrix([[0.,0.,1.]]) 
	Cj = Lj*Cz
	Ej = Lj*Dz
	aux22 = np.matrix([[F_max**2]]) # 20 %
	aux12 = cp.bmat([[Cj*X+Ej*C_h, Cj+Ej*D_h*C]])
	aux11 = cp.bmat([[X, np.eye(nx)],[np.eye(nx), Y]])
	LMI = cp.bmat([[aux11,aux12.T],[aux12,aux22]])
	Cons10 = LMI >> 0
	# w(1), for both voltage and frequency
	aux11 = cp.bmat([[X, np.eye(nx)],[np.eye(nx), Y]]) 
	aux12 = cp.bmat([[A*X + B*C_h, A+B*D_h*C],[A_h, Y*A+B_h*C]]) 
	aux22 = aux11
	LMI = cp.bmat([[aux11,aux12],[aux12.T,aux22]])
	Cons11 = LMI >> 0
	# 
	Lj = 1.
	Rj = 1.
	Dz = np.matrix([[0.,0.,1.]]) 
	Dzw = np.matrix([[0.,0.,1.]]) 
	Dw = np.matrix([[1.,1.,1.,1.,1.,1.]]).T
	Ej = Lj*Dz
	Dj = Lj*Dzw*Rj
	Fj = Dw*Rj
	#Cons12 = Dj + Ej*D_h*Fj == 0
	####################################################
 	#consts += [Cons10,Cons11]
	
	# constraint initial condition
	for i in xrange(y_nom.shape[1]):		
	#for i in xrange(1):		
		#------- w(2). for Voltage, something wrong considers the controller on steady state condition
		Rj = 1
		Dw = np.zeros((6,1))
		Fj = Dw*Rj
		#Bw = np.matrix([[x_nom[0,i],0.],[0.,x_nom[1,i]],[0.,0.] ])#*(np.sqrt(x_nom[0,i]**2+x_nom[1,i]**2) )
		Bj = x_nom[:,i]
		#Bj = np.matrix([[x_nom[0,i]],[x_nom[1,i]],[x_nom[2,i]] ])#*(np.sqrt(x_nom[0,i]**2+x_nom[1,i]**2) )
		aux11 = cp.bmat([[X, np.eye(nx)],[np.eye(nx), Y]])
		#aux12 = cp.bmat([[Bj+B*D_h*Fj],[Y*Bj+B_h*Fj]]).T
		aux12 = cp.bmat([[Bj],[Y*Bj]])
		aux22 = np.matrix([[V_max**2]]) # 20 %
		aux_LMI =  cp.bmat([[aux22, aux12.T],[aux12, aux11]])
		#x0 = x_nom[:,i]
		#aux_LMI = cp.bmat([[ np.eye(1), x0.T] , [x0, F]])
		aux_cons = aux_LMI >> 0
	#	consts += [aux_cons]
		#------- w(2). for Frequency, something wrong considers the controller on steady state condition
		Rj = 1
		Dw = np.zeros((6,1))
		Fj = Dw*Rj
		#Bw = np.matrix([[x_nom[0,i],0.],[0.,x_nom[1,i]],[0.,0.] ])#*(np.sqrt(x_nom[0,i]**2+x_nom[1,i]**2) )
		Bj = x_nom[:,i]
		#Bj = np.matrix([[x_nom[0,i]],[x_nom[1,i]],[x_nom[2,i]] ])#*(np.sqrt(x_nom[0,i]**2+x_nom[1,i]**2) )
		aux11 = cp.bmat([[X, np.eye(nx)],[np.eye(nx), Y]])
		#aux12 = cp.bmat([[Bj+B*D_h*Fj],[Y*Bj+B_h*Fj]]).T
		aux12 = cp.bmat([[Bj],[Y*Bj]])
		aux22 = np.matrix([[F_max**2]]) # 20 %
		aux_LMI =  cp.bmat([[aux22, aux12.T],[aux12, aux11]])
		#x0 = x_nom[:,i]
		#aux_LMI = cp.bmat([[ np.eye(1), x0.T] , [x0, F]])
		aux_cons = aux_LMI >> 0
		#consts += [aux_cons]

	#}}}

	#--------------------------------------------#
	#------- SOLVING THE OPTMIZATION PROBLEM-----#
	#{{{
	optprob = cp.Problem(1.*cp.Minimize(LMI4), constraints=consts)
	#optprob = cp.Problem(cp.Minimize(0.), constraints=consts)
	print("prob is DCP:", optprob.is_dcp())
	result = optprob.solve(solver=cp.CVXOPT,verbose=True, max_iters=10000, abstol=1.e-7, reltol=1.e-7,feastol=1.e-7,refinement = 20,kktsolver='robust')
	#result = optprob.solve(solver=cp.GUROBI)#,verbose=True, max_iters=10000, abstol=1.e-6, reltol=1.e-6,feastol=1.e-6,refinement = 50,kktsolver='robust')
	#result = optprob.solve(solver=cp.MOSEK,verbose=True,warm_start=True, mosek_params = {mosek.dparam.optimizer_max_time:  100.0,
        #                            mosek.iparam.intpnt_solve_form:   mosek.solveform.primal, mosek.dparam.basis_rel_tol_s: 1e-12, mosek.dparam.basis_tol_x: 1e-9 , mosek.dparam.ana_sol_infeas_tol : 1.e-8 })
	print '#------ LMI-SOLUTION-CVXPY(solver) ------#'
	print 'Dinamic output feedback'
	X_LMI = np.matrix(X.value,dtype=float)
	Y_LMI = np.matrix(Y.value,dtype=float)
	Bh_LMI = np.matrix(B_h.value,dtype=float)
	Ah_LMI = np.matrix(A_h.value,dtype=float)
	Dh_LMI = np.matrix(D_h.value,dtype=float)
	Ch_LMI = np.matrix(C_h.value,dtype=float)
	#----------- Solving dynamic output feedback
	MNT = np.asmatrix(np.eye(nx)) - X_LMI*Y_LMI
	u,s,vT = svd(MNT)
	M = np.asmatrix(u,dtype=float)*sqrtm(np.asmatrix(np.diag(s),dtype=float) )
	N = np.asmatrix(vT,dtype=float).T*sqrtm(np.asmatrix(np.diag(s),dtype=float) )
	#-----------------
	Dk = Dh_LMI
	Ck = (Ch_LMI - Dk*C*X_LMI)*(M.I).T
	Bk = N.I*(Bh_LMI - Y_LMI*B*Dk)
	Ak = N.I*(Ah_LMI - N*Bk*C*X_LMI - Y_LMI*B*Ck*M.T -Y_LMI*A*X_LMI - Y_LMI*B*Dk*C*X_LMI)*(M.I).T
	#}}}
	#
	#print Dc
	# Solving for operating point 
	Case = 2 # 0: voltage control ,1: output io current control with voltage reference
	#--- solving the steady state condition to include the voltage reference
	if Case == 0:
		#{{{
		Z0 = np.matrix([[0.,0.,1.,0.,0.,0.,0.],[0.,0.,0.,1.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.]])
		D0z = np.matrix([[0.,0.,0.],[0.,0.,0.],[0.,0.,1.]])
		aux1 = np.append(As+Bs*Dc*Cs, Bs*Cc,axis=1)
		aux2 = np.append(Bc*Cs, Ac,axis=1)
		A_aux = np.append(aux1,aux2,axis=0)
		B_aux = np.append(Bs, np.zeros((Ac.shape[0],Bs.shape[1])) , axis=0)
		P_aux = np.append(Ps, np.zeros((Ac.shape[0],Bs.shape[1])) , axis=0)
		#C_aux = np.append(Z0, np.zeros((Z0.shape[0],Ac.shape[0])), axis=1)
		C_aux = np.append(Z0+D0z*Dc*Cs, D0z*Cc, axis=1)
		D_aux = D0z
		A_bar = np.asmatrix(A_aux)
		B_bar = np.asmatrix(B_aux)
		C_bar = np.asmatrix(C_aux)
		D_bar = np.asmatrix(D_aux)
		P_bar = np.asmatrix(P_aux)
		#
		aux = C_bar*( (np.eye(A_bar.shape[0])-A_bar).I )*P_bar 
		Hcl1 = np.eye(aux.shape[0])-aux
		Hcl2 = C_bar*( (np.eye(A_bar.shape[0])-A_bar).I )*B_bar +D_bar
		Acl = Hcl2.I*Hcl1
		# --- Discrete System ----#
		(As_d, Bs_d, Cs_d, Ds_d, Ps_d) = sys.SS_dis
		
		np.savez('data_dsg_LQGLTR', Ac = Ac , Bc=Bc, Cc=Cc, Dc=Dc, As = As_d, Bs = Bs_d, Cs = Cs_d, Ps = Ps_d, Dw =Ds_d , dtd=sys.dtd, Acl=Acl, Case=Case)
		#}}}

	#
	if Case == 1: # considers the reference zref = [iod,ioq,vsd,vsq,w]
		#{{{
	# Calculating the matrix for the currents, voltage and frequency references
		#{{{
		(As,Bs,Cs,Ds,Ps,Dwz) = Model_Sys.SS_dis
		#As = np.asmatrix(data['As'])
		#Bs = np.asmatrix(data['Bs'])
		#Cs = np.asmatrix(data['Cs'])
		#Ps = np.asmatrix(data['Ps'])
		Z0 = np.matrix([[0.,0.,0.,0.,1.,0.,0.],[0.,0.,0.,0.,0.,1.,0.],[0.,0.,1.,0.,0.,0.,0.],[0.,0.,0.,1.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.]])
		D0z = np.matrix([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.],[0.,0.,0.],[0.,0.,1.]])
		#
		aux1 = np.append(As+Bs*Dk*Cs, Bs*Ck,axis=1)
		aux2 = np.append(Bk*Cs, Ak,axis=1)
		A_aux = np.append(aux1,aux2,axis=0)
		# 
		aux1 = np.append(-Bs*Dk, Bs, axis=1)
		aux2 = np.append(-Bk, np.zeros((Bk.shape[0],Bs.shape[1])), axis=1)
		B_aux = np.append(aux1,aux2 , axis=0)
		#
		P_aux = np.append(Ps, np.zeros((Ak.shape[0],Bs.shape[1])) , axis=0)
		#C_aux = np.append(Z0, np.zeros((Z0.shape[0],Ac.shape[0])), axis=1)
		C_aux = np.append(Z0+D0z*Dk*Cs, D0z*Ck, axis=1)
		D_aux = np.append(-D0z*Dk,D0z,axis=1)
		#
		A_bar = np.asmatrix(A_aux)
		B_bar = np.asmatrix(B_aux)
		C_bar = np.asmatrix(C_aux)
		D_bar = np.asmatrix(D_aux)
		P_bar = np.asmatrix(P_aux)
		#
		aux1 = C_bar*( (np.eye(A_bar.shape[0])-A_bar).I )*B_bar + D_bar
		aux2 = C_bar*( (np.eye(A_bar.shape[0])-A_bar).I )*P_bar 
		aux3 = np.eye(Z0.shape[0])- np.append(np.zeros((aux2.shape[0],2)),aux2,axis=1)
		Acl = aux1.I*aux3
		#Acl = np.asmatrix(data['Acl'])
		#}}}
		# [y0,u0].T = Acl*z_ref
		# --- Discrete System ----#
		#(As_d, Bs_d, Cs_d, Ds_d, Ps_d) = sys.SS_dis
		# --- Regulation Vsd, ws----#
		y_nom = Model_Sys.y_nom
		Cz = np.matrix([[0.,0.,1.,0.,0.,0.,0.],[0.,0.,0.,1.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.]])
		Dz = np.matrix([[0.,0.,0.],[0.,0.,0.],[0.,0.,1.]])
		d_nom = np.matrix([[Model_Sys.Vb,0.,Model_Sys.wb]]).T
		#Ayx = (np.eye(A.shape[0]) - A - B*Dk*C).I*B*Ck*(np.eye(Ak.shape[0])-Ak).I*Bk #+ Ps*d_nom
		z_nom = np.matrix(np.zeros((3,y_nom.shape[1])) )
		u_nom = np.matrix(np.zeros((3,y_nom.shape[1])) )
		#K_xy = Cz*Ayx + Dz*Dk + Dz*Ck*(np.eye(Ak.shape[0])-Ak).I*Bk
		K_uy = Ck*(np.eye(Ak.shape[0])-Ak).I*Bk + Dk
		#print K_xy
		for i in xrange(y_nom.shape[1]):
		#	y_nom = xnom[0:6,i]
			z_ref = np.matrix([[0.,0.,Model_Sys.Vb,0.,Model_Sys.wb]]).T
			#z_ref = np.matrix([[y_nom[4,i],y_nom[5,i],Model_Sys.Vb,0.,Model_Sys.wb]]).T
			z0 = (Acl*z_ref)#np.zeros((9,1))#
			y0 = z0[0:6,:]
			u0 = z0[6:9:]
			#print y_nom
			aux_nom = K_uy*(y_nom[:,i]-y0)+u0
			#print (As+Bs*K_uy*Cs).I
			#aux_nom = (np.eye(As.shape[0])-(As+Bs*K_uy*Cs)).I*(-Bs*K_uy*y0 + Bs*u0 + Ps*z_ref[2:5,0])
			print y0
			# to the input 
			#u_aux = K_uy*(Cs*aux_nom-y0)+u0
			#print u_aux[2,0]
			u_nom[:,i] = np.matrix([[aux_nom[0,0]],[aux_nom[1,0]],[aux_nom[2,0]]])
			#u_nom[:,i] = np.matrix([[aux_nom[2,0]],[aux_nom[3,0]],[u_aux[2,0]]])
			# to the voltage vs
			io_ = y_nom[4,i] + 1J*y_nom[5,i]
			vt_ = aux_nom[0,0] + 1J*aux_nom[1,0]
			wt_ = aux_nom[2,0]
			vs_ = (vt_ - (Model_Sys.Rf+1J*wt_*Model_Sys.Lf)*io_)/(1.+(Model_Sys.Rf+1J*wt_*Model_Sys.Lf)*1J*wt_*Model_Sys.Cf)
			#print vs_
			#z_nom[:,i] = np.matrix([[aux_nom[1,0]],[aux_nom[2,0]]])
			# obtaining z from u 
			z_nom[:,i] = np.matrix([[vs_.real],[vs_.imag],[aux_nom[2,0]]])#u_nom[:,i]#
			#z_nom[:,i] = np.matrix([[vs_.real],[vs_.imag],[aux_nom[2,0]]])
			# direct evaluation of z
			#z_nom[:,i] = K_xy*(y_nom[:,i]-y0)
	#}}}

	if Case == 2: # considers the reference zref = [vsd,vsq,w]
		#{{{z
	# Calculating the matrix for the currents, voltage and frequency references
		#{{{
		(As,Bs,Cs,Ds,Ps,Dwz) = Model_Sys.SS_dis
		As = np.matrix(As,dtype=float)
		Bs = np.matrix(Bs,dtype=float)
		Cs = np.matrix(Cs,dtype=float)
		Ds = np.matrix(Ds,dtype=float)
		Ps = np.matrix(Ps,dtype=float)
	
		Z0 = np.matrix([[0.,0.,1.,0.,0.,0.,0.],[0.,0.,0.,1.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.]],dtype=float)
		D0z = np.matrix([[0.,0.,0.],[0.,0.,0.],[0.,0.,1.]],dtype=float)
		#
		aux1 = np.append(As+Bs*Dk*Cs, Bs*Ck,axis=1)
		aux2 = np.append(Bk*Cs, Ak,axis=1)
		A_aux = np.append(aux1,aux2,axis=0)
		# 
		B_aux = np.append(Bs, np.zeros((Ak.shape[0],Bs.shape[1])), axis=0)
		#
		P_aux = np.append(Ps, np.zeros((Ak.shape[0],Bs.shape[1])) , axis=0)
		#C_aux = np.append(Z0, np.zeros((Z0.shape[0],Ac.shape[0])), axis=1)
		C_aux = np.append(Z0+D0z*Dk*Cs, D0z*Ck, axis=1)
		#D_aux = np.append(-D0z*Dk,D0z,axis=1)
		D_aux = D0z
		#
		A_bar = np.matrix(A_aux,dtype=float)
		B_bar = np.matrix(B_aux,dtype=float)
		C_bar = np.matrix(C_aux,dtype=float)
		D_bar = np.matrix(D_aux,dtype=float)
		P_bar = np.matrix(P_aux,dtype=float)
		#
		aux1 = C_bar*( (np.eye(A_bar.shape[0])-A_bar).I )*B_bar + D_bar
		aux2 = C_bar*( (np.eye(A_bar.shape[0])-A_bar).I )*P_bar 
		Acl = aux1.I
		Bcl = -Acl*aux2
		#Acl = np.asmatrix(data['Acl'])
		#}}}
		# [y0,u0].T = Acl*z_ref
		# --- Discrete System ----#
		#(As_d, Bs_d, Cs_d, Ds_d, Ps_d) = sys.SS_dis
		# --- Regulation Vsd, ws----#
		y_nom = Model_Sys.y_nom
		#Cz = np.matrix([[0.,0.,1.,0.,0.,0.,0.],[0.,0.,0.,1.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.]])
		#Dz = np.matrix([[0.,0.,0.],[0.,0.,0.],[0.,0.,1.]])
		#d_nom = np.matrix([[Model_Sys.Vb,0.,Model_Sys.wb]]).T
		#Ayx = (np.eye(A.shape[0]) - A - B*Dk*C).I*B*Ck*(np.eye(Ak.shape[0])-Ak).I*Bk #+ Ps*d_nom
		z_nom = np.matrix(np.zeros((3,y_nom.shape[1])),dtype=float )
		u_nom = np.matrix(np.zeros((3,y_nom.shape[1])),dtype=float )
		#K_xy = Cz*Ayx + Dz*Dk + Dz*Ck*(np.eye(Ak.shape[0])-Ak).I*Bk
		K_uy = Ck*(np.eye(Ak.shape[0])-Ak).I*Bk + Dk
		#print K_xy
		for i in xrange(y_nom.shape[1]):
		#	y_nom = xnom[0:6,i]
			z_ref = np.matrix([[Model_Sys.Vb,0.,Model_Sys.wb]]).T
			#z_ref = np.matrix([[y_nom[4,i],y_nom[5,i],Model_Sys.Vb,0.,Model_Sys.wb]]).T
			u0 = (Acl*z_ref + Bcl*z_ref)#np.zeros((9,1))#
			#y0 = z0[0:6,:]
			#u0 = z0[6:9:]
			#print y_nom
			aux_nom = K_uy*y_nom[:,i] + u0
			#print (As+Bs*K_uy*Cs).I
			#aux_nom = (np.eye(As.shape[0])-(As+Bs*K_uy*Cs)).I*(-Bs*K_uy*y0 + Bs*u0 + Ps*z_ref[2:5,0])
			#print y0
			# to the input 
			#u_aux = K_uy*(Cs*aux_nom-y0)+u0
			#print u_aux[2,0]
			u_nom[:,i] = np.matrix([[aux_nom[0,0]],[aux_nom[1,0]],[aux_nom[2,0]]])
			#u_nom[:,i] = np.matrix([[aux_nom[2,0]],[aux_nom[3,0]],[u_aux[2,0]]])
			# to the voltage vs
			io_ = y_nom[4,i] + 1J*y_nom[5,i]
			vt_ = aux_nom[0,0] + 1J*aux_nom[1,0]
			wt_ = aux_nom[2,0]
			vs_ = (vt_ - (Model_Sys.Rf+1J*wt_*Model_Sys.Lf)*io_)/(1.+(Model_Sys.Rf+1J*wt_*Model_Sys.Lf)*1J*wt_*Model_Sys.Cf)
			#print vs_
			#z_nom[:,i] = np.matrix([[aux_nom[1,0]],[aux_nom[2,0]]])
			# obtaining z from u 
			z_nom[:,i] = np.matrix([[vs_.real],[vs_.imag],[aux_nom[2,0]]])  #u_nom[:,i]#
			#z_nom[:,i] = np.matrix([[vs_.real],[vs_.imag],[aux_nom[2,0]]])
			# direct evaluation of z
			#z_nom[:,i] = K_xy*(y_nom[:,i]-y0)
	#}}}
	#--------------------------------------------#
	#print z_nom[0,:]
	return Ak,Bk,Ck,Dk,z_nom
	#}}}



	'''
	# ======= GENERALIZED H2; io_inf/wg ====== #
	#{{{ w = Rj*wj , zj = Lj*z, wj = [],zj =[vtd,vtq]
	# regulation ws/io
	Lj = np.matrix([[0.,0.,0.,0.,1.]],dtype=float)
	Rj = np.matrix([[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.]]).T
	Gam =(2.*np.pi*1./1000.)**2#np.sqrt(0.6)#np.sqrt(0.11)#*np.sqrt(2)#1.*np.sqrt(1.)/(3.)
	# regulation io vs wg
	#Lj = np.matrix([[1.,0.,0.,0.,0.],[0.,1.,0.,0.,0.]],dtype=float)
	#Rj = np.matrix([[0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.]]).T
	#Gam =(5500./(2.*np.pi*3.))**2#np.sqrt(0.6)#np.sqrt(0.11)#*np.sqrt(2)#1.*np.sqrt(1.)/(3.)
	Bj = Bw*Rj
	Cj = Lj*Cz
	Dj = Lj*Dzw*Rj
	Ej = Lj*Dz
	Fj = Dw*Rj
	sys_LMI = (Bj,Cj,Dj,Ej,Fj,Gam)
	(Cons4, Cons5,Cons6) = LMI_Gep_H2(sys_LMI)
	#
	#}}}
	#consts += [Cons4,Cons5]

	# ======= GENERALIZED H2; io_inf/vg ====== #
	#{{{ w = Rj*wj , zj = Lj*z, wj = [],zj =[vtd,vtq]
	# regulation ws/io
	Lj = np.matrix([[0.,0.,1.,0.,0.]],dtype=float)
	Rj = np.matrix([[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.]]).T
	Gam =(520*0.9/5500.)**2#np.sqrt(0.6)#np.sqrt(0.11)#*np.sqrt(2)#1.*np.sqrt(1.)/(3.)
	# regulation io vs wg
	#Lj = np.matrix([[1.,0.,0.,0.,0.],[0.,1.,0.,0.,0.]],dtype=float)
	#Rj = np.matrix([[1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]]).T
	#Gam =(5500./(520.*0.5))**2#np.sqrt(0.6)#np.sqrt(0.11)#*np.sqrt(2)#1.*np.sqrt(1.)/(3.)
	Bj = Bw*Rj
	Cj = Lj*Cz
	Dj = Lj*Dzw*Rj
	Ej = Lj*Dz
	Fj = Dw*Rj
	sys_LMI = (Bj,Cj,Dj,Ej,Fj,Gam)
	(Cons4, Cons5,Cons6) = LMI_Gep_H2(sys_LMI)
	#
	#}}}
	#consts += [Cons4,Cons5]


	# ======= Hinf norm for power signals REGULATION; ws/io ====== #
	#z{{{ w = Rj*wj , zj = Lj*z, wj = [],zj =[vtd,vtq]
	# regulation ws/io
	Lj = np.matrix([[0.,0.,0.,0.,1.]],dtype=float)
	Rj = np.matrix([[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.]]).T
	Gam = (np.pi/5500.)**2#np.sqrt(0.6)#np.sqrt(0.11)#*np.sqrt(2)#1.*np.sqrt(1.)/(3.)
	# regulation io vs wg
	#Lj = np.matrix([[1.,0.,0.,0.,0.],[0.,1.,0.,0.,0.]],dtype=float)
	#Rj = np.matrix([[1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]]).T
	#Gam =(5500./(520.*0.5))**2#np.sqrt(0.6)#np.sqrt(0.11)#*np.sqrt(2)#1.*np.sqrt(1.)/(3.)
	#Bj = Bw*Rj
	#Cj = Lj*Cz
	#Dj = Lj*Dzw*Rj
	#Ej = Lj*Dz
	#Fj = Dw*Rj
	#sys_LMI = (Bj,Cj,Dj,Ej,Fj,Gam)
	aux1 = (Lj,Rj,gamma)
	aux2 = (Cz,Dz,Bw,Dw,Dzw)
	#Cons_ss = LMI_H2_Static(sys_LMI)
	Cons_a= Hinf_LMI(aux1,aux2)
	#
	#}}}
	#consts += [Cons_a]

	# ======= Hinf norm for power signals REGULATION; vs/io ====== #
	#{{{ w = Rj*wj , zj = Lj*z, wj = [],zj =[vtd,vtq]
	# regulation ws/io
	Lj = np.matrix([[0.,0.,1.,0.,0.],[0.,0.,0.,1.,0.]],dtype=float)
	Rj = np.matrix([[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.]]).T
	Gam = (520.*0.05/5500.)**2#np.sqrt(0.6)#np.sqrt(0.11)#*np.sqrt(2)#1.*np.sqrt(1.)/(3.)
	# regulation io vs wg
	#Lj = np.matrix([[1.,0.,0.,0.,0.],[0.,1.,0.,0.,0.]],dtype=float)
	#Rj = np.matrix([[1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]]).T
	#Gam =(5500./(520.*0.5))**2#np.sqrt(0.6)#np.sqrt(0.11)#*np.sqrt(2)#1.*np.sqrt(1.)/(3.)
	#Bj = Bw*Rj
	#Cj = Lj*Cz
	#Dj = Lj*Dzw*Rj
	#Ej = Lj*Dz
	#Fj = Dw*Rj
	#sys_LMI = (Bj,Cj,Dj,Ej,Fj,Gam)
	aux1 = (Lj,Rj,gamma)
	aux2 = (Cz,Dz,Bw,Dw,Dzw)
	#Cons_ss = LMI_H2_Static(sys_LMI)
	Cons_a= Hinf_LMI(aux1,aux2)
	#
	#}}}
	#consts += [Cons_a]
	'''



