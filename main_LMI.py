import numpy as np
from scipy.special import factorial
#import cvxpy as cp
from sympy import *
from scipy.linalg import solve_discrete_are,expm,eigvals,expm,svd,pinv,eig,solve_continuous_are,sqrtm
z = symbols("z")

def C2D(A,B,C,D,P,dtd,method):
	#{{{
	# SDR: Simple Derivate Replacement
	# TDR: Tustin Derivate Replacement
	# ASZ: Asymptotic Sampling Zeros
	if method == 'SDR':
		A1 = np.eye(A.shape[0]) + dtd*A
		B1 = dtd*B
		C1 = C
		D1 = D
	if method == 'TDR':
		aux1 = np.asmatrix(np.eye(A.shape[0])*(2./dtd)-A)
		aux2 = np.asmatrix(np.eye(A.shape[0])*(2./dtd)+A)
		A1 = aux1.I*aux2
		B1 = aux1.I*B
		C1 = C
		D1 = C*aux1.I*B + D
	if method == 'ZOH':
		A1 = expm(A*dtd)
		B1 = np.asmatrix(np.zeros((A.shape[1],B.shape[1])) )
		for i in xrange(40):
			B1 = B1+ (A**i)*B*(dtd**(i+1))/(factorial(i+1))
		P1 = np.asmatrix(np.zeros((A.shape[1],P.shape[1])) )
		for i in xrange(40):
			P1 = P1+ (A**i)*P*(dtd**(i+1))/(factorial(i+1))
		C1 = C
		D1 = D
	return A1,B1,C1,D1,P1
	#}}}

def Model_Sys(pars,dtd):
	#{{{
	Rf = pars[0]
	Lf = pars[1]
	Cf = pars[2]
	Rg = pars[3]
	Lg = pars[4]
	wb = pars[5]
	Vb = pars[6]
	# State Space, Continuous time
	A0 = np.matrix([[-(Rf/Lf),wb,-1./Lf,0.,0.,0.,0.],
			[-wb,-Rf/Lf,0.,-1./Lf,0.,0.,0.],
			[1./Cf,0.,0.,wb,-1./Cf,0.,0.],
			[0.,1./Cf,-wb,0.,0.,-1./Cf,0.],
			[0.,0.,1./Lg,0.,-Rg/Lg,wb,0.],
			[0.,0.,0.,1./Lg,-wb,-Rg/Lg, -Vb/Lg],
			[0.,0.,0.,0.,0.,0.,0.]])
	B0 = np.matrix([[1./Lf,0.,0.],[0.,1./Lf,0.],[0.,0.,0.],[0.,0.,-0.*Vb],[0.,0.,0.],[0.,0.,0.],[0.,0.,-1]])
	B0w = np.matrix([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.],[0.,0.,0.],[-1./Lg,0.,0.],[0.,-1./Lg,0.],[0.,0.,1.]])
	C0 = np.asmatrix(np.append(np.eye(6),np.zeros((6,1)),axis=1))
	D0w = np.append(np.zeros((6,6)),np.eye(6),axis=1)
	#
	# SS Discrete
	Ad,Bd,Cd,Ddw,Bdw = C2D(A0,B0,C0,D0w,B0w,dtd,'ZOH')
	Dzw = np.zeros((6,3))
	SS_dis = (Ad,Bd,Cd,Ddw,Bdw,Dzw)
	Bdw = np.append(Bdw,Bd, axis=1)
	Bdw = np.append(Bdw,np.zeros((7,6)), axis=1)
	# chosen output z = [vsd,vsq,wt]
	Cz = np.matrix([[0.,0.,1.,0.,0.,0.,0.],[0.,0.,0.,1.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.]],dtype=float)
	Dz = np.matrix([[0.,0.,0.],[0.,0.,0.],[0.,0.,1.]],dtype=float)
	Dzw = np.matrix([[0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]],dtype=float)
	SS_ext_dis = (Ad,Bd,Cd,Ddw,Bdw,Cz,Dz,Dzw)
	#
	return SS_ext_dis
	#}}}


def SS_closed_loop(sys,ctr,ss_out):
	#{{{
	#ss = (A,B,C,Dw,Bw,Cz,Dz,Dzw)
	As = sys[0]
	Bs = sys[1]
	Cs = sys[2]
	Dw = sys[3]
	Bw = sys[4]
	Cz = sys[5]
	Dz = sys[6]
	Dzw = sys[7]

	Ac = ctr[0]
	Bc = ctr[1]
	Cc = ctr[2]
	Dc = ctr[3]

	Rj = ss_out[0]
	Lj = ss_out[1]
	#
	Bj = Bw*Rj
	Cj = Lj*Cz
	Dj = Lj*Dzw*Rj
	Ej = Lj*Dz
	Fj = Dw*Rj
	#
	aux1 = np.append(As+Bs*Dc*Cs, Bs*Cc,axis=1)
	aux2 = np.append(Bc*Cs,Ac,axis=1)
	Acl = np.append(aux1,aux2,axis=0) 
	#
	Bcl = np.append(Bs*Dc*Fj + Bj,Bc*Fj,axis=0)
	#
	Ccl = np.append(Cj + Ej*Dc*Cs, Ej*Cc,axis=1)
	Dcl = Dj + Ej*Dc*Fj
	return Acl,Bcl,Ccl,Dcl
	#}}}

def disc_bode(Sys,dtd):
#{{{
	A = np.asmatrix(Sys[0],dtype=float)
	B = np.asmatrix(Sys[1],dtype=float)
	C = np.asmatrix(Sys[2],dtype=float)
	D = np.asmatrix(Sys[3],dtype=float)
	w_f = np.log10(np.pi/dtd)
	Om = dtd*np.logspace(-3.,w_f,num=800)
	N = Om.shape[0]
	#
	aux = (z*eye(A.shape[0])-A) 
	Fi_1 = lambdify( (z),aux)
	svd_G = np.asmatrix( np.zeros((2,N)) ) 
	for i in xrange(N):
		Fi = Fi_1(np.exp(1J*Om[i]))
		aux = C*np.matrix(Fi,dtype=complex).I*B + D
		U,s,V = svd(aux)
		svd_G[0,i] = np.amax(s)  
		svd_G[1,i] = np.amin(s) 
	return svd_G,Om/dtd
#}}}


