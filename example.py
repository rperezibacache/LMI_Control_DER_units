# ----- Dependencies ----- #
import numpy as np
from scipy.linalg import svd,sqrtm,eig
import main_LMI as LMI
import cvxpy as cp


# ---------------------------------------------------------------------------- #'
# ----------------------- LOCAL FUNCTIONS ------------------------------------ #'
# ---------------------------------------------------------------------------- #'

def LMI_H2(sys):
	#{{{
	Rj=sys[0]
	Lj=sys[1]
	Bj = Bw*Rj
	Cj = Lj*Cz
	Dj =Lj*Dzw*Rj
	Ej = Lj*Dz
	Fj = Dw*Rj
	Q = sys[2]
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
	return (LMI1,LMI2)	
	#}}}

def Hinf_LMI(aux1):
	#{{{
	Lj = aux1[0]
	Rj = aux1[1]
	gamma = aux1[2]
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
	return Cons_a
	#}}}

def decay_time(Dsgn):
	#{{{
	Pi1T_P_Pi1 = cp.bmat([[X, np.eye(nx)],[np.eye(nx), Y]])
	Pi1T_PA_Pi1 = cp.bmat([[A*X + B*C_h, A+B*D_h*C],[A_h, Y*A+B_h*C]]) 
	#
	alfa = Dsgn[0]
	dtd = Dsgn[1]
	aux11 = -np.exp(-2*alfa*dtd)*Pi1T_P_Pi1 
	aux12 = Pi1T_PA_Pi1.T
	aux22 = -Pi1T_P_Pi1
	LMI = cp.bmat([[aux11,aux12],[aux12.T,aux22]])
	return LMI
	#}}}

# ---------------------------------------------------------------------------- #'
# ----------------------- END LOCAL FUNCTIONS -------------------------------- #'
# ---------------------------------------------------------------------------- #'



print ('# ---------------------------------------------------------------------------- #')
print ('# ----------------------- SYSTEM DEFINITION ---------------------------------- #')
print ('# ---------------------------------------------------------------------------- #')
# DER 1 
Sb = 2.e6 # nominal power DER unit
wb = 2.*np.pi*60. # nominal frequency 
Vb = 520. # peak phase to neutral
ib = 2.*Sb/(3.*Vb) # peak nominal current
dtd = 1./5000. # sampling time 
#
Rf = 1.62e-3
Lf = 43.e-6
Cf = 1.3e-3
Rg = 2.e-3
Lg = 9.3e-6
#
print ('Sb = %s[MVA], vb = %s[V], wb = %s[rad/s], ib = %s[kA]' % (Sb/(1.e6), Vb, round(wb,1), round(ib/(1.e3),2) ))
print ('Rf = %s[mOhm], Lf = %s[uH], Cf = %s[mF], Rg = %s[mOhm], Lg = %s[uH]' % (round(Rf*(1.e3),2), round(Lf*(1.e6),2), round(Cf*(1.e3),1), round(Rg*(1.e3),2),round(Lg*(1.e6),2) ))
print ('# ---------------------------------------------------------------------------- #')
# system model, return discrete-time representation
pars = (Rf,Lf,Cf,Rg,Lg,wb,Vb)
#
dis_SS = LMI.Model_Sys(pars,dtd)
#
A = dis_SS[0]
B = dis_SS[1]
C = dis_SS[2]
Dw = dis_SS[3]
Bw = dis_SS[4]
Cz = dis_SS[5]
Dz = dis_SS[6]
Dzw = dis_SS[7]

'# -------------------------------------------------------------------------------- #'
'# ----------------------- OPTIMIZATION PROBLEM  ---------------------------------- #'
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
q = cp.Variable(3)#Cj.shape[0])
Q_var = cp.diag(q)
# === Constraints === #
consts = []
#==============#
#===== H2 =====#
R1 = np.matrix(np.eye(12),dtype=float)
L1 = np.matrix(np.eye(3),dtype=float)
#
sys_LMI = (R1,L1,Q_var)
(Cons1, Cons2) = LMI_H2(sys_LMI)
consts += [Cons1,Cons2]
#===================#
#=== H_inf First ===#
gamma_2 = (2.*np.pi/ib)**2
L2 = np.matrix([[0.,0.,1.]])
R2 = np.matrix([[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.],
				[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,1.],	 ]).T
sys_LMI = (L2,R2,gamma_2)
#
Cons3 = Hinf_LMI(sys_LMI)
consts += [Cons3]
#====================#
#=== H_inf Second ===#
gamma_3 = (1.e-2/ib)**2#0.0001/5500.
L3 = np.matrix([[0.,0.,1.]])
R3 = np.matrix([[0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.],
				[0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.,0.],
 				[0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.,0.],
				[0.,0.,0.,0.,0.,0.,0.,0.,0.,1.,0.,0.],	 ]).T
sys_LMI = (L3,R3,gamma_3)
#
Cons4 = Hinf_LMI(sys_LMI)
consts += [Cons4]
#===================#
#=== H_inf Third ===#
gamma_4 = (Vb*0.1)**2
L4 = np.matrix([[0.,0.,1.]])
R4 = np.matrix([[1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
				[0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]]).T
sys_LMI = (L4,R4,gamma_4)
#	
Cons5 = Hinf_LMI(sys_LMI)
consts += [Cons5]
#===================#
#=== H_inf Fourth ===#
gamma_5 = (2.*np.pi)**2
L5 = np.matrix([[0.,0.,1.]])
R5 = np.matrix([[0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.]]).T
sys_LMI = (L5,R5,gamma_5)
#	
Cons6 = Hinf_LMI(sys_LMI)
consts += [Cons6]
#======================#
#=== Settling time  ===#
alfa = 30. 
#
sys_LMI = (alfa,dtd)
LMI7 = decay_time(sys_LMI)
Cons7 = LMI7 << 0
consts += [Cons7]

# -------------------------------------------------------------------------------- #
# ------------------ SOLVING OPTIMIZATION PROBLEM  ------------------------------- #
optprob = cp.Problem(cp.Minimize(cp.trace(Q_var) ), constraints=consts)
print("prob is DCP:", optprob.is_dcp())
result = optprob.solve(solver=cp.CVXOPT,verbose=True, max_iters=10000, abstol=5.e-6, reltol=5.e-6,feastol=5.e-6,refinement = 30,kktsolver='robust')
# ------------------  OPTIMIZATION PROBLEM SOLVED  ------------------------------- #
# -------------------------------------------------------------------------------- #
#
# -------------------------------------------------------------------------------- #
# ------------------ RECOVERUNG THE CONTROLLER  ------------------------------- #
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
ss = (A,B,C,Dw,Bw,Cz,Dz,Dzw)
LMI_ctr = (Ak,Bk,Ck,Dk)
print ('# ----------------------------------------- #')
print ('# -------- Controller --------------------- #')
print 'Ac = ',Ak
print 'Bc = ',Bk
print 'Cc = ',Ck
print 'Dc = ',Dk
print ('# ----------------------------------------- #')
print ('# --------- Norms-------------------------- #')
print 'tr(Q*) = ', round(Q_LMI.trace()[0,0],4)
ss_out = (R2,L2)
(Acl,Bcl,Ccl,Dcl) = LMI.SS_closed_loop(ss,LMI_ctr,ss_out)
aux = (Acl,Bcl,Ccl,Dcl)	
svd,Om = LMI.disc_bode(aux,dtd)	
print 'gamma_2 = %s, gamma*_2 = %s ' % (gamma_2, np.max(svd))
ss_out = (R3,L3)
(Acl,Bcl,Ccl,Dcl) = LMI.SS_closed_loop(ss,LMI_ctr,ss_out)
aux = (Acl,Bcl,Ccl,Dcl)	
svd,Om = LMI.disc_bode(aux,dtd)	
print 'gamma_3 = %s, gamma*_3 = %s ' % (gamma_3,np.max(svd))
ss_out = (R4,L4)
(Acl,Bcl,Ccl,Dcl) = LMI.SS_closed_loop(ss,LMI_ctr,ss_out)
aux = (Acl,Bcl,Ccl,Dcl)	
svd,Om = LMI.disc_bode(aux,dtd)	
print 'gamma_4 = %s, gamma*_4 = %s ' % (gamma_4,np.max(svd))
ss_out = (R5,L5)
(Acl,Bcl,Ccl,Dcl) = LMI.SS_closed_loop(ss,LMI_ctr,ss_out)
aux = (Acl,Bcl,Ccl,Dcl)	
svd,Om = LMI.disc_bode(aux,dtd)	
print 'gamma_5 = %s, gamma*_5 = %s ' % (gamma_5,np.max(svd))
print ('# ----------------------------------------- #')
print ('# --------- settling time ----------------- #')
ss_out = (1.,1.)
(Acl,Bcl,Ccl,Dcl) = LMI.SS_closed_loop(ss,LMI_ctr,ss_out)
w_eig,v =eig(np.asmatrix(Acl,dtype=float) )
w_n = np.absolute(np.log(w_eig)/dtd)
xi_n = -np.cos(np.angle(np.log(w_eig) ) )
print 'tau*_s = ', 1./min(w_n) 
print ('# --------- your design is DONE!--------------------- #')
print ('# --------------------------------------------------- #')


