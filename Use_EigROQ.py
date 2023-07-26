import numpy as np
import lal
import lalsimulation
import multiprocessing as mp
import pickle
import EigROQ

import time
start_runtime = time.time()

################################################################################################

#Setting up boundary conditions and tolerance requirements.
mc_low = 8.6
mc_high = 11.8
q_low = 1
q_high = 4
s1sphere_low = [0, 0, 0]
s1sphere_high = [0.8, np.pi, 2.0*np.pi]
s2sphere_low = [0, 0, 0]
s2sphere_high = [0.8, np.pi, 2.0*np.pi]
iota_low = 0
iota_high = np.pi
phiref_low = 0
phiref_high = 2*np.pi
seglen = 8
f_min = 20
f_max = 1024
distance = 10 * lal.lal.PC_SI * 1.0e6  # 10 Mpc is default 
laldict = lal.CreateDict()
approximant = lalsimulation.IMRPhenomXPHM
lalsimulation.SimInspiralWaveformParamsInsertPhenomXPHMThresholdMband(laldict, 0)

#Setting up tolerance requirements.
Nwave = 4000
Nits = 2
Nmax0 = 1e5
Nmaxf = 1e6
ROB_tol0 = 1e-2
ROB_tolf = 1e-4
EIM_tol = ROB_tolf
max_rounds=5
max_weight=1e4
train_extra='sum'
n_check_point=10

#Computing waveforms in batches and checkpointing
Nbatch = 2000
Ncheckpoint = 10000

#Setting up the requirements of the tests
Ntests = int(1e6)
nbins = 100
ROB_err_thr=1e-10
psd_name = 'aLIGOO3LowT1800545'

#multiprocessing arguments
nprocesses = mp.cpu_count()-1

##############################################################################################

#take input arguments
import argparse
# Define the arguments of this program 
parser = argparse.ArgumentParser()
parser.add_argument("--LinearROQ", dest='LinearROQ', action='store_true', help="Compute Linear ROQ")
parser.add_argument("--QuadraticROQ", dest='QuadraticROQ', action='store_true', help="Compute Quadratic ROQ")
parser.add_argument("--ROQParams", dest='ROQParams', action='store_true', help="Print file with ROQ parameters")
parser.add_argument("--TestROQ", dest='TestROQ', action='store_true', help="Test the ROQ")
#load arguments
args = parser.parse_args()

if args.LinearROQ: 
	#run EigROQ for linear basis
	EigROQ.EigROQ(mc_low=mc_low, mc_high=mc_high, q_low=q_low, q_high=q_high, s1sphere_low=s1sphere_low, s1sphere_high=s1sphere_high, s2sphere_low=s2sphere_low, s2sphere_high=s2sphere_high, iota_low=iota_low, iota_high=iota_high, phiref_low=phiref_low, phiref_high=phiref_high, seglen=seglen, f_min=f_min, f_max=f_max, distance=distance, laldict=laldict, approximant=approximant, basis_type='linear', Nwave=Nwave, Nits=Nits, Nmax0=Nmax0, Nmaxf=Nmaxf, ROB_tol0=ROB_tol0, ROB_tolf=ROB_tolf, EIM_tol=EIM_tol, Nbatch=Nbatch, Ncheckpoint=Ncheckpoint, max_rounds=max_rounds, max_weight=max_weight, train_extra=train_extra, n_check_point=n_check_point, nprocesses=nprocesses)

elif args.QuadraticROQ:
	#run EigROQ for quadratic basis
	EigROQ.EigROQ(mc_low=mc_low, mc_high=mc_high, q_low=q_low, q_high=q_high, s1sphere_low=s1sphere_low, s1sphere_high=s1sphere_high, s2sphere_low=s2sphere_low, s2sphere_high=s2sphere_high, iota_low=iota_low, iota_high=iota_high, phiref_low=phiref_low, phiref_high=phiref_high, seglen=seglen, f_min=f_min, f_max=f_max, distance=distance, laldict=laldict, approximant=approximant, basis_type='quadratic', Nwave=Nwave, Nits=Nits, Nmax0=Nmax0, Nmaxf=Nmaxf, ROB_tol0=ROB_tol0, ROB_tolf=ROB_tolf, EIM_tol=EIM_tol, Nbatch=Nbatch, Ncheckpoint=Ncheckpoint, max_rounds=max_rounds, max_weight=max_weight, train_extra=train_extra, n_check_point=n_check_point, nprocesses=nprocesses)

elif args.ROQParams:	
	#Generate ROQParams
	EigROQ.ROQparams(mc_low=mc_low, mc_high=mc_high, q_low=q_low, q_high=q_high, s1sphere_low=s1sphere_low, s1sphere_high=s1sphere_high, s2sphere_low=s2sphere_low, s2sphere_high=s2sphere_high, iota_low=iota_low, iota_high=iota_high, phiref_low=phiref_low, phiref_high=phiref_high, seglen=seglen, f_min=f_min, f_max=f_max)

elif args.TestROQ:

	#Test the ROQ
	EigROQ.TestROQ(mc_low=mc_low, mc_high=mc_high, q_low=q_low, q_high=q_high, s1sphere_low=s1sphere_low, s1sphere_high=s1sphere_high, s2sphere_low=s2sphere_low, s2sphere_high=s2sphere_high, iota_low=iota_low, iota_high=iota_high, phiref_low=phiref_low, phiref_high=phiref_high, seglen=seglen, f_min=f_min, f_max=f_max, distance=distance, laldict=laldict, approximant=approximant, EIM_tol=EIM_tol, Ntests=Ntests, nbins=nbins, ROB_err_thr=ROB_err_thr, psd_name=psd_name, Nbatch=Nbatch, Ncheckpoint=Ncheckpoint, nprocesses=nprocesses)

#Runtime
print("\nRuntime: %s seconds" % (time.time() - start_runtime))


