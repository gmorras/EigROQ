import numpy as np
import lal
import lalsimulation
import multiprocessing as mp
import pickle
from datetime import datetime

#function to return the current time
def current_time():
    return f"{datetime.now():%Y-%m-%d %H:%M:%S}"

#function to convert spherical coordinates into cartesian coordinates (inspired in the pyroq function)
def spherical_to_cartesian(sph):
	x = sph[0] * np.sin(sph[1]) * np.cos(sph[2])
	y = sph[0] * np.sin(sph[1]) * np.sin(sph[2])
	z = sph[0] * np.cos(sph[1])
	car = [x, y, z]
	for idx, val in enumerate(car):
		if np.abs(val) < 1e-9:
			car[idx] = 0
	return car

#function to compute the effective inspiral spin parameter from params
def chi_eff(parampoint):
	#extract parameters from parampoint
	q = parampoint[1]
	s1 = parampoint[2]
	th1 = parampoint[3]
	s2 = parampoint[5]
	th2 = parampoint[6]
	
	#return chi_eff
	return (q*s1*np.cos(th1) + s2*np.cos(th2))/(q+1)

#function to compute the effective precession spin parameter from params
def chi_p(parampoint):
	#extract parameters from parampoint
	q = parampoint[1]
	s1 = parampoint[2]
	th1 = parampoint[3]
	s2 = parampoint[5]
	th2 = parampoint[6]
	#return chi_p
	return max(s1*np.sin(th1), ((4+3*q)/(q*(4*q + 3)))*s2*np.sin(th2))
	
	
#function to generate normalized waveforms (inspired in the pyroq function)
def generate_a_normalized_waveform_from_parampoint(basistype, parampoint, distance, delta_f, f_min, f_max, approximant, laldict=lal.CreateDict()):
	#extract parameters from parampoint
	mc = parampoint[0]
	q = parampoint[1]
	s1 = spherical_to_cartesian(parampoint[2:5])
	s2 = spherical_to_cartesian(parampoint[5:8])
	iota = parampoint[8]
	phiref = parampoint[9]
	ecc = 0
	lambda1 = 0
	lambda2 = 0
	if len(parampoint) == 11:
		ecc = parampoint[10]
	if len(parampoint) == 12:
		lambda1 = parampoint[10]
		lambda2 = parampoint[11]
		lalsimulation.SimInspiralWaveformParamsInsertTidalLambda1(laldict, lambda1)
		lalsimulation.SimInspiralWaveformParamsInsertTidalLambda2(laldict, lambda2)

	#convert mc, q to m1, m2 (assuming that q>1)
	test_mass2 = lal.lal.MSUN_SI*mc*(q**(-0.6))*((1 + q)**0.2)
	test_mass1 = test_mass2*q
	#compute waveform using lal
	[plus_test, cross_test] = lalsimulation.SimInspiralChooseFDWaveform(test_mass1, test_mass2, s1[0], s1[1], s1[2], s2[0], s2[1], s2[2], distance, iota, phiref, 0, ecc, 0, delta_f, f_min, f_max, 0, laldict, approximant)
	hp = plus_test.data.data
	hp_test = hp[int(f_min / delta_f) : int(f_max / delta_f)]
	#if the basistype is quadratic, return |h|^2
	if basistype == "quadratic": hp_test = (np.abs(hp_test))**2
	#compute the norm
	hp_norm = np.linalg.norm(hp_test)
	
	#return normalized waveform (if the norm is not 0)
	if hp_norm>0:
		return hp_test/hp_norm
	else:
		return hp_test

#function to generate a list of N random waveforms
def generate_waveforms(basis_type, distance, delta_f, f_min, f_max, approximant, laldict=lal.CreateDict(), nprocesses=0, params_low=None, params_high=None, Nwave=None, params=None):
	
	if params is None:
		#generate the necessary random points
		params = np.random.uniform(params_low, params_high, size=(Nwave,len(params_low)))

	#consider the single process case
	if nprocesses<=1:
		#loop over the random points generated
		waveforms = list()
		for param in params:
			#generate a normalized waveform at this parameter point and add it to waveform list
			waveforms.append(generate_a_normalized_waveform_from_parampoint(basis_type, param, distance, delta_f, f_min, f_max, approximant, laldict=laldict))
			
	#consider the multi-processing case
	else:
		#create a global function evaluating all static arguments
		global generate_waveform_of_param
		def generate_waveform_of_param(param):
			return generate_a_normalized_waveform_from_parampoint(basis_type, param, distance, delta_f, f_min, f_max, approximant, laldict=laldict)

		with mp.Pool(nprocesses) as pool:
			#compute the waveforms using multiprocessing
			waveforms = pool.starmap(generate_waveform_of_param,[(param,) for param in params], )

		#delete the global function
		del generate_waveform_of_param
		
	return np.array(waveforms)

#function to generate initial parameter points
def generate_initial_params(params_low, params_high, Nwave, NMc=3, Neta=3, Ns1=3, Nth1=3, Ns2=3, Nth2=3, Niota=3):
	
	#compute the values for the different parameters
	Mcs = np.linspace(params_high[0]**(-5/3), params_low[0]**(-5/3), NMc)**(-3/5)
	etas = np.linspace(params_low[1]/((1+params_low[1])**2), params_high[1]/((1+params_high[1])**2),Neta)
	qs = ((0.5/etas) - 1) + np.sqrt((((0.5/etas) - 1)**2)-1)
	s1s = np.linspace(params_low[2], params_high[2], Ns1)
	th1s = np.arccos(np.linspace(np.cos(params_low[3]), np.cos(params_high[3]), Nth1))
	s2s = np.linspace(params_low[5], params_high[5], Ns2)
	th2s = np.arccos(np.linspace(np.cos(params_low[6]), np.cos(params_high[6]), Nth2))
	iotas = np.arccos(np.linspace(np.cos(params_low[8]), np.cos(params_high[8]), Niota))

	#take into account that if s=0, theta doesn't matter, and generate random phis
	S1s = list()
	for s1 in s1s:
		if s1<1e-5: 
			S1s.append([s1, 0.0, 0.0])
			continue
		for th1 in th1s:
			S1s.append([s1, th1, np.random.uniform(params_low[4], params_high[4])])

	#the same for S2
	S2s = list()
	for s2 in s2s:
		if s2<1e-5: 
			S2s.append([s2, 0.0, 0.0])
			continue
		for th2 in th2s:
			S2s.append([s2, th2, np.random.uniform(params_low[7], params_high[7])])

	
	#construct the corner part params
	initial_params = list()
	for Mc in Mcs:
		for q in qs:
			for S1 in S1s:
				for S2 in S2s:
					for iota in iotas: 
						initial_params.append([Mc,q,*S1,*S2,iota,np.random.uniform(params_low[9], params_high[9])])

	#print initial params
	print('\nInitial  Mcs:', np.array(Mcs))
	print('Initial    qs:', np.array(qs))
	print('Initial   S1s:', np.array(S1s))
	print('Initial   S2s:', np.array(S2s))
	print('Initial iotas:', np.array(iotas))
	print('%s deterministic and %s random initial parameters\n'%(len(initial_params), max(0, Nwave-len(initial_params))))
	
	#add random parameter to complete initial_params
	initial_params = np.array(initial_params)
	if len(initial_params)<Nwave:
		initial_params = np.append(initial_params, np.random.uniform(params_low, params_high, size=(Nwave-len(initial_params),len(params_low))), axis=0)		
	
	#return the initial parameters	
	return initial_params

#function to generate N waveforms such that their ROB or EIM representation error is larger than tol
def generate_params_over_tol(tol, Nmax, Nbatch, basis_type, distance, delta_f, f_min, f_max, approximant, params_low, params_high, Nwave, Ncheckpoint, checkpoint_dict, laldict=lal.CreateDict(), nprocesses=0, basis=None, Bj=None, emp_nodes=None):

	#check if we are looking for parameters over ROB_err or EIM_err
	if Bj is not None and emp_nodes is not None: 
		ROB_err = False
		rep_err_str = 'EIM'
	else: 
		ROB_err = True
		rep_err_str = 'ROB'
		
	#try to load from checkpoint
	if 'Ngen' in checkpoint_dict:
		#load params
		all_params = np.load(basis_type+'params.npy')
	else:
		#compute the parampoints for all possible waveforms that will be generated
		all_params = np.random.uniform(params_low, params_high, size=(int(Nmax+Nbatch), len(params_low)))	
		#save them
		np.save(basis_type+'params.npy', all_params)
		#initialize relevant stuff in checkpoint_dict
		checkpoint_dict['Ngen'] = 0
		checkpoint_dict['sel_idxs'] = np.array([], dtype=int)
		checkpoint_dict['sel_rep_err'] = np.array([])

	#loop over params
	if ROB_err: use_basis = np.transpose(np.conj(basis))
	for i0params in range(checkpoint_dict['Ngen'], Nmax, Nbatch):
		
		#generate waveforms to test
		cidxs = np.arange(i0params, i0params+Nbatch)
		waveforms = generate_waveforms(basis_type, distance, delta_f, f_min, f_max, approximant, laldict=laldict, nprocesses=nprocesses, params=all_params[cidxs])

		#compute the ROB representation error: <h - P_m h, h - P_m h> = <h, h> - sum_i |<e_i,h>|^2
		if ROB_err: rep_err = (np.linalg.norm(waveforms, axis=1)**2) - (np.linalg.norm(np.matmul(waveforms, use_basis), axis=1)**2)
		#otherwise, compute EIM_error
		else: rep_err = np.linalg.norm(waveforms-np.matmul(waveforms[:,emp_nodes], Bj), axis=1)**2
		
		#select only those points which have a finite representation error
		rep_err = rep_err[np.isfinite(rep_err)]
		
		#compute the temporary rep_err = [rep_err, sel_rep_err]
		tmp_rep_err = np.append(rep_err, checkpoint_dict['sel_rep_err'])
		tmp_sel_idxs = np.append(cidxs, checkpoint_dict['sel_idxs'])
		
		#sort tmp_rep_err from largest to smallest
		idxs_sort = np.flip(np.argsort(tmp_rep_err))
		
		#pick up to the Nwave largest
		checkpoint_dict['sel_rep_err'] = tmp_rep_err[idxs_sort][:min(Nwave, len(tmp_rep_err))]
		checkpoint_dict['sel_idxs'] = tmp_sel_idxs[idxs_sort][:min(Nwave, len(tmp_sel_idxs))]
		
		print(current_time(), 'Computed: %s/%s -> With %s err>%.3g: %s/%s -> max %s err: %.3g -> min %s err: %.3g'%(i0params+Nbatch, Nmax, rep_err_str, tol, np.sum(checkpoint_dict['sel_rep_err']>tol), Nwave, rep_err_str, checkpoint_dict['sel_rep_err'][0], rep_err_str, checkpoint_dict['sel_rep_err'][-1]))
		
		#if the waveform with the smallest error is over the tol, break the loop
		if (checkpoint_dict['sel_rep_err'][-1]>tol) and (len(checkpoint_dict['sel_rep_err'])==Nwave): break
		
		#checkpoint every Ncheckpoint waveforms computed
		if (i0params//Ncheckpoint) != ((i0params+Nbatch)//Ncheckpoint):
			checkpoint_dict['Ngen'] = i0params+Nbatch
			with open(basis_type+'checkpoint_dict.pickle', 'wb') as handle: pickle.dump(checkpoint_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

	#make sure to checkpoint also at the end
	checkpoint_dict['Ngen'] = Nmax+1
	with open(basis_type+'checkpoint_dict.pickle', 'wb') as handle: pickle.dump(checkpoint_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

	#return the parameter points over threashold
	return all_params[checkpoint_dict['sel_idxs']], checkpoint_dict

#function to compute the matrix M_AB that is going to be diagonalized
def M_AB(basis_type, distance, delta_f, f_min, f_max, approximant, params, Nbatch, laldict=lal.CreateDict(), nprocesses=0, basis=None):
	
	#find how many batches we have to do
	BatchNum = int(np.ceil(len(params)/Nbatch))
	
	#compute the indexes of the batches
	Blims = np.arange(0, len(params)+1, len(params)//BatchNum)
	Blims[-1] = len(params)
	
	#compute waveforms of first batch
	waveforms1 = generate_waveforms(basis_type, distance, delta_f, f_min, f_max, approximant, laldict=laldict, nprocesses=nprocesses, params=params[Blims[0]:Blims[1]])
	
	#loop over batches
	Matrix = np.zeros((len(params),len(params)), dtype=waveforms1.dtype)
	for ib1 in range(BatchNum):
		
		#compute diagonal part of the matrix M_AB = <h_A, h_B> 
		Matrix[Blims[ib1]:Blims[ib1+1], Blims[ib1]:Blims[ib1+1]] = np.matmul(np.conj(waveforms1), np.transpose(waveforms1))
		
		#compute the matrix M_AB = <h_A - P h_A, h_B - P h_B> = <h_A, h_b> - sum_k <h_A, e_k> <e_k, h_B>
		if basis is not None:
			#compute <e_k, h_B1>
			ek_hA1 = np.matmul(np.conj(basis), np.transpose(waveforms1))
			#compute the matrix M_AB
			Matrix[Blims[ib1]:Blims[ib1+1], Blims[ib1]:Blims[ib1+1]] = Matrix[Blims[ib1]:Blims[ib1+1], Blims[ib1]:Blims[ib1+1]] - np.matmul(np.conj(np.transpose(ek_hA1)), ek_hA1)	
	
		#compute off-diagonal parts
		for ib2 in range(BatchNum-1, ib1, -1):
			
			#compute waveforms of second batch
			waveforms2 = generate_waveforms(basis_type, distance, delta_f, f_min, f_max, approximant, laldict=laldict, nprocesses=nprocesses, params=params[Blims[ib2]:Blims[ib2+1]])
			
			#compute Matrix M_AB
			Matrix[Blims[ib1]:Blims[ib1+1], Blims[ib2]:Blims[ib2+1]] = np.matmul(np.conj(waveforms1), np.transpose(waveforms2))
			
			#compute the matrix M_AB = <h_A - P h_A, h_B - P h_B> = <h_A, h_b> - sum_k <h_A, e_k> <e_k, h_B>
			if basis is not None:
				#compute <e_k, h_B2>
				ek_hA2 = np.matmul(np.conj(basis), np.transpose(waveforms2))

				#compute the matrix M_AB
				Matrix[Blims[ib1]:Blims[ib1+1], Blims[ib2]:Blims[ib2+1]] = Matrix[Blims[ib1]:Blims[ib1+1], Blims[ib2]:Blims[ib2+1]] - np.matmul(np.conj(np.transpose(ek_hA1)), ek_hA2)
			
			#use that matrix is hermitian
			Matrix[Blims[ib2]:Blims[ib2+1], Blims[ib1]:Blims[ib1+1]] = np.transpose(np.conj(Matrix[Blims[ib1]:Blims[ib1+1], Blims[ib2]:Blims[ib2+1]]))
			
		#since waveforms2 now is waveforms_{ib1+1}, we can set waveforms1 to waveforms2 for next iteration
		if BatchNum>1: waveforms1 = waveforms2
	
	return Matrix

#Compute the orthogonal part of vecs w.r.t basis
def ortho_to_basis(vecs, basis):

	#ortogonalize with Gramm-Schmidt
	for base in basis: vecs = vecs - np.tensordot(np.dot(vecs, np.conj(base)), base, axes=0)

	#return the normalized part of the vectors orthogonal to basis
	return vecs/(np.linalg.norm(vecs, axis=1)[...,np.newaxis])

#function to compute the eigenvectors in the waveform domain given the parameters and the eigenvectos
def compute_eigen_vecs_wf(eigen_vecs, params, Nbatch, basis_type, distance, delta_f, f_min, f_max, approximant, laldict=lal.CreateDict(), basis=None, nprocesses=0):

	#find how many batches we have to do
	BatchNum = int(np.ceil(len(params)/Nbatch))

	#compute the indexes of the batches
	Blims = np.arange(0, len(params)+1, len(params)//BatchNum)
	Blims[-1] = len(params)

	#loop over batches
	for ib in range(BatchNum):
		#compute waveforms in this batch
		waveforms = generate_waveforms(basis_type, distance, delta_f, f_min, f_max, approximant, laldict=laldict, nprocesses=nprocesses, params=params[Blims[ib]:Blims[ib+1]])
		#compute the contribution of these waveforms to the waveform domain eigenvectors
		if ib==0: eigen_vecs_wf = np.matmul(np.transpose(eigen_vecs[Blims[ib]:Blims[ib+1]]), waveforms)
		else: eigen_vecs_wf = eigen_vecs_wf + np.matmul(np.transpose(eigen_vecs[Blims[ib]:Blims[ib+1]]), waveforms)

	#normalize the eigenvectors
	eigen_vecs_wf = eigen_vecs_wf/np.linalg.norm(eigen_vecs_wf, axis=1)[...,np.newaxis]

	#if a basis is given, compute orthogonal part of vecs w.r.t basis
	if basis is not None: eigen_vecs_wf = ortho_to_basis(eigen_vecs_wf, basis)

	#return normalized eigenvectors in waveform domain
	return eigen_vecs_wf

#function to generate the eigenvalues and eigenvectors of M_AB=<h_A - P h_A, h_B - P h_B> sorted by their maximum contributions from larger to smaller
def sorted_eig_M_AB(basis_type, distance, delta_f, f_min, f_max, approximant, params, Nbatch, laldict=lal.CreateDict(), nprocesses=0, basis=None):
	
	#compute eigenvalues and eigenvectors of the matrix M_AB = <h_A - P h_A, h_B - P h_B>
	eigen_vals, eigen_vecs = np.linalg.eigh(M_AB(basis_type, distance, delta_f, f_min, f_max, approximant, params, Nbatch, laldict=laldict, nprocesses=nprocesses, basis=basis))	

	#consider only the eigenvalues which are larger than 0
	idxs_pos = (eigen_vals > 0)
	eigen_vals = eigen_vals[idxs_pos]
	eigen_vecs = eigen_vecs[:, idxs_pos]
	
	#sort max_A lambda_B*|E_AB|**2 from larger to smaller
	idxs_sort = np.flip(np.argsort(np.amax(eigen_vals[np.newaxis,:]*(np.abs(eigen_vecs)**2), axis=0)))
	
	#return the eigenvalues and eigenvectors sorted accordingly
	return eigen_vals[idxs_sort], eigen_vecs[:,idxs_sort]
	
#function to compute the ROB that describes the waveforms with params with a tolerance better than tol
def EIG_ROB(tol, basis_type, distance, delta_f, f_min, f_max, approximant, params, Nbatch, laldict=lal.CreateDict(), nprocesses=0, basis=None, generate_EIM=False, checkpoint_dict=None, max_rounds=100, max_weight=1e4, train_extra='sum'):

	#if there are parameters over tolerance, change basis
	if len(params) != 0:
			
		#compute eigenvalues and eigenvectors of the matrix M_AB = <h_A - P h_A, h_B - P h_B> sorted by their maximum contributions from larger to smaller
		eigen_vals, eigen_vecs = sorted_eig_M_AB(basis_type, distance, delta_f, f_min, f_max, approximant, params, Nbatch, laldict=laldict, nprocesses=nprocesses, basis=basis)
		
		#compute contribution_AB = lambda_B*|E_AB|**2
		contributions = eigen_vals[np.newaxis,:]*(np.abs(eigen_vecs)**2)
		
		#find how many contributions have to be added to make sigma<tol
		current_sigma = np.sum(contributions, axis=1)
		for icontr in range(len(eigen_vals)):
			
			#update the current sigma
			current_sigma = current_sigma - contributions[:, icontr]
			
			#if we have reached the desired tolerance, stop
			if np.amax(current_sigma)<tol:
				#count the number of selected eigenvalues
				N_sel = icontr+1
				break

		#get the eigenvectors (they are orthonormal)
		eigen_vecs_sel = eigen_vecs[:, :N_sel]
		
		#compute the normalized eigenvectors in waveform domain, which is the new basis
		new_basis = compute_eigen_vecs_wf(eigen_vecs_sel, params, Nbatch, basis_type, distance, delta_f, f_min, f_max, approximant, laldict=laldict, basis=basis, nprocesses=nprocesses)
			
		#if there was a previous basis, append the new basis onto it
		if basis is not None:
			print('\n'+ current_time(), 'Number of new basis elements added:', len(new_basis)) 
			basis = np.append(basis, new_basis, axis=0)
		else: basis = new_basis
	#if we do not have to generate the EIM, save basis and return
	if not generate_EIM:
		np.save(basis_type+'basis.npy', basis)
		return basis 		
	#otherwise, generate EIM
	else:
		#load all the parameters that were selected
		all_params = np.load(basis_type+'params.npy', mmap_mode='r') #leave it in disk since we are only going to take a small part
		sel_params = all_params[checkpoint_dict['sel_idxs']]

		#compute eigenvalues and eigenvectors of the matrix M_AB = <h_A - P h_A, h_B - P h_B> sorted by their maximum contributions from larger to smaller
		eigen_vals, eigen_vecs = sorted_eig_M_AB(basis_type, distance, delta_f, f_min, f_max, approximant, sel_params, Nbatch, laldict=laldict, nprocesses=nprocesses, basis=basis)

		#compute a maximum of Nbatch eigenvectors in the waveform domain
		max_idx = min(len(eigen_vals), Nbatch)
		print('\n'+ current_time(), 'Using %s/%s eigenvalues for EIM generation -> maximum ROB error missed: %.3g'%(max_idx, len(eigen_vals), np.amax(np.sum(eigen_vals[np.newaxis,max_idx:]*(np.abs(eigen_vecs[:, max_idx:])**2), axis=1))), '\n')
		eigen_vecs_train_wf = compute_eigen_vecs_wf(eigen_vecs[:, :max_idx], sel_params, Nbatch, basis_type, distance, delta_f, f_min, f_max, approximant, laldict=laldict, basis=basis, nprocesses=nprocesses)

		#compute the Empirical Interpolation model
		emp_nodes = create_EIM_walk_train(basis, eigen_vals[:max_idx], eigen_vecs[:, :max_idx], eigen_vecs_train_wf, max_rounds=max_rounds, max_weight=max_weight, train_extra=train_extra)
		
		#compute Bj and save the EIM
		Bj, checkpoint_dict = Compute_And_Save_EIM(basis, emp_nodes, basis_type, f_min, f_max, delta_f, checkpoint_dict)

		#return the new basis and EIM
		return basis, Bj, emp_nodes, checkpoint_dict


#algorithm based on ortogonalizing the rows of A
def create_EIM_orto(basis):

	#compute norms of columns
	col_norms2 = np.linalg.norm(basis, axis=0)**2

	#loop over basis elements
	emp_nodes = list()
	orto_basis = list()
	for ib in range(len(basis)-1):
		
		#choose the emp_node as the column with the largest norm
		emp_nodes.append(np.argmax(col_norms2))

		#add the selected column to the ortogonal basis with Gramm-Schmidt
		vec = basis[:, emp_nodes[-1]]
		for base in orto_basis: vec = vec - np.dot(vec, np.conj(base))*base
		orto_basis.append(vec/np.linalg.norm(vec))

		#update the norm of the ortogonal part of the columns: |v_a|^2 = |v_a|^2 - |<w_ib,v_a>|^2
		col_norms2 = col_norms2 - np.abs(np.dot(np.conj(orto_basis[-1]), basis))**2

	#add the last basis element (no need to ortogonalize it)
	emp_nodes.append(np.argmax(col_norms2))

	#sort the emp_nodes
	emp_nodes = np.sort(emp_nodes)
	
	#return the emp_nodes and the final invA
	return emp_nodes

#algorithm based on random walk around intial solution, to minimize froebenius norm of inverse
def create_EIM_walk_Frob(basis, max_rounds=100):

	#compute the initial EIM model
	emp_nodes = create_EIM_orto(basis)
	
	#compute the initial inverse of the matrix
	invA = np.linalg.inv(basis[:, emp_nodes])
	
	#compute the corresponding ||A^-1||_F and print it
	Finv_min = np.linalg.norm(invA)**2
	print(current_time(), '%s/%s -> ||A^-1||_F = %s'%(0, max_rounds, np.sqrt(Finv_min)))
		
	#make the necessary number of rounds around the columns
	for iround in range(max_rounds):
			
		#check whether emp_nodes stays constant in this round
		const_emp = True
		#loop over emp_nodes
		for iemp in range(len(emp_nodes)):
			#keep track of whether emp_nodes changes for this iemp
			change_this_iemp = False
			#keep track if we didn't initialize this iemp
			not_initialized_this_iemp=True
			#loop over nodes we can walk to
			old_emp_node = emp_nodes[iemp] 
			for demp in [-1,1]:
				#try to walk in the direction marked by demp
				new_emp_node = old_emp_node + demp			
				while True:

					#check if this node is out of range
					if (new_emp_node<0) or (new_emp_node>=len(basis[0, :])): break
					#check if this node is already in emp_nodes	
					if new_emp_node in emp_nodes: break
					
					#if needed, precompute what we will need to update ||A^-1||_F for this iemp
					if not_initialized_this_iemp:
						invA_iemp_invAH = np.dot(np.conj(invA), invA[iemp, :])
						norm_invA_iemp = np.abs(invA_iemp_invAH[iemp])
						F_invA = Finv_min
						not_initialized_this_iemp = False
								
					#update the value of ||A^-1||_F^2
					invA_new_emp = np.dot(invA, basis[:, new_emp_node])
					Finv_c = F_invA + (norm_invA_iemp/(np.abs(invA_new_emp[iemp])**2))*(1 + (np.linalg.norm(invA_new_emp)**2)) - 2*np.real(np.dot(invA_iemp_invAH, invA_new_emp)/invA_new_emp[iemp])
					
					#if this one is better, keep it
					if Finv_c<Finv_min:
						Finv_min = Finv_c
						emp_nodes[iemp] = new_emp_node
						const_emp = False
						change_this_iemp = True
						#continue walking in this direction
						new_emp_node = new_emp_node + demp
					#otherwise stop walking in this direction
					else: break
		
			#update the inverse of A if it changed at this iemp
			if change_this_iemp:
				#update the inverse of A by taking into account that the iemp column of A has been updated to basis[:, emp_nodes[iemp]]
				#this is O(n^2) instead of O(n^3) like directly computing the inverse
				invA_new_emp = np.dot(invA, basis[:, emp_nodes[iemp]])
				invA_tmp = invA - np.tensordot(invA_new_emp/invA_new_emp[iemp], invA[iemp, :], axes=0)
				invA_tmp[iemp, :] = invA[iemp, :]/invA_new_emp[iemp]
				invA = invA_tmp
		
		print(current_time(), '%s/%s -> ||A^-1||_F = %s'%(iround+1, max_rounds, np.sqrt(Finv_min)))
		
		#if emp_nodes did not change in this round, break loop
		if const_emp: break		
	
	#sort the emp_nodes
	emp_nodes = np.sort(emp_nodes)

	#return the emp_nodes
	return emp_nodes

#algorithm based on random walk around intial solution, to minimize the maximum EIM error
def create_EIM_walk_train(basis, eigen_vals, eigen_vecs, eigen_vecs_train_wf, max_rounds=100, max_weight=1e4, train_extra='sum'):
	
	#normalize eigen_vecs_train_wf by the sqrt of lambda
	eigen_vecs_train_wf = np.sqrt(eigen_vals[:,np.newaxis])*eigen_vecs_train_wf
	
	#compute the average value sqrt(E[lambda u_a^* u_a])
	col_norm_eigen_vecs_train_wf = np.linalg.norm(eigen_vecs_train_wf, axis=0)

	#add a small number to col_norm_eigen_vecs_train_wf to make the maximum relative weight equal to max_weight
	add_num = max(((np.amax(col_norm_eigen_vecs_train_wf)**2)-((max_weight*np.amin(col_norm_eigen_vecs_train_wf))**2))/((max_weight**2)-1), 0)
	if add_num>0:
		col_norm_eigen_vecs_train_wf = np.sqrt(col_norm_eigen_vecs_train_wf**2 + add_num)

	#compute the initial EIM model
	emp_nodes = create_EIM_walk_Frob(np.divide(basis, col_norm_eigen_vecs_train_wf[np.newaxis,:]), max_rounds=max_rounds)

	if train_extra is not None:

		#precompute the sigma_ROB factors needed
		sigma_ROB = np.sum(eigen_vals[np.newaxis,:]*(np.abs(eigen_vecs)**2), axis=1)
		sum_sigma_ROB = np.sum(eigen_vals)

		#compute the initial inverse of the matrix and the initial lu/lu_EIM
		invA = np.linalg.inv(basis[:, emp_nodes])
		lu = eigen_vecs_train_wf[:, emp_nodes]
		lu_EIM = np.matmul(lu, invA)			
		if train_extra == 'sum':
			#compute the sum of EIM errors
			train_EIM_err_min = sum_sigma_ROB + (np.linalg.norm(lu_EIM)**2)
			max_EIM_err = np.amax(sigma_ROB + (np.linalg.norm(np.matmul(np.conj(eigen_vecs), lu_EIM), axis=1)**2))
			print(current_time(), '%s/%s -> sum EIM err = %s -> max EIM err = %.3g'%(0, max_rounds, train_EIM_err_min, max_EIM_err))
		elif train_extra == 'max':
			#compute the maximum EIM error 
			train_EIM_err_min = np.amax(sigma_ROB + (np.linalg.norm(np.matmul(np.conj(eigen_vecs), lu_EIM), axis=1)**2))
			print(current_time(), '%s/%s -> max EIM err = %s'%(0, max_rounds, train_EIM_err_min))

		#make the necessary number of rounds around the columns
		for iround in range(max_rounds):
			#check whether emp_nodes stays constant in this round
			const_emp = True
			#loop over emp_nodes
			for iemp in range(len(emp_nodes)):
				#keep track of whether emp_nodes changes for this iemp
				change_this_iemp = False
				#keep track if we didn't initialize this iemp
				not_initialized_this_iemp=True
				#loop over nodes we can walk to
				old_emp_node = emp_nodes[iemp] 
				for demp in [-1,1]:
					#try to walk in the direction marked by demp
					new_emp_node = old_emp_node + demp			
					while True:
						#check if this node is out of range
						if (new_emp_node<0) or (new_emp_node>=len(basis[0, :])): break
						#check if this node is already in emp_nodes	
						if new_emp_node in emp_nodes: break

						#update the values of train_EIM_err_c using previous info
						invA_new_emp = np.dot(invA, basis[:, new_emp_node])
						if train_extra == 'sum':
						
							#if needed, precompute what we will need to update ||A^-1||_F for this iemp
							if not_initialized_this_iemp:
								lu_EIM_invAiemp = np.dot(np.conj(lu_EIM), invA[iemp, :])
								norm_invA_iemp = np.linalg.norm(invA[iemp, :])**2
								train_EIM_err_ref = train_EIM_err_min
								not_initialized_this_iemp = False
							
							#compute alpha_A
							alpha_A = (eigen_vecs_train_wf[:, new_emp_node] - np.dot(lu, invA_new_emp))/invA_new_emp[iemp]
							#compute the sum of EIM errors
							train_EIM_err_c = train_EIM_err_ref + ((np.linalg.norm(alpha_A)**2)*norm_invA_iemp) + (2*np.real(np.dot(alpha_A, lu_EIM_invAiemp)))
							
						elif train_extra == 'max':
							#update the value of lu_EIM = np.matmul(eigen_vecs_train_wf[:, emp_nodes], invA) using the previous value
							lu_EIM_tmp = lu_EIM + np.tensordot(eigen_vecs_train_wf[:, new_emp_node] - np.dot(lu, invA_new_emp), invA[iemp, :]/invA_new_emp[iemp], axes=0)
							#compute the maximum EIM error
							train_EIM_err_c = np.amax(sigma_ROB + (np.linalg.norm(np.matmul(np.conj(eigen_vecs), lu_EIM_tmp), axis=1)**2))

						#if this one is better, keep it
						if train_EIM_err_c<train_EIM_err_min:
							train_EIM_err_min = train_EIM_err_c
							emp_nodes[iemp] = new_emp_node
							const_emp = False
							change_this_iemp = True
							#continue walking in this direction
							new_emp_node = new_emp_node + demp
						#otherwise stop walking in this direction
						else: break

				#update lu_EIM and the inverse of A if it changed at this iemp
				if change_this_iemp:
					#update lu_EIM before invA (it uses old invA)
					invA_new_emp = np.dot(invA, basis[:, emp_nodes[iemp]])
					lu_EIM = lu_EIM + np.tensordot(eigen_vecs_train_wf[:, emp_nodes[iemp]] - np.dot(lu, invA_new_emp), invA[iemp, :]/invA_new_emp[iemp], axes=0)
					lu[:, iemp] = eigen_vecs_train_wf[:, emp_nodes[iemp]]
					#update the inverse of A by taking into account that the iemp column of A has been updated to basis[:, emp_nodes[iemp]]
					#this is O(n^2) instead of O(n^3) like directly computing the inverse
					invA_tmp = invA - np.tensordot(invA_new_emp/invA_new_emp[iemp], invA[iemp, :], axes=0)
					invA_tmp[iemp, :] = invA[iemp, :]/invA_new_emp[iemp]
					invA = invA_tmp

			if train_extra == 'sum': 
				max_EIM_err = np.amax(sigma_ROB + (np.linalg.norm(np.matmul(np.conj(eigen_vecs), lu_EIM), axis=1)**2))
				print(current_time(), '%s/%s -> sum EIM err = %s -> max EIM err = %.3g'%(iround+1, max_rounds, train_EIM_err_min, max_EIM_err))
			elif train_extra == 'max': print(current_time(), '%s/%s -> max EIM err = %s'%(iround+1, max_rounds, train_EIM_err_min))

			#if emp_nodes did not change in this round, break loop
			if const_emp: break		
		
		#sort the emp_nodes
		emp_nodes = np.sort(emp_nodes)

	#print the 22 norm of invA
	print(current_time(), 'Maximum increase in error because of EIM: ||inv(A)||_2^2 = %.1f'%(norm22(np.linalg.inv(basis[:, emp_nodes]))))

	#return the emp_nodes
	return emp_nodes


#function to compute the square of the matrix spectral norm
def norm22(Matrix):

	#return maximum eigenvalue of M^H M
	return np.amax(np.linalg.eigvalsh(np.matmul(np.transpose(np.conj(Matrix)), Matrix)))

#compute the Empirical interpolation and return the Bj, emp_nodes and norm22invA
def Compute_And_Save_EIM(basis, emp_nodes, basis_type, f_min, f_max, delta_f, checkpoint_dict):
	
	#compute the matrix invAT
	invAT = np.linalg.inv(basis[:, emp_nodes])

	print('\n'+ current_time(), 'Saving EIM and basis of shape', basis.shape, '\n')	
	#compute and save emp_nodes, fnodes, Bj and basis
	np.save('emp_nodes_'+basis_type+'.npy', emp_nodes)
	freqs = np.arange(f_min,f_max,delta_f)
	np.save('fnodes_'+basis_type+'.npy', freqs[emp_nodes])
	Bj = np.matmul(invAT, basis)
	np.save('B_'+basis_type+'.npy', Bj)		
	np.save(basis_type+'basis.npy', basis)

	#compute the square of the two-norm of A^-1
	checkpoint_dict['norm22invA'] = norm22(invAT)
	with open(basis_type+'checkpoint_dict.pickle', 'wb') as handle: pickle.dump(checkpoint_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
	
	#return Bj, emp_nodes
	return Bj, checkpoint_dict

#function to compute the EIM error of waveforms with params
def compute_EIM_err(Bj, emp_nodes, basis_type, distance, delta_f, f_min, f_max, approximant, params, Nbatch, laldict=lal.CreateDict(), nprocesses=0):

	#initialize EIM
	EIM_err = np.array([])

	#find how many batches we have to do
	BatchNum = int(np.ceil(len(params)/Nbatch))

	#compute the indexes of the batches
	Blims = np.arange(0, len(params)+1, len(params)//BatchNum)
	Blims[-1] = len(params)

	#loop over batches
	for ib in range(BatchNum):
		#compute waveforms in this batch
		waveforms = generate_waveforms(basis_type, distance, delta_f, f_min, f_max, approximant, laldict=laldict, nprocesses=nprocesses, params=params[Blims[ib]:Blims[ib+1]])
		#append the EIM error
		EIM_err = np.append(EIM_err, np.linalg.norm(waveforms-np.matmul(waveforms[:,emp_nodes], Bj), axis=1)**2)

	return EIM_err

#compute the matrix U^EIM_AB = sqrt(lambda_A lambda_B) <u_A - I[u_A], u_B - I[u_B]>
def U_EIM_AB(eigen_vals, eigen_vecs, basis, Bj, emp_nodes, basis_type, distance, delta_f, f_min, f_max, approximant, params, Nbatch, laldict=lal.CreateDict(), nprocesses=0):
	
	#we can compute the transpose of the A matrix as
	invAT = np.linalg.inv(basis[:, emp_nodes])
	
	#find how many batches we have to do
	BatchNum = int(np.ceil(len(params)/Nbatch))

	#compute the indexes of the batches
	Blims = np.arange(0, len(params)+1, len(params)//BatchNum)
	Blims[-1] = len(params)

	#loop over batches
	#compute (A^-1 h_A[emp_nodes] - V^H h_A)
	for ib in range(BatchNum):

		#compute waveforms in this batch
		waveforms = generate_waveforms(basis_type, distance, delta_f, f_min, f_max, approximant, laldict=laldict, nprocesses=nprocesses, params=params[Blims[ib]:Blims[ib+1]])
		
		#compute (A^-1 h_A[emp_nodes] - V^H h_A) and append it to h_EIM
		h_EIM_tmp = np.matmul(waveforms[:,emp_nodes], invAT) - np.matmul(waveforms, np.conj(np.transpose(basis)))
		if ib==0:  h_EIM = h_EIM_tmp
		else: h_EIM = np.append(h_EIM, h_EIM_tmp, axis=0)
			
	#compute sqrt(lambda_A) u^EIM_A[emp_nodes]
	lu_EIM = np.matmul(np.transpose(eigen_vecs), h_EIM)
	
	#Now compute the Matrix U^EIM_AB = sqrt(lambda_A lambda_B)*(d_AB + <I[u_A], I[u_B]>)
	return np.diag(eigen_vals) + np.matmul(np.conj(lu_EIM), np.transpose(lu_EIM))

#function to compute the EIM that describes the waveforms with params with a tolerance better than tol
def EIG_EIM(EIM_tol, basis, Bj, emp_nodes, checkpoint_dict, basis_type, distance, delta_f, f_min, f_max, approximant, params, Nbatch, laldict=lal.CreateDict(), max_rounds=100, max_weight=1e4, train_extra='sum', n_check_point=25, nprocesses=0):

	#if there are no parameters, just return basis
	if len(params) == 0: return basis
	
	#compute eigenvalues and eigenvectors of the matrix M_AB = <h_A - P h_A, h_B - P h_B> sorted by their maximum contributions from larger to smaller
	eigen_vals, eigen_vecs = sorted_eig_M_AB(basis_type, distance, delta_f, f_min, f_max, approximant, params, Nbatch, laldict=laldict, nprocesses=nprocesses, basis=basis)

	#compute the maximum sigma ROB that we are missing when preselecting eigenvalues and print it
	N_presel = min(Nbatch, len(eigen_vals))
	print('\n'+ current_time(), 'Preselected %s/%s eigenvalues for further analysis -> maximum ROB error missed: %.3g'%(N_presel, len(eigen_vals), np.amax(np.sum(eigen_vals[np.newaxis,N_presel:]*(np.abs(eigen_vecs[:, N_presel:])**2), axis=1))), '\n')

	#preselect Nbatch eigenvalues in waveform domain
	eigen_vals = eigen_vals[:N_presel]
	eigen_vecs = eigen_vecs[:, :N_presel]
	eigen_vecs_wf = compute_eigen_vecs_wf(eigen_vecs, params, Nbatch, basis_type, distance, delta_f, f_min, f_max, approximant, laldict=laldict, basis=basis, nprocesses=nprocesses)
	
	#Run the algorithm until all waveforms are fitted by the EIM, we have to add more contributions
	count_no_checkpoint = 0
	while True:

		#compute ROB error
		sigma_ROB = np.sum(eigen_vals[np.newaxis,:]*(np.abs(eigen_vecs)**2), axis=1)
		
		#we can compute the transpose of the A matrix as
		invAT = np.linalg.inv(basis[:, emp_nodes])
		#compute sqrt(lambda_A) u^EIM_A[emp_nodes]
		lu_EIM = np.sqrt(eigen_vals[:,np.newaxis])*np.matmul(eigen_vecs_wf[:,emp_nodes], invAT)
		#compute the EIM error
		EIM_err = sigma_ROB + (np.linalg.norm(np.matmul(np.conj(eigen_vecs), lu_EIM), axis=1)**2)
		print(current_time(), 'With EIMerr>%.3g: %s/%s | maxEIMerr: %.3g | minEIMerr: %.3g | sumEIMerr: %.3g'%(EIM_tol, np.sum(EIM_err>EIM_tol), len(params), np.amax(EIM_err), np.amin(EIM_err), np.sum(EIM_err)))

		#if there is no EIM_err over threashold, stop the loop
		if np.amax(EIM_err)<EIM_tol: break
		
		#find the eigenvector with the largest contribution to the EIM error: max_{A,B} (lambda_B |E_{AB}|^2 (1 + <I[u_B], I[u_B]>))
		idx_sel = np.argmax(np.amax(np.abs(eigen_vecs)**2, axis=0)*(eigen_vals + (np.linalg.norm(lu_EIM, axis=1)**2)))

		#append the new basis elements into the previous basis
		basis = np.vstack((basis, eigen_vecs_wf[idx_sel]))
		print('\n'+ current_time(), 'Added the eigenvector', idx_sel,'-> Total basis elements:', len(basis))
		 
		#update eigen_vals and eigen_vecs to be the non-selected ones
		idxs_nosel = np.full(len(eigen_vals), True)
		idxs_nosel[idx_sel] = False
		eigen_vals = eigen_vals[idxs_nosel]
		eigen_vecs = eigen_vecs[:, idxs_nosel]
		eigen_vecs_wf = eigen_vecs_wf[idxs_nosel,:]

		#compute the Empirical Interpolation model
		emp_nodes = create_EIM_walk_train(basis, eigen_vals, eigen_vecs, eigen_vecs_wf, max_rounds=max_rounds, max_weight=max_weight, train_extra=train_extra)
		
		#add one to the counter of elements added before checkpoint  
		count_no_checkpoint = count_no_checkpoint + 1
		#checkpoint if required
		if count_no_checkpoint >= n_check_point:
			#compute Bj and save the EIM
			Bj, checkpoint_dict = Compute_And_Save_EIM(basis, emp_nodes, basis_type, f_min, f_max, delta_f, checkpoint_dict)
			#restart checkpoint counter
			count_no_checkpoint = 0

	#return the new basis and EIM
	Bj, checkpoint_dict = Compute_And_Save_EIM(basis, emp_nodes, basis_type, f_min, f_max, delta_f, checkpoint_dict)
	return basis, Bj, emp_nodes, checkpoint_dict

#function to run EigROQ fully
def EigROQ(mc_low=None, mc_high=None, q_low=None, q_high=None, s1sphere_low=None, s1sphere_high=None, s2sphere_low=None, s2sphere_high=None, iota_low=None, iota_high=None, phiref_low=None, phiref_high=None, seglen=None, f_min=20, f_max=1024, distance=10*lal.lal.PC_SI*1.0e6, laldict=lal.CreateDict(), approximant=lalsimulation.IMRPhenomPv2, basis_type='linear', Nwave=20000, Nits=2, Nmax0=1e6, Nmaxf=1e7, ROB_tol0=1e-2, ROB_tolf=1e-6, EIM_tol=1e-4, Nbatch=5000, Ncheckpoint=100000, max_rounds=100, max_weight=1e4, train_extra='sum', n_check_point=25, nprocesses=0):
	
	#make sure that Nmax0 and Nmaxf are integers
	Nmax0, Nmaxf = int(Nmax0), int(Nmaxf)
	
	#basis_type has to be 'linear' or 'quadratic'
	assert basis_type in ['linear', 'quadratic']
	
	#train_extra has to be 'sum', 'max' or None
	assert train_extra in ['sum', 'max', None]
	
	#compute params low and high
	params_low = [mc_low, q_low, s1sphere_low[0], s1sphere_low[1], s1sphere_low[2], s2sphere_low[0], s2sphere_low[1], s2sphere_low[2], iota_low, phiref_low]
	params_high = [mc_high, q_high, s1sphere_high[0], s1sphere_high[1], s1sphere_high[2], s2sphere_high[0], s2sphere_high[1], s2sphere_high[2], iota_high, phiref_high]

	#compute the frequency array
	delta_f = 1/seglen
	f_max = f_max + delta_f
	freq = np.arange(f_min,f_max,delta_f)
	
	print(current_time(), 'Computing '+basis_type+' ROQ with a EIM tolerance of', EIM_tol)
	
	#Try to load a checkpoint dictionary
	try:
		with open(basis_type+'checkpoint_dict.pickle', 'rb') as handle:
			checkpoint_dict = pickle.load(handle)
		print(current_time(), 'Loaded checkpoint dictionary at iteration %s/%s'%(checkpoint_dict['ROB_it'], Nits))
		#load basis
		basis = np.load(basis_type+'basis.npy')
		
	except: 
		#if the dictionary could not be loaded, create it
		checkpoint_dict = {}
		
		#start algorithm by generating intial parameter points
		params = generate_initial_params(params_low, params_high, Nwave)

		#compute the basis
		basis = EIG_ROB(ROB_tol0, basis_type, distance, delta_f, f_min, f_max, approximant, params, Nbatch, laldict=laldict, nprocesses=nprocesses)
		
		#initialize checkpoint dict and save it
		checkpoint_dict['ROB_it'] = 0
		with open(basis_type+'checkpoint_dict.pickle', 'wb') as handle: pickle.dump(checkpoint_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

	#go over the tolerance iterations
	ROB_tols = np.geomspace(ROB_tol0, ROB_tolf, Nits+1)
	Nmaxs = np.geomspace(Nmax0, Nmaxf, Nits+1).astype(int)
	for it in range(checkpoint_dict['ROB_it'], Nits):
		
		#set the number of maximum waveform evaluations to the one corresponding to this tolerance
		Nmax = Nmaxs[it]
		
		#set the tolerance
		tol = ROB_tols[it+1]
		
		#start main loop
		while True:
					
			#print info before next iteration
			print('\n'+ current_time(), 'Number of basis elements:', len(basis))
			print(current_time(), 'Work with tol=%.3g, Nmax=%s\n'%(tol, Nmax))
			
			#generate parameter points above ROB tolerance
			sel_params, checkpoint_dict = generate_params_over_tol(tol, Nmax, Nbatch, basis_type, distance, delta_f, f_min, f_max, approximant, params_low, params_high, Nwave, Ncheckpoint, checkpoint_dict, laldict=laldict, nprocesses=nprocesses, basis=basis)
			params_over_tol = sel_params[checkpoint_dict['sel_rep_err']>tol]
			
			#if we are in the last iteration, and the number of parameters over tolerance was smaller than Nwave, compute EIM
			if (len(params_over_tol)<Nwave) and it==(Nits-1):
				basis, Bj, emp_nodes, checkpoint_dict = EIG_ROB(tol, basis_type, distance, delta_f, f_min, f_max, approximant, params_over_tol, Nbatch, laldict=laldict, nprocesses=nprocesses, basis=basis, generate_EIM=True, checkpoint_dict=checkpoint_dict, max_rounds=max_rounds, max_weight=max_weight, train_extra=train_extra)
			#otherwise just compute the new basis to fit these waveforms with tol
			else: basis = EIG_ROB(tol, basis_type, distance, delta_f, f_min, f_max, approximant, params_over_tol, Nbatch, laldict=laldict, nprocesses=nprocesses, basis=basis)

			#delete the relevant keys from the checkpoint dict and save it
			del checkpoint_dict['Ngen'], checkpoint_dict['sel_idxs'], checkpoint_dict['sel_rep_err']
			with open(basis_type+'checkpoint_dict.pickle', 'wb') as handle: pickle.dump(checkpoint_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

			#if the number of params was smaller than Nwave, stop
			if len(params_over_tol)<Nwave: break

		#save that we are in this iteration
		checkpoint_dict['ROB_it'] = it+1
		with open(basis_type+'checkpoint_dict.pickle', 'wb') as handle: pickle.dump(checkpoint_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

	#try to load the empirical interpolation model
	try:
		Bj = np.load('B_'+basis_type+'.npy')
		emp_nodes =  np.load('emp_nodes_'+basis_type+'.npy')
		
		#make sure that it is the one corresponding to basis
		assert Bj.shape == basis.shape
		print(current_time(), 'Loaded Empirical interpolation model with shape', Bj.shape, 'and ||inv(A)||_2^2 = %.1f'%(checkpoint_dict['norm22invA']))
	except: pass
		
	#refine to get a tolerance better than tol even after Empirical Interpolation
	while True:
		
		print(current_time(), 'Work with tol=%.3g, Nmax=%s\n'%(EIM_tol, Nmaxf))
		
		#generate waveforms above EIM tolerance
		sel_params, checkpoint_dict = generate_params_over_tol(EIM_tol, Nmaxf, Nbatch, basis_type, distance, delta_f, f_min, f_max, approximant, params_low, params_high, Nwave, Ncheckpoint, checkpoint_dict, nprocesses=nprocesses, laldict=laldict, Bj=Bj, emp_nodes=emp_nodes)
		
		#number of parameters over tol
		N_over_tol = np.sum(checkpoint_dict['sel_rep_err']>EIM_tol)
		if N_over_tol==0: break
		
		#compute the new EIM to fit these waveforms with tol	
		basis, Bj, emp_nodes, checkpoint_dict = EIG_EIM(EIM_tol, basis, Bj, emp_nodes, checkpoint_dict, basis_type, distance, delta_f, f_min, f_max, approximant, sel_params, Nbatch, laldict=laldict, max_rounds=max_rounds, max_weight=max_weight, train_extra=train_extra, n_check_point=n_check_point, nprocesses=nprocesses)

		#delete the relevant keys from the checkpoint dict and save it
		del checkpoint_dict['Ngen'], checkpoint_dict['sel_idxs'], checkpoint_dict['sel_rep_err']
		with open(basis_type+'checkpoint_dict.pickle', 'wb') as handle: pickle.dump(checkpoint_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

		#if the number of params was smaller than Nwave, stop
		if N_over_tol < Nwave: break

	#return the final Empirical Interpolation Model and Reduced Basis
	return Bj, emp_nodes, basis	

#function to create the params.dat file
def ROQparams(mc_low=None, mc_high=None, q_low=None, q_high=None, s1sphere_low=None, s1sphere_high=None, s2sphere_low=None, s2sphere_high=None, iota_low=None, iota_high=None, phiref_low=None, phiref_high=None, ecc_low=0, ecc_high=0, lambda1_low=0, lambda1_high=0, lambda2_low=0, lambda2_high=0, seglen=None, f_min=20, f_max=1024):

	#generate the ROQ parameter file to run the roq
	with open('params.dat', 'w') as ROQParams_file:
		#compute minimum component mass
		comp_min = mc_low*(q_high**(-0.6))*((1+q_high)**0.2)

		#obtain maximum and minimum of chip=max(chi1*sin(th1), (1/q)*((4+3*q)/(4*q+3))*chi2*sin(th2)) (q>1)
		if (s1sphere_high[1]>=(np.pi/2)) and (s2sphere_high[1]>=(np.pi/2)):
			chip_min = 0
			chip_max = max(s1sphere_high[0], (1/q_low)*((4+3*q_low)/(4*q_low+3))*s2sphere_high[0])
		if s1sphere_high[1]==0 and s2sphere_high[1]==0:
			chip_min = 0
			chip_max = 0
		
		#save all parameters in file
		ROQParams_file.write('#flow fhigh seglen q-min q-max comp-min chirpmass-min chirpmass-max chiL1-min chiL1-max chiL2-min chiL2-max chi_p-min chi_p-max thetaJ-min thetaJ-max alpha0-min alpha0-max ecc-low ecc-high lambda1-low lambda1-high lambda2-low lambda2-high\n')
		ROQParams_file.write('%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s'%(f_min, f_max+1/seglen, seglen, q_low, q_high, comp_min, mc_low, mc_high, -s1sphere_high[0], s1sphere_high[0], -s2sphere_high[0], s2sphere_high[0], chip_min, chip_max, iota_low, iota_high, phiref_low, phiref_high, ecc_low, ecc_high, lambda1_low, lambda1_high, lambda2_low, lambda2_high))
	
	return True

#function to compute the mismatch, defined as 1 - <h, h_EIM>/sqrt(<h,h> <h,h>_EIM)
def mismatch(wf, emp_l, B_l, emp_q, B_q, psd=None):

	#if we do not have a psd, do not weigh the inner product
	if psd is None:
		return 1 - np.sum(np.conj(wf)*np.matmul(wf[:,emp_l], B_l), axis=1)/np.sqrt(np.sum(np.matmul(np.abs(wf[:,emp_q])**2, B_q), axis=1)*np.sum(np.abs(wf)**2, axis=1))
	#otherwise, weigh the inner product by the PSD
	else:
		PSD = psd[np.newaxis,:]
		return 1 - np.sum(np.conj(wf)*np.matmul(wf[:,emp_l], B_l)/PSD, axis=1)/np.sqrt(np.sum(np.matmul(np.abs(wf[:,emp_q])**2, B_q)/PSD, axis=1)*np.sum((np.abs(wf)**2)/PSD, axis=1))

	
#function to test the ROQ
def TestROQ(mc_low=None, mc_high=None, q_low=None, q_high=None, s1sphere_low=None, s1sphere_high=None, s2sphere_low=None, s2sphere_high=None, iota_low=None, iota_high=None, phiref_low=None, phiref_high=None, seglen=None, f_min=20, f_max=1024, distance=10*lal.lal.PC_SI*1.0e6, laldict=lal.CreateDict(), approximant=lalsimulation.IMRPhenomPv2, EIM_tol=1e-4, Ntests=int(1e7), nbins=100, ROB_err_thr=1e-10, psd_name = 'aLIGOO3LowT1800545', Nbatch=5000, Ncheckpoint=100000, nprocesses=0):

	#define the psd functions
	psd_funcs = {'aLIGOO3LowT1800545': lalsimulation.SimNoisePSDaLIGOaLIGOO3LowT1800545}
	
	print(current_time(), 'Test %s points of ROQ with tol=%.3g'%(Ntests, EIM_tol))

	#compute params low and params high
	params_low = [mc_low, q_low, s1sphere_low[0], s1sphere_low[1], s1sphere_low[2], s2sphere_low[0], s2sphere_low[1], s2sphere_low[2], iota_low, phiref_low]
	params_high = [mc_high, q_high, s1sphere_high[0], s1sphere_high[1], s1sphere_high[2], s2sphere_high[0], s2sphere_high[1], s2sphere_high[2], iota_high, phiref_high]

	#compute the frequency freq
	delta_f = 1/seglen
	f_max = f_max + delta_f
	freq = np.arange(f_min,f_max,delta_f)

	#create lalseries
	lalseries = lal.CreateREAL8FrequencySeries('', lal.LIGOTimeGPS(0), 0, delta_f, lal.DimensionlessUnit, int(f_max/delta_f)+1)

	#put the psd on it
	psd_funcs[psd_name](lalseries, f_min)

	#select the frequencies above f_min
	psd = np.array(lalseries.data.data)[:-1][np.arange(0,f_max,delta_f)>=f_min]

	#load the ROB and EIM
	Model = {}
	for basis_type in ['linear', 'quadratic']:
		Model['basis'+basis_type] = np.load(basis_type+'basis.npy')
		Model['B_'+basis_type] = np.load('B_'+basis_type+'.npy')
		Model['emp_nodes_'+basis_type] = np.load('emp_nodes_'+basis_type+'.npy')
		
		#print some info
		print(current_time(), 'Number of '+basis_type+' basis elements:', len(Model['basis'+basis_type]))

	#try to load the parameters and the test_dict, otherwise, create them
	try:
		#load parameters
		all_params = np.load('test_params.npy')
		#load test_dict
		with open('test_dict.pickle', 'rb') as handle: test_dict = pickle.load(handle)
	except:
		#compute the parampoints for all possible waveforms that will be generated
		all_params = np.random.uniform(params_low, params_high, size=(int(Ntests+Nbatch), len(params_low)))
		
		#save them
		np.save('test_params.npy', all_params)

		#make a dictionary to store all the errors
		test_dict = {}
		for basis_type in ['linear', 'quadratic']:
			test_dict['ROB_err'+basis_type] = np.array([])
			test_dict['EIM_err'+basis_type] = np.array([])
		test_dict['mismatch'] = np.array([])
		test_dict['mismatch_psd'] = np.array([])
		
		#save it
		with open('test_dict.pickle', 'wb') as handle: pickle.dump(test_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

	#loop over params
	for i0params in range(len(test_dict['ROB_errlinear']), Ntests, Nbatch):

		#generate linear waveforms to test
		waveforms = generate_waveforms('linear', distance, delta_f, f_min, f_max, approximant, laldict=laldict, nprocesses=nprocesses, params=all_params[i0params:(i0params+Nbatch)])
		
		#compute the mismatch <h_linear, h_EIM_linear> without psd
		test_dict['mismatch'] = np.append(test_dict['mismatch'], mismatch(waveforms, Model['emp_nodes_linear'], Model['B_linear'], Model['emp_nodes_quadratic'], Model['B_quadratic'],psd=None))

		#compute the mismatch <h_linear, h_EIM_linear> with psd
		test_dict['mismatch_psd'] = np.append(test_dict['mismatch_psd'], mismatch(waveforms, Model['emp_nodes_linear'], Model['B_linear'], Model['emp_nodes_quadratic'], Model['B_quadratic'],psd=psd))
		
		#loop over basis_types
		for basis_type in ['linear', 'quadratic']:
		
			#append the EIM and ROB error	
			if basis_type == 'quadratic': 
				waveforms = np.abs(waveforms)**2/np.linalg.norm(np.abs(waveforms)**2, axis=1)[:, np.newaxis]
				
			test_dict['EIM_err'+basis_type] = np.append(test_dict['EIM_err'+basis_type], np.linalg.norm(waveforms-np.matmul(waveforms[:,Model['emp_nodes_'+basis_type]], Model['B_'+basis_type]), axis=1)**2)
			test_dict['ROB_err'+basis_type] = np.append(test_dict['ROB_err'+basis_type], (np.linalg.norm(waveforms, axis=1)**2) - (np.linalg.norm(np.matmul(waveforms, np.transpose(np.conj(Model['basis'+basis_type]))), axis=1)**2))
			
		#print info	
		print(current_time(), 'Computed: %s/%s -> With EIM err>%.3g: [%s, %s] -> max EIM err: [%.3g, %.3g] -> max ROB err: [%.3g, %.3g] -> max mismatch: %.3g (wth psd: %.3g)'%(i0params+Nbatch, Ntests, EIM_tol, np.sum(test_dict['EIM_errlinear']>EIM_tol), np.sum(test_dict['EIM_errquadratic']>EIM_tol), np.amax(test_dict['EIM_errlinear']), np.amax(test_dict['EIM_errquadratic']), np.amax(test_dict['ROB_errlinear']), np.amax(test_dict['ROB_errquadratic']),  np.amax(np.abs(test_dict['mismatch'])),  np.amax(np.abs(test_dict['mismatch_psd']))))
		
		#checkpoint every Ncheckpoint waveforms computed
		if (i0params//Ncheckpoint) != ((i0params+Nbatch)//Ncheckpoint):
			with open('test_dict.pickle', 'wb') as handle: pickle.dump(test_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
	
	#print info outside the loop
	print('\n'+ current_time(), 'Computed: %s/%s -> With EIM err>%.3g: [%s, %s] -> max EIM err: [%.3g, %.3g] -> max ROB err: [%.3g, %.3g] -> max mismatch: %.3g (wth psd: %.3g)'%(len(test_dict['ROB_errlinear']), Ntests, EIM_tol, np.sum(test_dict['EIM_errlinear']>EIM_tol), np.sum(test_dict['EIM_errquadratic']>EIM_tol), np.amax(test_dict['EIM_errlinear']), np.amax(test_dict['EIM_errquadratic']), np.amax(test_dict['ROB_errlinear']), np.amax(test_dict['ROB_errquadratic']),  np.amax(np.abs(test_dict['mismatch'])),  np.amax(np.abs(test_dict['mismatch_psd']))))
			
	#checkpoint also at the end
	with open('test_dict.pickle', 'wb') as handle: pickle.dump(test_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

	#Plots
	import matplotlib.pyplot as plt
	plt.rcParams.update({'font.size': 24})
	plt.rcParams.update({'lines.linewidth': 2})

	# if the Plots directory doesn't exist, then create it.
	import os
	if not os.path.exists('Plots/'): os.makedirs('Plots/')

	for basis_type in ['linear', 'quadratic']:

		#make plot of contribution of each ROQ element
		plt.figure(figsize=(16,10))
		if basis_type == 'linear': typicalh = freq[Model['emp_nodes_'+basis_type]]**(-7/6)
		if basis_type == 'quadratic': typicalh = freq[Model['emp_nodes_'+basis_type]]**(-7/3)
		ROQ_contr = np.linalg.norm(Model['B_'+basis_type], axis=1)*typicalh
		plt.plot(np.arange(len(ROQ_contr))+1, ROQ_contr)
		plt.xlabel('Basis Number')
		plt.ylabel('Typical contribution')
		plt.yscale('log')
		plt.title(basis_type+' ROQ')
		plt.xlim(1, len(ROQ_contr))
		plt.grid(True)
		plt.tight_layout()
		plt.savefig('Plots/'+basis_type+'_ROQ_contributions.png')

		#make a histogram of errors
		plt.figure(figsize=(16,10))
		bins = np.geomspace(np.amin(test_dict['EIM_err'+basis_type]), max(np.amax(test_dict['EIM_err'+basis_type]),1.2*EIM_tol), nbins+1) 
		plt.xscale('log')
		plt.yscale('log')
		plt.hist(test_dict['EIM_err'+basis_type], bins=bins, histtype='step')
		plt.axvline(x=EIM_tol, label='tol=%.3g'%(EIM_tol))
		plt.ylabel('Number of Random Test Points')
		plt.xlabel('Surrogate Error')
		plt.xlim(bins[0], bins[-1])
		plt.title(basis_type+' ROQ $\\rightarrow$ With EIM err>%.3g: %s/%s'%(EIM_tol, np.sum(test_dict['EIM_err'+basis_type]>EIM_tol), len(test_dict['EIM_err'+basis_type])))
		plt.grid(True)
		plt.tight_layout()
		plt.legend(loc='upper right')
		plt.savefig('Plots/'+basis_type+'_ROQ_EIM_err.png')

		#compute norm22invA
		norm22invA = norm22(np.transpose(Model['B_'+basis_type]))

		#make a histogram of EIM_err/ROB_err
		plt.figure(figsize=(16,10))
		idx_plot = test_dict['ROB_err'+basis_type]>ROB_err_thr
		sEIM_sROB = test_dict['EIM_err'+basis_type][idx_plot]/test_dict['ROB_err'+basis_type][idx_plot]
		bins = np.geomspace(min(np.amin(sEIM_sROB),1), max(np.amax(sEIM_sROB),1.5*norm22invA), nbins+1) 
		plt.xscale('log')
		plt.yscale('log')
		plt.hist(sEIM_sROB, bins=bins, histtype='step')
		plt.axvline(x=norm22invA, label='$||A||_2^2 = %.1f$'%(norm22invA))
		plt.ylabel('Number of Test Points with $\\sigma_\\mathrm{ROB}>%.2g$'%(ROB_err_thr))
		plt.xlabel('$\\sigma_\\mathrm{EIM}/\\sigma_\\mathrm{ROB}$')
		plt.xlim(bins[0], bins[-1])
		plt.title(basis_type+' ROQ')
		plt.legend(loc='upper right')
		plt.grid(True)
		plt.tight_layout()
		plt.savefig('Plots/'+basis_type+'_sEIM_sROB.png')
	
	#compute the waveform and its EIM representation for different cases
	if not os.path.exists('Plots/waveforms/'): os.makedirs('Plots/waveforms/')
	plot_quantity_str = ['EIM_errlinear', 'EIM_errquadratic', 'mismatch', 'sEIM_sROB_linear', 'sEIM_sROB_quadratic']
	cases = ['max', 'min', 'median']
	for pquant_str in plot_quantity_str:
		print()
		#select the quantity to plot
		if pquant_str in ['EIM_errlinear', 'EIM_errquadratic', 'mismatch']: 
			pquant = np.abs(test_dict[pquant_str])
		elif pquant_str[:9] == 'sEIM_sROB':
			basis_type = pquant_str[10:]
			idx_plot = test_dict['ROB_err'+basis_type]>ROB_err_thr
			pquant = test_dict['EIM_err'+basis_type][idx_plot]/test_dict['ROB_err'+basis_type][idx_plot]
		for case in cases:
			#select the case of that quantity to plot
			if case == 'max': iplot = np.argmax(pquant)
			if case == 'min': iplot = np.argmin(pquant)
			if case == 'median': iplot = np.argmin(np.abs(pquant - np.median(pquant)))
			#parameters of the waveform to plot
			pquant_plot = pquant[iplot]
			#edit iplot in the case of the ratio, to take into account the cut
			if pquant_str[:9] == 'sEIM_sROB': iplot = np.arange(len(test_dict['EIM_errlinear']))[idx_plot][iplot] 
			param_plot = all_params[iplot]
			print(current_time(), case+'_'+pquant_str+' (%.3g)'%(pquant_plot) ,'parameters:', param_plot.tolist())
			#make the plot
			waveform = generate_a_normalized_waveform_from_parampoint('linear', param_plot, distance, delta_f, f_min, f_max, approximant, laldict=laldict)
			waveform_EIM = np.matmul(waveform[Model['emp_nodes_linear']], Model['B_linear'])
			waveform_quad = np.abs(waveform)**2/np.linalg.norm(np.abs(waveform)**2)
			waveform_quad_EIM = np.matmul(waveform_quad[Model['emp_nodes_quadratic']], Model['B_quadratic'])
			#make a plot of Re(h+), |h+|^2
			fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(21,19))
			plt.suptitle(r'$\mathcal{M}_c=%.3g, q=%.3g, \chi_\mathrm{eff}=%.3f, \chi_\mathrm{p}=%.3f, \iota=%.1f^{\circ}$'%(param_plot[0], param_plot[1], chi_eff(param_plot), chi_p(param_plot), param_plot[8]*(180/np.pi)) +'\n'+  r'$\sigma^\mathrm{EIM}_\mathrm{linear}=%.3g, \sigma^\mathrm{ROB}_\mathrm{linear}=%.3g, \sigma^\mathrm{EIM}_\mathrm{quad}=%.3g, \sigma^\mathrm{ROB}_\mathrm{quad}=%.3g, \mathrm{MM}=%.3g$'%(test_dict['EIM_errlinear'][iplot], test_dict['ROB_errlinear'][iplot], test_dict['EIM_errquadratic'][iplot], test_dict['ROB_errquadratic'][iplot], np.abs(test_dict['mismatch'][iplot])))
			#plot of Re(h+)
			axs[0,0].plot(freq, np.real(waveform))
			axs[0,0].plot(freq, np.real(waveform_EIM), '--')
			axs[0,0].scatter(freq[Model['emp_nodes_linear']], np.real(waveform[Model['emp_nodes_linear']]))
			axs[0,0].set_ylabel(r'Re($h_+$)')
			axs[0,0].grid(True, which='both')
			#plot of |h_+|^2
			axs[1,0].plot(freq, waveform_quad, label='Original Waveform')
			axs[1,0].plot(freq, waveform_quad_EIM, '--', label='Empirical Interpolant')
			axs[1,0].scatter(freq[Model['emp_nodes_quadratic']], waveform_quad[Model['emp_nodes_quadratic']], label='Interpolation Nodes')
			axs[1,0].set_ylabel(r'$|h_+|^2$')
			axs[1,0].set_yscale('log')
			axs[1,0].grid(True, which='both')
			#common stuff
			axs[1,0].set_xscale('log')
			axs[1,0].set_xlabel('$f$ [Hz]')
			axs[1,0].set_xlim(freq[0], freq[-1])
			axs[1,0].legend()
			#plot of |h_+ - h_+^{EIM}|^2
			w_err_linear = np.abs(waveform - waveform_EIM)**2
			axs[0,1].plot(freq, w_err_linear)
			axs[0,1].set_ylabel(r'$|h_+ - h_+^\mathrm{EIM}|^2$')
			axs[0,1].set_yscale('log')
			axs[0,1].set_ylim(max(np.amin(w_err_linear), 1e-5*np.amax(w_err_linear)), np.amax(w_err_linear))
			axs[0,1].grid(True, which='both')
			#plot of ||h_+|^2 - |h_+^{EIM}|^2|^2
			w_err_quad = np.abs(waveform_quad - waveform_quad_EIM)**2
			axs[1,1].plot(freq, w_err_quad)
			axs[1,1].set_ylabel(r'$||h_+|^2 - |h_+^\mathrm{EIM}|^2|^2$')
			axs[1,1].set_yscale('log')
			axs[1,1].set_ylim(max(np.amin(w_err_quad), 1e-5*np.amax(w_err_quad)), np.amax(w_err_quad))
			axs[1,1].grid(True, which='both')
			#common stuff
			axs[1,1].set_xscale('log')
			axs[1,1].set_xlabel('$f$ [Hz]')
			axs[1,1].set_xlim(freq[0], freq[-1])
			#save figure						
			plt.tight_layout()
			plt.savefig('Plots/waveforms/waveform_'+case+'_'+pquant_str+'.png')
	
	#make a histogram of the Mismatch
	plt.figure(figsize=(16,10))
	x_min = max(min(np.amin(np.abs(test_dict['mismatch'])), np.amin(np.abs(test_dict['mismatch_psd']))), ROB_err_thr)
	x_max = max(np.amax(np.abs(test_dict['mismatch'])), np.amax(np.abs(test_dict['mismatch_psd'])))
	bins = np.geomspace(x_min, x_max, nbins+1) 
	plt.xscale('log')
	plt.yscale('log')
	plt.hist(np.abs(test_dict['mismatch']), bins=bins, histtype='step', label='No PSD')
	plt.hist(np.abs(test_dict['mismatch_psd']), bins=bins, histtype='step', label=psd_name)
	plt.ylabel('Number of Test Points')
	plt.xlabel(r'$\left| 1 - \frac{\langle h, h^\mathrm{ROQ}\rangle}{\sqrt{\langle h, h \rangle \langle h^\mathrm{ROQ}, h^\mathrm{ROQ}\rangle}} \right|$')
	plt.xlim(bins[0], bins[-1])
	plt.legend(loc = 'upper left')
	plt.grid(True)
	plt.tight_layout()
	plt.savefig('Plots/mismatch.png')

	#plot the psd used
	plt.figure(figsize = (16,10))
	plt.loglog(freq, psd)
	plt.xlim(np.amin(freq), np.amax(freq))
	plt.ylim(np.amin(psd[psd>0]), np.amax(psd))
	plt.xlabel('PSD [1/$\\sqrt{\\mathrm{Hz}}$]')
	plt.ylabel('Frequency [Hz]')
	plt.title(psd_name)
	plt.savefig('Plots/PSD.png')

	return True
		
