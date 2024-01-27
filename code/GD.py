from amm import amm
import numpy as np
from params import params
import inspect

def gradient_descenting_search_dist(simulated_sample, initial_parameters, given_parameter,learning_rate=0.01,delta=(0.01,0.02),epsilon=1e-9):
	a = given_parameter['alpha']
	x0 = given_parameter['x_0']
	Rx0 = given_parameter['Rx0']
	phi = given_parameter['phi']
	q = given_parameter['q']
	zeta = given_parameter['zeta'] # To make sure the constrain would not be violated
	N_pools = given_parameter['N_pools']
	# lambd = 1/1.126 # represent the cost when swap y to x
	lambd = 1
	old_initial_parameters = initial_parameters
	# parameters = np.array([initial_parameters[j] - initial_parameters[-1] for j in range(N_pools-1)])
	# old_parameters = parameters
	sample_number = simulated_sample.shape[0]
	cvar, old_cvar = 0,100
	q_quantile = 0
	shorten_factor = 1
	close_to_constrain = False
	violate_constrain = False
	free_index = [i for i in range(N_pools-1)] # the indices that can be any value between 0 and 1
	constrained_index = [N_pools-1] # the index that is 1 - sum of free index
	fixed_index = [] # the indices that equals zero
	no_descenting_counter = 0 
	run_time_counter = 0
	while abs(cvar - old_cvar) > epsilon:
		run_time_counter += 1
		if run_time_counter >= 1e5:
			# print('current line: ',inspect.currentframe().f_back.f_lineno)
			break
		if len(free_index) == 0:
			# There is no freedom, we have found optimal point
			# print('current line: ',inspect.currentframe().f_back.f_lineno)
			break
		initial_LP_coin = Rx0 + 0.5 * x0 * initial_parameters
		y_coins = simulated_sample[:,:,1] @ (x0 * initial_parameters / initial_LP_coin)
		best_price = np.max(simulated_sample[:,:,0]/(simulated_sample[:,:,1] + np.outer(y_coins, (1 - phi)) - simulated_sample[:,:,1] * (x0 * initial_parameters / initial_LP_coin * (1-phi))) * (1-phi),axis= 1)
		# best_price = np.max(simulated_sample[:,:,0]/simulated_sample[:,:,1] ,axis= 1)
		approximate_sample = np.log(0.5 * simulated_sample[:,:,0] @ (initial_parameters / initial_LP_coin) + 0.5 * lambd * ((simulated_sample[:,:,1] @ (initial_parameters / initial_LP_coin)).T * best_price).T)
		# print(approximate_sample[::25])
		combined_sample = [(approximate_sample[i], simulated_sample[i],best_price[i]) for i in range(sample_number)]
		combined_sample.sort(key=lambda x:x[0])
		alpha_index = int((1-a)*sample_number)
		q_index = int((1-q)*sample_number)
		old_cvar = cvar
		cvar = -np.mean([combined_sample[i][0] for i in range(alpha_index)])
		if close_to_constrain:
			k = 1
		else:
			k = 3
		q_quantile = np.mean([combined_sample[i][0] for i in range(q_index-k+1,q_index+k+1)])
		# if run_time_counter % 2000 == 0:
		# 	print('Searching near the boundary: ', close_to_constrain)
		# 	print('q quantile: ', q_quantile)
		if close_to_constrain:
			if q_quantile <= zeta + delta[0]:
				# If the previous step is close to the boundary and the new step is even more closer,  
				# then we stop our search and return the previous parameters
				violate_constrain = True
				# print('current line: ',inspect.currentframe().f_back.f_lineno)
				break
			elif q_quantile <= zeta + delta[1]:
				# Then new step is still close to boundary, we need to search in the direction that is  
				# vertical to the gradient of boundary condition, to make the new point no closer to the boundary
				grad_cvar = -np.mean(((0.5 * np.array([combined_sample[i][1][:,0] for i in range(alpha_index)]) * (Rx0 / initial_LP_coin**2) + 0.5 * lambd * (np.array([combined_sample[i][1][:,1] * combined_sample[i][2] for i in range(alpha_index)]) * (Rx0 / initial_LP_coin**2))).T / np.exp([combined_sample[i][0] for i in range(alpha_index)])).T,axis=0)
				grad_cvar = (grad_cvar - np.sum(grad_cvar[constrained_index]))[free_index]
				grad_cons = np.mean(((0.5 * np.array([combined_sample[i][1][:,0] for i in range(q_index-k+1,q_index+k+1)]) * (Rx0 / initial_LP_coin**2) + 0.5 * lambd * (np.array([combined_sample[i][1][:,1] * combined_sample[i][2] for i in range(q_index-k+1,q_index+k+1)]) * (Rx0 / initial_LP_coin**2))).T / np.exp([combined_sample[i][0] for i in range(q_index-k+1,q_index+k+1)])).T,axis=0)
				# grad_cons = 0.5 * np.array([combined_sample[i][1][:,0] for i in range(q_index-k+1,q_index+k+1)]) @ (Rx0 / initial_LP_coin**2) + 0.5 * lambd * ((np.array([combined_sample[i][1][:,1] for i in range(q_index-k+1,q_index+k+1)]) @ (Rx0 / initial_LP_coin**2)).T * best_price).T
				grad_cons = (grad_cons - np.sum(grad_cons[constrained_index]))[free_index]
				old_initial_parameters = initial_parameters
				# If the gradient of CVaR is opposite with the gradient of constrain, nothing 
				# is need to be done, otherwise we need to substract the gradient of constrain
				# from the gradient of CVaR
				if np.sum(grad_cvar * grad_cons) >0:
					grad_cvar -= np.sum(grad_cvar * grad_cons)/np.sum(grad_cons**2) * grad_cons
				if all(np.abs(grad_cvar) < epsilon):
					# In this case, two gradient is in the same direction, we should stop
					# print('current line: ',inspect.currentframe().f_back.f_lineno)
					break
				# check if all parameters are positive and have a sum smaller than 1 after update
				# If not, take another stepsize so that the next point will be on the boundary
				# Assume once we arrive at the boundary we will always on the boundary
				temp_param = initial_parameters[free_index] - learning_rate * grad_cvar * shorten_factor
				if any(temp_param <= 0) or np.sum(temp_param) >= 1:
					stepsize = np.min(np.array([initial_parameters[free_index[i]]/(grad_cvar[i] * shorten_factor) for i in range(len(free_index)) if temp_param[i] < 0]))
					if np.sum(grad_cvar) > 0:
						stepsize = min(stepsize, (1-np.sum(initial_parameters[free_index]))/np.sum(grad_cvar))
					if stepsize == 0:
						# This means it can descent any more in difinition domain
						# But it might be because the initial paramter is on the boundary and the
						# free_index haven't been updated
						no_descenting_counter += 1
						old_cvar = 100
						if no_descenting_counter >=2:
							# print('current line: ',inspect.currentframe().f_back.f_lineno)
							break
					temp_param = initial_parameters[free_index] - stepsize * grad_cvar * shorten_factor
					initial_parameters[free_index] = temp_param
					initial_parameters[constrained_index] = 1 - np.sum(initial_parameters[free_index])
					initial_parameters[fixed_index] = 0
					to_fix_index = []
					for i in range(len(temp_param)):
						if temp_param[i] == 0:
							to_fix_index.append(free_index[i])
					for i in to_fix_index:
						free_index.remove(i)
						fixed_index.append(i)
					if np.sum(temp_param) == 1:
						fixed_index.append(constrained_index[0])
						constrained_index[0] = free_index[-1]
						free_index = free_index[:-1]
				else:
					initial_parameters[free_index] = temp_param
					initial_parameters[constrained_index] = 1 - np.sum(initial_parameters[free_index])
					initial_parameters[fixed_index] = 0
			else:
				# If the new step is no longer close to the boundary, go back to normal searching method
				close_to_constrain = False

		if not close_to_constrain:
			if q_quantile <= zeta + delta[0]:
				# If the previous step is no close to the boundary but the new step is really close to 
				# boundary, we need to go back to previous step and shorten our searching step
				shorten_factor *= 0.5
				if shorten_factor <= 0.01:
					# If we ran into this situation several times, it's better to regenerate samples.
					# print('current line: ',inspect.currentframe().f_back.f_lineno)
					break
				old_cvar = 100
				initial_parameters = old_initial_parameters
				# parameters = old_parameters
				continue
			elif q_quantile <= zeta + delta[1]:
				# In this case, we get closer to the boundary and need to make sure our (1-q) 
				# quantile does not descent further. Thus we jump to another searching method.
				old_cvar = 100
				close_to_constrain = True
				continue
			else:
				# In this case, we are away from the constrain, we can omit the constrain.
				grad_cvar = -np.mean(((0.5 * np.array([combined_sample[i][1][:,0] for i in range(alpha_index)]) * (Rx0 / initial_LP_coin**2) + 0.5 * lambd * (np.array([combined_sample[i][1][:,1] * combined_sample[i][2] for i in range(alpha_index)]) * (Rx0 / initial_LP_coin**2))).T / np.exp([combined_sample[i][0] for i in range(alpha_index)])).T,axis=0)
				grad_cvar = (grad_cvar - np.sum(grad_cvar[constrained_index]))[free_index]
				if all(np.abs(grad_cvar) < epsilon):
					# In this case, local minimum is achieved, we should stop
					# print('current line: ',inspect.currentframe().f_back.f_lineno)
					break
				old_initial_parameters = initial_parameters
				# old_parameters = parameters
				# check if all parameters are positive and have a sum smaller than 1 after update
				# If not, take another stepsize so that the next point will be on the boundary
				# Assume once we arrive at the boundary we will always on the boundary
				temp_param = initial_parameters[free_index] - learning_rate * grad_cvar * shorten_factor
				if any(temp_param <= 0) or np.sum(temp_param) >= 1:
					stepsize = learning_rate
					for i in range(len(free_index)):
						if temp_param[i] <= 0:
							if grad_cvar[i] == 0:
								stepsize = 0
							else:
								stepsize = min(stepsize,initial_parameters[free_index[i]]/(grad_cvar[i] * shorten_factor))
					if np.sum(grad_cvar) > 0:
						stepsize = min(stepsize, (1-np.sum(initial_parameters[free_index]))/np.sum(grad_cvar))
					if stepsize == 0:
						# This means it can descent any more in difinition domain
						# But it might be because the initial paramter is on the boundary and the
						# free_index haven't been updated
						no_descenting_counter += 1
						if no_descenting_counter >=2:
							# print('current line: ',inspect.currentframe().f_back.f_lineno)
							break
					temp_param = initial_parameters[free_index] - stepsize * grad_cvar * shorten_factor
					initial_parameters[free_index] = temp_param
					initial_parameters[constrained_index] = 1 - np.sum(initial_parameters[free_index])
					initial_parameters[fixed_index] = 0
					to_fix_index = []
					for i in range(len(temp_param)):
						if temp_param[i] == 0:
							to_fix_index.append(free_index[i])
					for i in to_fix_index:
						free_index.remove(i)
						fixed_index.append(i)
					if np.sum(temp_param) == 1:
						fixed_index.append(constrained_index[0])
						constrained_index[0] = free_index[-1]
						free_index = free_index[:-1]
				else:
					initial_parameters[free_index] = temp_param
					initial_parameters[constrained_index] = 1 - np.sum(initial_parameters[free_index])
					initial_parameters[fixed_index] = 0
		# if run_time_counter <= 2:
		# 	print('first search gradient: ',grad_cvar," with respect to index: ",free_index)
		# 	print("target value: ",(old_cvar,cvar))
	return (cvar,q_quantile),initial_parameters,violate_constrain,run_time_counter
"""	while abs(cvar - old_cvar > epsilon):
		approximate_sample = simulated_sample @ initial_parameters
		combined_sample = [(approximate_sample[i], simulated_sample[i]) for i in range(sample_number)]
		combined_sample.sort(key=lambda x:x[0])
		alpha_index = int((1-a)*sample_number)
		q_index = int((1-q)*sample_number)
		cvar = np.mean([approximate_sample[i] for i in range(alpha_index)])
		old_initial_parameters = initial_parameters
		old_parameters = parameters
		old_cvar = cvar
		if close_to_constrain:
			k = 1
		else:
			k = 5
		q_quantile = np.mean([approximate_sample[i] for i in range(q_index-k+1,q_index+k+1)])
		if on_boundary:
			if (q_quantile < zeta):
				# the constrain has been violated, we need to go back to previous step and shorten our step
				boundary_fail = True
				# print('current line: ',inspect.currentframe().f_back.f_lineno)
				break
			elif q_quantile > zeta:
				# machine accuracy might be needed to be taken into consideration
				shorten_factor = 1
				on_boundary = False
			else:
				constrained_index = np.nonzero(boundary_condition)[0]
				if constrained_index is None:
					on_boundary = False
				else:
					free_index = [i for i in range(N_pools-1) if i != constrained_index]
					cvar_grad = np.mean([[simulated_sample[i,j] - simulated_sample[i,-1] -(simulated_sample[i,constrained_index] - simulated_sample[i,-1]) * boundary_condition[j] / boundary_condition[[constrained_index]] for j in free_index] for i in range(alpha_index)] ,axis = 0)
					parameters[free_index] -= learning_rate * cvar_grad
					parameters[constrained_index] = (boundary_condition[-1] - np.sum(boundary_condition[free_index] * parameters[free_index])) / boundary_condition[constrained_index]
					initial_parameters[:-1] = parameters
					initial_parameters[-1] = 1 - np.sum(parameters)
		if not on_boundary:
			if (q_quantile < zeta):
				# the constrain has been violated, we need to go back to previous step and shorten our step
				close_to_constrain = True
				shorten_factor *= 0.5
				cvar += 10
				initial_parameters = old_initial_parameters
				parameters = old_parameters
				continue
			elif q_quantile == zeta:
				# machine accuracy might be needed to be taken into consideration
				shorten_factor = 1
				on_boundary = True
			else:
				shorten_factor = 1
				on_boundary = False
			cvar_grad = np.mean([[simulated_sample[i,j] - simulated_sample[i,N_pools-1] for j in range(N_pools-1)] for i in range(alpha_index)] ,axis = 0)
			q_quantile_grad = np.mean([[simulated_sample[i,j] - simulated_sample[i,N_pools-1] for j in range(N_pools-1)] for i in range(q_index-k+1,q_index+k+1)] ,axis = 0)
			if on_boundary:
				boundary_condition[:-1] = q_quantile_grad
				boundary_condition[-1] =  np.sum(q_quantile_grad*parameters)
				continue
			if q_quantile - learning_rate * shorten_factor * np.sum(q_quantile_grad*cvar_grad) > zeta:
				stepsize = learning_rate
			else:
				close_to_constrain = True
				stepsize = (q_quantile - zeta) / np.sum(q_quantile_grad*cvar_grad)
			parameters -= stepsize * cvar_grad * shorten_factor
			initial_parameters[:-1] = parameters
			initial_parameters[-1] = 1 - np.sum(parameters)
	return (cvar,q_quantile), initial_parameters*x0		"""

def generate_simulated_sample(initial_parameters, given_parameter, batch_size):
	""" Fix the seed """
	np.random.seed(given_parameter['seed'])

	# 参考2.8，先取出初态，t0先x换LP，之后模拟到T，T出换LP，后计算最优化条件
	""" Initialise the pools according to percentage"""
	Rx0 = given_parameter['Rx0']
	Ry0 = given_parameter['Ry0']
	phi = given_parameter['phi']

	pools = amm(Rx=Rx0 , Ry=Ry0 , phi=phi)
	# N_pools = given_parameter['N_pools']
	""" X-coins for each pool"""
	x_0 = given_parameter['x_0']
	xs_0 = x_0 * initial_parameters
	l = pools.swap_and_mint(xs_0)
	""" Simulate paths of trading in the pools """
	T = given_parameter['T']
	kappa = given_parameter['kappa']
	p = given_parameter['p']
	sigma = given_parameter['sigma']
	end_pools , Rx_t , Ry_t , v_t , event_type_t , event_direction_t = pools.simulate(kappa = kappa , p = p, sigma = sigma , T = T, batch_size = batch_size)
	initial_LP_coin = Rx0 + 0.5 * x_0 * initial_parameters
	simulated_sample = np.array([np.array([end_pools[i].Rx,end_pools[i].Ry]).T for i in range(len(end_pools))])
	y_coins = simulated_sample[:,:,1] @ (x_0 * initial_parameters / initial_LP_coin)
	best_price = np.max(simulated_sample[:,:,0]/(simulated_sample[:,:,1] + np.outer(y_coins, (1 - phi)) - simulated_sample[:,:,1] * (x_0 * initial_parameters / initial_LP_coin * (1-phi))) * (1-phi),axis= 1)
	# best_price = np.max(simulated_sample[:,:,0]/simulated_sample[:,:,1] ,axis= 1)
	# lambd = 1/1.1262
	approximate_sample = np.log(0.5 * simulated_sample[:,:,0] @ (initial_parameters / initial_LP_coin) + 0.5 * ((simulated_sample[:,:,1] @ (initial_parameters / initial_LP_coin)).T * best_price).T)
	real_sample = np.log(np.array([end_pools[i].burn_and_swap(l) for i in range(len(end_pools))]) / x_0)
	print(np.mean(np.abs(approximate_sample - real_sample)))
	qtl = np.quantile(-real_sample, given_parameter['alpha'])
	cvar = np.mean(-real_sample[-real_sample >= qtl])
	P_zeta = len(real_sample[real_sample >= given_parameter['zeta']])/len(real_sample)
	return simulated_sample,(cvar,P_zeta)

if __name__ == "__main__":
	# initial_parameters = np.ones(params['N_pools']) / params['N_pools']
	# initial_parameters = np.array([0.5,0.1,0.04,0.04,0.3,0.02])
	initial_parameters = np.random.rand(params['N_pools'])
	initial_parameters /= np.sum(initial_parameters)

	old_real_cvar,real_cvar = 100,0
	counter = 0
	old_initial_parameters = 0
	while abs(old_real_cvar - real_cvar) > 1e-7:
		counter += 1
		# if counter <= 15:
		# 	batch_size = 300
		# else:
		# 	batch_size = 1000
		batch_size = params['batch_size']
		real_cvar, real_P_zeta = 0,0
		simulated_sample,(real_cvar, real_P_zeta) = generate_simulated_sample(initial_parameters,params,batch_size)
		print("the CVaR of current weight is: ",real_cvar)
		print("the probability of larger than zeta is: ",real_P_zeta)
		if real_P_zeta < params['q']:
			initial_parameters = old_initial_parameters
			break
		cvar,q_quantile,violate_constrain,run_time_counter =0,0,0,0
		old_initial_parameters = initial_parameters
		(cvar,q_quantile),initial_parameters,violate_constrain,run_time_counter = gradient_descenting_search_dist(simulated_sample, initial_parameters, params,learning_rate=0.001,delta=(0.001,0.005))
		print('approximate CVaR and quantile are: ',cvar,", ",q_quantile)
		print('investment weight: ', initial_parameters)
		print("searching rounds: ", run_time_counter)
		if counter >= 15:
			break
	# np.random.seed(params['seed'])
	simulated_sample,(real_cvar, real_P_zeta) = generate_simulated_sample(initial_parameters,params,params['batch_size'])
	print("the CVaR of current weight is: ",real_cvar)
	print("the probability of larger than zeta is: ",real_P_zeta)