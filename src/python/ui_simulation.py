import sys
import numpy as np
from scipy.optimize import bisect
from scipy.optimize import minimize

class Model:

	def __init__(self, model_params, grid_params):

		# model parameters
		self.beta = model_params["beta"]
		self.gamma = model_params["gamma"]
		self.xi = model_params["xi"]
		self.theta = model_params["theta"]
		self.T = model_params["T"]
		self.T_B = model_params["T_B"]
		self.r = model_params["r"]
		self.wage = model_params["wage"]
		self.y_bar = model_params["y_bar"]
		self.initial_asset = model_params["initial_asset"]

		# policy instruments
		self.b_u = 0.5
		self.b_e = 0.0
		self.tau = 15.8 * self.b_u / (self.T - 24.3)

		# grid
		self.asset_grid = np.linspace(grid_params["asset_min"], \
											grid_params["asset_max"], \
											grid_params["asset_size"])

	def reset(self, model_params):
		# model parameters
		self.beta = model_params["beta"]
		self.gamma = model_params["gamma"]
		self.xi = model_params["xi"]
		self.theta = model_params["theta"]
		self.T = model_params["T"]
		self.T_B = model_params["T_B"]
		self.r = model_params["r"]
		self.wage = model_params["wage"]
		self.y_bar = model_params["y_bar"]
		self.initial_asset = model_params["initial_asset"]


	def set_calibrated_params(self, params):

		if isinstance(params, list) or isinstance(params, np.ndarray):
			self.xi = params[0] 
			self.theta = params[1]

		elif isinstance(params, dict):
			self.xi = params["xi"]
			self.theta = params["theta"]
			
		else:
			sys.exit("params must be either a list/array or a dict.")

	# utility from consumption
	def util_c(self, c):
		u = c ** (1 - self.gamma) / (1 - self.gamma)
		return u

	# marginal util from consumption
	def mutil_c(self, c):
		mutil = c ** (-self.gamma)
		return mutil

	# inverse marginal util from consumption
	def mutil_c_inv(self, mu):
		c = mu ** (-1 / self.gamma)
		return c

	# disutility of search
	def util_s(self, s):
		u = self.theta * s ** (1 + self.xi) / (1 + self.xi)
		return u

	# marginal disutil of search
	def mutil_s(self, s):
		mutil = self.theta  * s ** self.xi
		return mutil

	# inverse marginal disutil of search
	def mutil_s_inv(self, mu):

		search = (mu / self.theta) ** (1 / self.xi)
		search = np.maximum(0, search)
		search = np.minimum(1, search)
		return search

	# income
	def income(self, t, emp):

		if emp == 1:
			if t < self.T_B:
				return self.wage + self.b_e - self.tau
			else:
				return self.wage - self.tau
		elif emp == 0:
			if t < self.T_B:
				return self.b_u + self.y_bar
			else:
				return self.y_bar
		else:
			sys.exit("emp should be 0 or 1")



class Solution:

	def __init__(self, model):

		self.model = model

		# consumption policy function
		T, a_size = model.T, len(model.asset_grid)
		self.cons_u = np.empty([T, a_size])

		# expected marginal utilities
		self.Emutil_u = np.empty([T, a_size])

		# value function
		self.value = np.empty([T, a_size])
		self.value_u = np.empty([T, a_size])



	def solve_employed(self, t):
		"""
		Given time t, compute optimal consumption c_e and value v_e.
		Since individuals do not lose their job once they get it, 
		the optimal consumption is equal to the lifetime income divided by remainig periods.
		Input:
			t: int
		Output:
			c_e: array
			v_e: array
		"""
		time_remain = self.model.T - t  # index start from zero
		c_e = self.model.asset_grid / time_remain + self.model.wage - self.model.tau

		if t < self.model.T_B:
			c_e = c_e +  self.model.b_e * (self.model.T_B - t) / time_remain

		v_e = time_remain * self.model.util_c(c_e)

		return c_e, v_e




	def solve_unemployed(self, t):
		"""
		Given time t, compute optimal consumption c_e and value v_e.
		Solve Euler equations

		Input:
			t: int
		Output:
			c_u: array
			v_u: array
		"""

		beta, r = self.model.beta, self.model.r

		a_grid = self.model.asset_grid
		a_size = a_grid.size
		a_now = a_grid.reshape(a_size, 1)
		a_next = a_grid.reshape(1, a_size)
		income = self.model.income(t = t, emp = 0)


		if t == self.model.T - 1:
			c_u = a_grid + income
			v_u = self.model.util_c(c_u)
		else:
			# create a matrix containing errors in Euler equations
			# row: today's asset, column: tomorrow's asset
			c_now = a_now + income - a_next / (1 + r)
			emu_next = self.Emutil_u[t + 1, :]
			temp = beta * (1 + r) * emu_next
			euler_error = self.model.mutil_c_inv(temp).reshape(1, a_size) - c_now

			# solve euler equation
			# take euler_error as x and a_grid as f(x)
			a_opt = [np.interp(0, euler_error[i, :], a_grid) for i in range(a_size)]
			c_u = a_grid + income - np.array(a_opt) / (1 + r)

			v_next = [np.interp(a_opt[i], a_grid, self.value[t + 1, :]) for i in range(a_size)]
			v_u = self.model.util_c(c_u) + beta * np.array(v_next)

		return c_u, v_u


	def solve(self):
		"""
		Solve the model backward from t = T-1 to t = 0.

		"""

		for t in reversed(range(self.model.T)):

			c_e, v_e = self.solve_employed(t)
			c_u, v_u = self.solve_unemployed(t)

			search = self.model.mutil_s_inv(v_e - v_u)

			# store solutions
			self.cons_u[t, :] = c_u
			self.value_u[t, :] = v_u
			self.value[t, :] = search * v_e + (1.0 - search) * v_u - self.model.util_s(search)
			self.Emutil_u[t, :] = search * self.model.mutil_c(c_e) \
							+ (1 - search) * self.model.mutil_c(c_u) 


	def compute_consumption_employed_exact(self, a, t):
		"""
		Given asset level and time, this function solves the consumption level of employed people.
		Input:
			a: asset
			t: time
		Output:
			c_e: consumption
		"""

		time_remain = self.model.T - t  # index start from zero
		c_e = a / time_remain + self.model.wage - self.model.tau

		if t < self.model.T_B:
			c_e = c_e +  self.model.b_e * (self.model.T_B - t) / time_remain

		return c_e


	def compute_duration(self, compensated = False):
		"""
		Simulate the model forward to compute average unemployment duration.
		Since employment is an absorbing state, I only need to follow the choices of unemployed people.
		Input:
			compensated: True or False
		Output:
			duration: compensated = True -> compute UI-compensated duration, compensated = False -> compute unemp duration
		"""

		duration = 0
		survival = 1

		a = self.model.initial_asset

		if compensated == True:
			T = self.model.T_B
		else:
			T = self.model.T

		for t in range(T):

			# search effort
			search = self.solve_search(a, t)
			survival = survival * (1 - search)
			duration += survival

			# consumption
			c = np.interp(a, self.model.asset_grid, self.cons_u[t, :])

			# update asset
			a = (1 + self.model.r) * (a - c + self.model.income(t = t, emp = 0))


		return duration

	def compute_duration_elasticity(self, compensated = False):
		"""
		Compute the elasticity of unemployment duration with respect to UI benefits
		Input:
			compensated: True or False
		Output:
			duration: compensated = True -> compute UI-compensated duration, compensated = False -> compute unemp duration
		"""

		b_u = self.model.b_u
		b_u_1 = 0.98 * b_u
		b_u_2 = 1.02 * b_u

		self.model.b_u = b_u_1
		self.solve()
		D_1 = self.compute_duration(compensated)

		self.model.b_u = b_u_2
		self.solve()
		D_2 = self.compute_duration(compensated)

		self.model.b_u = b_u

		if D_1 == 0 or D_2 == 0:
			return 1e+6

		els = (np.log(D_2) - np.log(D_1)) / (np.log(b_u_2) - np.log(b_u_1))
		return els

	def compute_duration_derivative(self, emp, compensated = False):
		"""
		Compute the derivative of unemployment duration with respect to UI benefits or work incentives
		Input:
			emp: 0 or 1
			compensated: True or False
		Output:
			out: compensated = True -> compute derivative of UI-compensated duration, 
				 compensated = False -> compute unemp duration
				 emp = 1 -> derivative with respect to work incentives
				 emp = 0 -> derivative with respect to UI benefits
		"""

		if emp == 0:
			b = self.model.b_u
		else:
			b = self.model.b_e

		b_1 = b + 0.05
		b_2 = b - 0.05					

		if emp == 0:
			self.model.b_u = b_1
		else:
			self.model.b_e = b_1

		self.solve()
		D_1 = self.compute_duration(compensated)

		if emp == 0:
			self.model.b_u = b_2
		else:
			self.model.b_e = b_2

		self.solve()
		D_2 = self.compute_duration(compensated)

		if emp == 0:
			self.model.b_u = b
		else:
			self.model.b_e = b

		out = (D_2 - D_1) / (b_2 - b_1)
		return out


	def compute_consumption_drop(self, t_end):
		"""
		Compute consumption drop upon unemployment

		Input:
			t_end: the end of periods for which consumption drop is computed
		Output:
			abs(c_drop): absolute value of consumption drop upon unemployment
		"""

		a = self.model.initial_asset
		c_0 = self.model.wage - self.model.tau + (a / self.model.T)  # hypothetical consumption if no job loss

		for t in range(t_end):

			# consumption
			c = np.interp(a, self.model.asset_grid, self.cons_u[t, :])

			# update asset
			a = (1 + self.model.r) * (a - c + self.model.income(t = t, emp = 0))

		c_drop = (c - c_0) / c_0
		return abs(c_drop)


	def solve_search(self, a, t):
		"""
		Given asset and time, this function solves search effort

		Input:
			a: asset
			t: time
		Output:
			search: search effort
		"""

		# compute v_e
		time_remain = self.model.T - t  # index start from zero
		c_e = a / time_remain + self.model.wage - self.model.tau

		if t < self.model.T_B:
			c_e = c_e +  self.model.b_e * (self.model.T_B - t) / time_remain

		v_e = time_remain * self.model.util_c(c_e)

		# compute v_u
		v_u = np.interp(a, self.model.asset_grid, self.value_u[t, :])

		search = self.model.mutil_s_inv(v_e - v_u)

		return search




def estimation_objective(m, moments, params):
	"""
	The objective function in the estimation.
	Input:
		m: Instance of the Model class
		moments: data moments used in the estimation
		params: parameter value at which the objective function is evaluated
	Output:
		out: sum of squared differences between data moments and simulated moments
	"""

	m.xi = params[0] 
	m.theta = params[1] * 100

	sol = Solution(m)
	sol.solve()

	D = sol.compute_duration(compensated = False)
	els = sol.compute_duration_elasticity()

	obj = np.empty(2)
	obj[0] = (D - moments["duration"]) ** 2
	obj[1] = (els - moments["elasticity"]) ** 2

	obj_scale = np.array([0.01, 1])
	out = np.dot(obj, obj_scale)
	return out

def estimation_objective_unscaled(m, moments, params):

	params[1] = params[1] * 0.01
	return estimation_objective(m, moments, params)


def estimate_parameters(m, moments):
	"""
	Choose parameter values that minimize the distance between data moments and simulated moments.
	Input:
		m: Instance of the Model class
		moments: data moments used in the estimation
	Output:
		params: parameters that minimize the distance between data momoents and simulated moments
	"""

	est_obj = lambda x: estimation_objective(m, moments, x)
	x0 = [1.0, 0.5]
	bounds = [(0.01, 2.0), (0.01, 10.0)]

	result = minimize(fun = est_obj, x0 = x0, bounds = bounds, method = "L-BFGS-B")

	print(result)

	params = result.x
	params[1] = params[1] * 100

	return params



class Counterfactual:
	
	def __init__(self, model):

		self.model = model



	def compute_net_expenditure(self, tau, b_u, b_e):
		"""
		Given taxes, UI benefits, and work incentives, this function computes
		the net expenditure given by (T-D)tau - D_Bb_u - (T_B-D_B)b_u
		Input:
			tau: tax
			b_u: UI benefits
			b_e: work incentives
		Output:
			out: net expenditure
		"""

		self.model.tau = tau
		self.model.b_u = b_u
		self.model.b_e = b_e

		sol = Solution(self.model)
		sol.solve()

		D = sol.compute_duration(compensated = False)
		D_B = sol.compute_duration(compensated = True)

		out = (self.model.T - D) * tau - D_B * b_u - (self.model.T_B - D_B) * b_e
		return out

	def compute_tax(self, b_u, b_e):
		"""
		Given UI benefits and work incentives, this function computes 
		the tax rate satisfying the budget constraint
		Input:
			b_u: UI benefits
			b_e: work incentives
		Output:
			tau: tax rate that satisfies the budget constraint given (b_u, b_e)
		"""

		fun = lambda x: self.compute_net_expenditure(x, b_u, b_e)

		tau_max = 0.05
		while fun(tau_max) < 0:
			tau_max = tau_max * 1.1

		tau = bisect(fun, a = 0, b = tau_max)
		return tau

	def compute_MU_unemployed(self, sol, t_end = 26):
		"""
		Computes the weighted average of marginal utilities of unemployed people
		Input:
			sol: Instance of the Solution class
			t_end: periods over which the average is taken
		Output:
			MU_u: weighted average of marginal utilities
		"""

		MU_u = 0
		a = self.model.initial_asset
		duration = 0
		survival = 1

		for t in range(t_end):

			# search effort
			search = sol.solve_search(a, t)
			survival = survival * (1 - search)
			duration += survival

			# consumption
			c = np.interp(a, self.model.asset_grid, sol.cons_u[t, :])

			# Marginal util
			MU_u += survival * self.model.mutil_c(c)

			# update asset
			a = (1 + self.model.r) * (a - c + self.model.income(t = t, emp = 0))


		MU_u = MU_u / duration
		return MU_u


	def compute_MU_employed(self, sol, t_end = 26):
		"""
		Computes the weighted average of marginal utilities of employed people
		Input:
			sol: Instance of the Solution class
			t_end: periods over which the average is taken
		Output:
			MU_e: weighted average of marginal utilities
		"""

		MU_e = 0
		mu_e = 0
		a = self.model.initial_asset
		duration = 0
		survival = 1
		exit_prob_arr = np.zeros(t_end)
		mu_arr = np.zeros(t_end)

		for t in range(t_end):

			# search effort
			search = sol.solve_search(a, t)
			exit_prob_arr[t] = survival * search
			survival = survival * (1 - search)
			duration += survival

			# consumption
			c_e = sol.compute_consumption_employed_exact(a, t)
			mu_arr[t] = self.model.mutil_c(c_e)
			c_u = np.interp(a, self.model.asset_grid, sol.cons_u[t, :])

			# marginal util
			## average over s
			mu_e = np.dot(mu_arr, exit_prob_arr) / exit_prob_arr.sum()

			## average over time
			MU_e += (1 - survival) * mu_e

			# update asset
			a = (1 + self.model.r) * (a - c_u + self.model.income(t = t, emp = 0))

			# avoid too small assets that cause errors
			if survival <= 1e-4:
				t_end = t
				break

		MU_e = MU_e / (t_end - duration)
		return MU_e


	def compute_welfare_effect_unemployed(self, b_u, b_e):
		"""
		Compute consumption smoothing gain and moral hazard cost associated with changes in UI benefits
		Input:
			b_u: UI benefits
			b_e: work incentives
		Output:
			CS: consumption smoothing gain
			FE: fiscal externality (moral hazard cost)
		"""
		tau = self.compute_tax(b_u, b_e)
		self.model.tau, self.model.b_u, self.model.b_e = tau, b_u, b_e
		sol = Solution(self.model)
		sol.solve()

		T, T_B = self.model.T, self.model.T_B

		MU_u = self.compute_MU_unemployed(sol, t_end = T_B)
		MU_e = self.compute_MU_employed(sol, t_end = T)
		CS = (MU_u - MU_e) / MU_e

		D_B = sol.compute_duration(compensated = True)
		D_B_deriv = sol.compute_duration_derivative(emp = 0, compensated = True)
		D_deriv = sol.compute_duration_derivative(emp = 0, compensated = False)

		FE = (D_B_deriv * (b_u - b_e) + D_deriv * tau) / D_B

		return CS, FE

	def compute_welfare_effect_employed(self, b_u, b_e):
		"""
		Compute consumption smoothing gain and moral hazard cost associated with changes in work incentives
		Input:
			b_u: UI benefits
			b_e: work incentives
		Output:
			CS: consumption smoothing gain
			FE: fiscal externality (moral hazard cost)
		"""
		tau = self.compute_tax(b_u, b_e)
		self.model.tau, self.model.b_u, self.model.b_e = tau, b_u, b_e
		sol = Solution(self.model)
		sol.solve()

		T, T_B = self.model.T, self.model.T_B

		MU_e_short = self.compute_MU_employed(sol, t_end = T_B)
		MU_e_long = self.compute_MU_employed(sol, t_end = T)
		CS = (MU_e_short - MU_e_long) / MU_e_long

		D_B = sol.compute_duration(compensated = True)
		D_B_deriv = sol.compute_duration_derivative(emp = 1, compensated = True)
		D_deriv = sol.compute_duration_derivative(emp = 1, compensated = False)

		FE = (D_B_deriv * (b_u - b_e) + D_deriv * tau) / (T_B - D_B)

		return CS, FE		
