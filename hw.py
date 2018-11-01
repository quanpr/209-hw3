import numpy as np
import pdb
from copy import deepcopy
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from statistics import stdev


class robot:
	# Initialization
	# state_mean: robot think where it is
	# state_cov: how certain the robot know where it is
	# gt_state: the robot actual location
	#
	# All state variables are in (x y rotation) order. Unit: x, y in mm; rotation in radian from 0 to 2pi
	# robot coordinate convention: positvie x point South, positive y point East, 0 rotation point to North
	def __init__(self, state_mean = np.zeros((3,1)), state_cov = np.zeros((3,3)), gt_state = np.zeros((3,1))):
		self.state_mean = state_mean
		self.state_cov = state_cov
		self.gt_state = gt_state
		# how long robot stay in stationary
		self.time_pass = 0.0

		# transistion model covariance matrix
		self.transistion_cov = np.zeros((2,2))
		self.transistion_cov[0][0], self.transistion_cov[1][1]  = 3.65**2, 0.086**2

		# observation model covariance matrix
		self.ob_cov = np.zeros((3,3))
		self.ob_cov[0][0], self.ob_cov[1][1], self.ob_cov[2][2] = 9**2, 9**2, 0.0021**2

		# Wall Boundaries
		self.Dx, self.Dy = 750, 500

	# F from Kalman Filter for action in time update
	def time_update_state_matrix(self, action):
		matrix = np.zeros((3,3))
		vt = action[0][0]
		matrix[0][0], matrix[1][1], matrix[2][2]  = 1.0, 1.0, 1.0
		theta = self.gt_state[2][0]
		matrix[0][2] = vt*np.sin(theta)
		matrix[1][2] = -vt*np.cos(theta)
		return matrix

	# W from Kalman Filter for noise in time update
	def time_update_noise_matrix(self):
		theta = self.gt_state[2][0]
		matrix = np.zeros((3,2))
		matrix[0][0] = -np.cos(theta)
		matrix[1][0] = -np.sin(theta)
		matrix[2][1] = 1
		return matrix

	# To determine robot are facing which region (subfunction to generate observation)
	#
	# we divide the map to 4 region according to robot's x, y location
	# in each region, we use different projection equation to calculate measured distance
	def find_region(self, state):
		x, y, theta = state[0][0], state[1][0], state[2][0]
		Dx, Dy = self.Dx, self.Dy
		theta1 = np.arctan(y/x)
		theta2 = np.arctan((Dy-y)/x)
		theta3 = np.arctan(y/(Dx-x))
		theta4 = np.arctan((Dy-y)/(Dx-x))

		if 0 <= theta <= theta1 or 2*np.pi-theta2 <= theta <= 2*np.pi:
			return 1
		elif theta1 <= theta <= np.pi-theta3:
			return 2
		elif np.pi-theta3 <= theta <= np.pi+theta4:
			return 3
		else:
			return 4

	# Determine the distance that range sensor will measured (subfunction to generate observation)
	def distance_function(self, state):
		function = {}
		x, y, theta = state[0][0], state[1][0], state[2][0]
		region = self.find_region(state)
		Dx, Dy = self.Dx, self.Dy
		# cos(x) -> x/cos(theta)
		function[1] = lambda idx: idx[0]/np.cos(theta)
		# cos(90-x) = sin(x) -> y/sin(theta)
		function[2] = lambda idx: idx[1]/np.sin(theta)
		# cos(180-x) = -cos(x) -> (Dx-x)/(-cos(theta))
		function[3] = lambda idx: (Dx-idx[0])/(-np.cos(theta))
		# cos(270-x) = -sin(x) -> (Dy-y)/(-sin(theta))
		function[4] = lambda idx: (Dy-idx[1])/(-np.sin(theta))
		idx = (x,y)
		return function[region](idx)

	# Determine the next position and rotation given action pair commands
	#
	# state: current x, y, rotation
	# action: action pair commands, control both servos independantly
	# add_noise: determine whether a gaussian noise should be added to action
	def next_state(self,state,action, add_noise = False):
		new_state= np.zeros((3,1),dtype = float)
		noise = np.zeros((2,1))

		if(add_noise == True):
			noise[0][0] += np.random.normal(0,3.65)  #set 0.05 percent for std
			noise[1][0] += np.random.normal(0,0.0021) #motion noise in radius

		# velocity in mm/s
		v = (action[0][0]+action[1][0]) /2 * 40

		# angular velocity in rad/s for robot body (not wheel)
		phi = (action[0][0]-action[1][0]) * 20/85 #radius

		# result after action
		new_state[0][0] = state[0][0] - (v +noise[0][0]) * np.cos(state[2][0])
		new_state[1][0] = state[1][0] - (v +noise[0][0]) * np.sin(state[2][0])
		new_state[2][0] = state[2][0] + (phi + noise[1][0])  #rotation ratial

		#apply boundary condition and normalize for rotation, x, y
		new_state[2][0] %= 2 * np.pi

		new_state[0][0] = max(0, min(707.5, new_state[0][0]))
		new_state[1][0] = max(0, min(407.5, new_state[1][0]))

		return new_state

	# generate estimated observation for observation observation update
	# state: current x, y, rotation of robot
	# add_noise: determine whether a gaussian noise should be added to measurement
	def generate_observation(self,state,add_noise = False):
		observ = np.zeros((3,1))

		# deteremine sensor reading without noise
		self.distance_function(state)
		observ[0][0]=self.distance_function(state) #front sensor
		right_sensor_state = deepcopy(state)
		right_sensor_state[2][0] += - np.pi/2
		right_sensor_state[2][0] %= 2*np.pi
		observ[1][0]= self.distance_function(right_sensor_state) #right sensor
		observ[2][0] = state[2][0]

		# add noise to measurement
		if(add_noise == True):
			noise = np.zeros((3,1))
			noise[0][0] = np.random.normal(0,9)
			noise[1][0] = np.random.normal(0,9)
			noise[2][0] = np.random.normal(self.time_pass * 0.0014,0.0021)
			observ += noise
		return observ

	# H from kalman filter for observation update
	# each states will have a cooresponding H matrix from one of the four matrix
	def ob_update_state_matrix(self, state):
		x, y, theta = state[0][0], state[1][0], state[2][0]
		region = self.find_region(state)
		Dx, Dy = self.Dx, self.Dy

		# Assign value for Front sensor into H matrix
		state_matrix = np.zeros((3,3))
		if region == 1:
			state_matrix[0][0] = 1/np.cos(theta)
			state_matrix[0][2] = x*np.sin(theta)/(np.cos(theta)**2)
			state_matrix[2][2] = 1.0
		elif region == 2:
			state_matrix[0][1] = 1/np.sin(theta)
			state_matrix[0][2] = -y*np.cos(theta)/(np.sin(theta)**2)
			state_matrix[2][2] = 1.0
		elif region == 3:
			state_matrix[0][0] = 1/np.cos(theta)
			state_matrix[0][2] = -(Dx-x)*np.sin(theta)/(np.cos(theta)**2)
			state_matrix[2][2] = 1.0
		else:
			state_matrix[0][1] = 1/np.sin(theta)
			state_matrix[0][2] = (Dy-y)*np.cos(theta)/(np.sin(theta)**2)
			state_matrix[2][2] = 1.0

		# Assign value for Right sensor into H matrix
		theta0 = (theta-np.pi/2)%(2*np.pi)
		state0 = deepcopy(state)
		state0[2][0] = theta0
		region0 = self.find_region(state0)

		if region0 == 1:
			state_matrix[1][0] = 1/np.cos(theta0)
			state_matrix[1][2] = x*np.sin(theta0)/(np.cos(theta0)**2)
		elif region0 == 2:
			state_matrix[1][1] = 1/np.sin(theta0)
			state_matrix[1][2] = -y*np.cos(theta0)/(np.sin(theta0)**2)
		elif region0 == 3:
			state_matrix[1][0] = 1/np.cos(theta0)
			state_matrix[1][2] = -(Dx-x)*np.sin(theta0)/(np.cos(theta0)**2)
		else:
			state_matrix[1][1] = 1/np.sin(theta0)
			state_matrix[1][2] = (Dy-y)*np.cos(theta0)/(np.sin(theta0)**2)

		return state_matrix

	def ob_update_noise_matrix(self):
		return np.identiy(3)

	# Time update from Kalman filter
	# update state mean and convariance matrix base on action pair, F and W matrix
	#
	# time update will reset gryo bias
	def time_update(self, action):
		self.state_mean = self.next_state(state=self.state_mean, action=action, add_noise=False)
		Ft = self.time_update_state_matrix(action)
		Wt = self.time_update_noise_matrix()
		self.state_cov = np.dot(Ft, np.dot(self.state_cov, np.transpose(Ft)))+\
							np.dot(Wt, np.dot(self.transistion_cov, np.transpose(Wt)))
		# When time update, we assume it will reset gryo bias since the robot will move
		self.time_pass = 0

	# Observation update from Kalman Filter
	# update state mean and convariance matrix base on observation, expected observation
	# , H and R matrix
	#
	# observation update will increase the effect of gryo bias
	def observation_update(self):
		Ht = self.ob_update_state_matrix(self.gt_state)
		R = self.ob_cov
		estimated_ob = self.generate_observation(self.state_mean, add_noise=False)
		real_ob = self.generate_observation(self.gt_state, add_noise=True)
		K = np.dot(np.transpose(Ht), np.linalg.pinv((np.dot(Ht, np.dot(self.state_cov, np.transpose(Ht)))+R)))
		self.state_mean = self.state_mean + np.dot(np.dot(self.state_cov, K), real_ob - estimated_ob)
		self.state_cov = self.state_cov - np.dot(np.dot(self.state_cov, K), np.dot(Ht, self.state_cov))
		self.time_pass += 1

		# clamp the state_mean
		# x between (42.5, 707.5)
		# y between (42.5, 407.5)
		# theta between (0, 2xpi)
		self.state_mean[0][0] = max(0, min(707.5, self.state_mean[0][0]))
		self.state_mean[1][0] = max(0, min(407.5, self.state_mean[1][0]))
		self.state_mean[2][0] %= 2*np.pi

	# generate action pair in radian
	# input velocity in mm/s and degree in rotation
	def action_pair_generate(self,move,rotate):
			action_pair = np.zeros((2,1))
			if (move >= 74): #max speed
				move = 73
			elif (move <= -74):
				move = -73

			if (rotate > 49): #max rotate speed
				rotate = 49
			elif (rotate < -49):
				rotate = -49

			if (move != 0):
				action_pair[0][0] = action_pair[1][0] = move/40.0
			elif(rotate != 0):
				action_pair[0][0] = rotate * np.pi / 180.0*85/40
				action_pair[1][0] = -action_pair[0][0]
			return action_pair

def test_case4(robot):
	action_pair_array = [robot.action_pair_generate(30,0),
						robot.action_pair_generate(0,90),
						robot.action_pair_generate(30,0),
						robot.action_pair_generate(30,0),
						robot.action_pair_generate(0,-90),
						robot.action_pair_generate(30,0),
						robot.action_pair_generate(50,0),
						robot.action_pair_generate(50,0),
						robot.action_pair_generate(0,45),
						robot.action_pair_generate(30,0),
						robot.action_pair_generate(30,0),
						robot.action_pair_generate(30,0),
						robot.action_pair_generate(0,45),
						robot.action_pair_generate(30,0),
						robot.action_pair_generate(30,0),
						robot.action_pair_generate(30,0),
						robot.action_pair_generate(0,45),
						robot.action_pair_generate(50,0),
						robot.action_pair_generate(50,0),
						robot.action_pair_generate(50,0),
						robot.action_pair_generate(50,0),
						robot.action_pair_generate(0,45),
						robot.action_pair_generate(50,0),
						robot.action_pair_generate(50,0),
						robot.action_pair_generate(50,0),
						robot.action_pair_generate(50,0),
						robot.action_pair_generate(0,45),
						robot.action_pair_generate(50,0),
						robot.action_pair_generate(50,0),
						robot.action_pair_generate(50,0),
						robot.action_pair_generate(0,45),
						robot.action_pair_generate(30,0),
						robot.action_pair_generate(30,0),
						robot.action_pair_generate(30,0),
						robot.action_pair_generate(0,45),
						robot.action_pair_generate(30,0),
						robot.action_pair_generate(30,0),
						robot.action_pair_generate(30,0),
						]

	covariance, mean = [], []
	x_mean, x_cov = [], []
	y_mean, y_cov = [], []
	theta_mean, theta_cov = [], []
	gt = []
	for a in action_pair_array:
		robot.gt_state = robot.next_state(robot.gt_state,a,True)
		robot.time_update(a)
		robot.observation_update()
		gt.append([robot.gt_state[0][0], robot.gt_state[1][0], robot.gt_state[2][0]])

		covariance.append(robot.state_mean)
		mean.append(robot.state_mean)

		x_mean.append(robot.state_mean[0][0])
		y_mean.append(robot.state_mean[1][0])
		theta_mean.append(robot.state_mean[2][0])

		x_cov.append(abs(robot.state_cov[0][0]**0.5))
		y_cov.append(abs(robot.state_cov[1][1]**0.5))
		theta_cov.append(abs(robot.state_cov[2][2]**0.5))

	idx = [i for i in range(0, len(action_pair_array))]
	plt.figure()
	#pdb.set_trace()
	plt.ylabel('uncertainty in y axis')
	plt.xlabel('uncertainty in x axis')
	plt.errorbar(x_mean, y_mean, xerr=x_cov, yerr=y_cov, fmt='o', ecolor='r', capthick=2, label='estimated random variable')
	plt.title("Ground truth trajectory vs estimated trajectory with uncertainty")
	plt.plot([g[0] for g in gt], [g[1] for g in gt], 'g', label='ground truth position')
	plt.legend()
	plt.grid()
	plt.show()

	plt.figure()
	#pdb.set_trace()
	plt.ylabel('uncertainty in orientation')
	plt.xlabel('time duration')
	plt.errorbar(idx, theta_mean, yerr=theta_cov, fmt='o', ecolor='b', capthick=2, label='estimated random variable')
	plt.title("Ground truth orientation vs estimated orientation with uncertainty")
	plt.plot(idx, [g[2] for g in gt], 'g', label='ground truth orientation')
	plt.legend()
	plt.grid()
	plt.show()

def test_case3(robot):
	i = 0
	covariance, mean = [], []
	x_mean, x_cov = [], []
	y_mean, y_cov = [], []
	theta_mean, theta_cov = [], []
	#while i <= 100:
	while i < 20:
		robot.observation_update()
		i += 1
		if i % 2 == 0:
		#if True:
			covariance.append(robot.state_mean)
			mean.append(robot.state_mean)
			#print(robot.state_cov, '\r\n', robot.state_mean)
			#pdb.set_trace()
			x_mean.append(robot.state_mean[0][0])
			y_mean.append(robot.state_mean[1][0])
			theta_mean.append(robot.state_mean[2][0])

			x_cov.append(abs(robot.state_cov[0][0]**0.5))
			y_cov.append(abs(robot.state_cov[1][1]**0.5))
			theta_cov.append(abs(robot.state_cov[2][2]**0.5))

	idx = [i for i in range(1, 11)]
	plt.figure()
	#pdb.set_trace()
	plt.ylabel('uncertainty in y position')
	plt.xlabel('time duration')
	plt.errorbar(idx, y_mean, yerr=y_cov, fmt='o', ecolor='g', capthick=2, label='uncertainty x')
	plt.title("Error bar with ground truth y = 250, initial uncertainty 400 mm")
	#plt.plot(idx, [robot.gt_state[1][0] for _ in idx], label='ground truth position')
	plt.grid()
	plt.show()

	plt.figure()
	#pdb.set_trace()
	plt.ylabel('uncertainty in x position')
	plt.xlabel('time duration')
	plt.errorbar(idx, x_mean, yerr=x_cov, fmt='o', ecolor='r', capthick=2)
	plt.title("Error bar with ground truth x = 375, initial uncertainty 400 mm")
	plt.grid()
	plt.show()

	plt.figure()
	#pdb.set_trace()
	plt.ylabel('uncertainty in orientation')
	plt.xlabel('time duration')
	plt.errorbar(idx, theta_mean, yerr=theta_cov, fmt='o', ecolor='b', capthick=2)
	plt.title("Error bar with ground truth orientation = pi**2, initial uncertainty sqrt(pi/2)")
	plt.grid()
	plt.show()

###
if __name__ == '__main__':
	state_mean, state_cov, gt_state = np.zeros((3,1)), np.zeros((3,3)), np.zeros((3,1))
	#state_mean[0][0], state_mean[1][0], state_mean[2][0] = 375, 250, np.pi/2
	#state_cov[0][0], state_cov[1][1], state_cov[2][2] = 10000, 10000, np.pi
	state_mean[0][0], state_mean[1][0], state_mean[2][0] = 375, 250, np.pi
	state_cov[0][0], state_cov[1][1], state_cov[2][2] = 400**2, 400**2, np.pi**2
	gt_state[0][0], gt_state[1][0], gt_state[2][0] = 375, 250, np.pi/2

	robot = robot(state_mean, state_cov, gt_state)
	#test_case2(robot,state)
	test_case3(robot)
	pdb.set_trace()


