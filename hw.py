import numpy as np
import pdb
from copy import deepcopy

class robot:
	def __init__(self, state_mean = np.zeros((3,1)), state_cov = np.zeros((3,3)), gt_state = np.zeros((3,1))):
		# x, y, theta
		# theta within the range of [0, 2pi)
		self.state_mean = state_mean
		self.state_cov = state_cov
		# x, y, theta
		self.gt_state = gt_state
		#self.time_update_noise_cov = np.zeros((3,3))
		#self.ob_update_noise_cov = np.zeros((3,3))
		self.time_duration = 1
		self.time_pass = 0.0
		# transistion model covariance matrix
		self.transistion_cov = np.zeros((2,2))
		self.transistion_cov[0][0], self.transistion_cov[1][1]  = 3.65**2, 0.086**2
		# observation model covariance matrix
		self.ob_cov = np.zeros((3,3))
		self.ob_cov[0][0], self.ob_cov[1][1], self.ob_cov[2][2] = 9**2, 9**2, 0.0021**2
		self.Dx, self.Dy = 750, 500

	def time_update_state_matrix(self, action):
		matrix = np.zeros((3,3))
		vt = action[0][0]
		matrix[0][0], matrix[1][1], matrix[2][2]  = 1.0, 1.0, 1.0
		theta = self.gt_state[2][0]
		matrix[0][2] = vt*np.sin(theta)
		matrix[1][2] = -vt*np.cos(theta)
		return matrix

	def time_update_noise_matrix(self):
		theta = self.gt_state[2][0]
		matrix = np.zeros((3,2))
		matrix[0][0] = -np.cos(theta)
		matrix[1][0] = -np.sin(theta)
		matrix[2][1] = 1
		return matrix

	def find_region(self, state):
		#x, y, theta = self.gt_state[0][0], self.gt_state[1][0], self.gt_state[2][0]
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

	def next_state(self,state,action, add_noise = False):
		#old_state = self.gt_state
		new_state= np.zeros((3,1),dtype = float)
		noise = np.zeros((2,1))

		if(add_noise == True):
			noise[0][0] += np.random.normal(0,3.65)  #set 0.05 percent for std
			noise[1][0] += np.random.normal(0,0.0021) #motion noise in radius

		v = (action[0][0]+action[1][0]) /2 * 40
		phi = (action[0][0]-action[1][0]) * 20/85 #radius

		new_state[0][0] = state[0][0] - (v +noise[0][0]) * np.cos(state[2][0])
		new_state[1][0] = state[1][0] - (v +noise[0][0]) * np.sin(state[2][0])
		new_state[2][0] = state[2][0] + (phi + noise[1][0])  #rotation ratial

		#apply boundary condition and normalize for theta, x, y

		new_state[2][0] %= 2 * np.pi


		if(new_state[0][0] <= 42.5):  # +42.5
			new_state[0][0] = 42.5
		if(new_state[0][0] >= 707.5): # -42.5
			new_state[0][0] = 707.5

		if(new_state[1][0] <= 42.5): # +42.5
			new_state[1][0] = 42.5
		if(new_state[1][0] >= 457.5): # -42.5
			new_state[1][0] = 457.5
		#time += 1
		#self.gt_state += [x_move,y_move,theta_turn]
		return new_state

	def generate_observation(self,state,add_noise = False):
		observ = np.zeros((3,1))

		self.distance_function(state)
		observ[0][0]=self.distance_function(state) #front sensor
		right_sensor_state = deepcopy(state)
		right_sensor_state[2][0] += - np.pi/2
		right_sensor_state[2][0] %= 2*np.pi
		observ[1][0]= self.distance_function(right_sensor_state) #right sensor
		observ[2][0] = state[2][0]

		if(add_noise == True):
			noise = np.zeros((3,1))
			noise[0][0] = np.random.normal(0,9)
			noise[1][0] = np.random.normal(0,9)
			noise[2][0] = np.random.normal(self.time_pass * 0.0014,0.0021)
			observ += noise
			#self.time_pass += 1  #add time when observation is takend by robot
		return observ

	def ob_update_state_matrix(self, state):
		x, y, theta = state[0][0], state[1][0], state[2][0]
		region = self.find_region(state)
		Dx, Dy = self.Dx, self.Dy
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

	def time_update(self, action):
		self.state_mean = self.next_state(state=self.state_mean, action=action, add_noise=False)
		Ft = self.time_update_state_matrix(action)
		Wt = self.time_update_noise_matrix()
		self.state_cov = np.dot(Ft, np.dot(self.state_cov, np.transpose(Ft)))+\
							np.dot(Wt, np.dot(self.transistion_cov, np.transpose(Wt)))

	def observation_update(self):
		Ht = self.ob_update_state_matrix(self.gt_state)
		R = self.ob_cov
		estimated_ob = self.generate_observation(self.state_mean, add_noise=False)
		real_ob = self.generate_observation(self.gt_state, add_noise=True)
		K = np.dot(np.transpose(Ht), np.linalg.pinv((np.dot(Ht, np.dot(self.state_cov, np.transpose(Ht)))+R)))
		#pdb.set_trace()
		self.state_mean = self.state_mean + np.dot(np.dot(self.state_cov, K), real_ob - estimated_ob)
		self.state_cov = self.state_cov - np.dot(np.dot(self.state_cov, K), np.dot(Ht, self.state_cov))

		# clamp the state_mean
		# x between (42.5, 707.5)
		# y between (42.5, 407.5)
		# theta between (0, 2xpi)
		self.state_mean[0][0] = max(0, min(707.5, self.state_mean[0][0]))
		self.state_mean[1][0] = max(0, min(407.5, self.state_mean[1][0]))
		self.state_mean[2][0] %= 2*np.pi

	def action_pair_generate(self,move,rotate): #input mm/s and degree in rotation
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
def test_case1(robot):
	state = np.array([[375]
					,[250],
					[0]])
	robot.gt_state = state
	action_pair_array = [robot.action_pair_generate(0,45),
						robot.action_pair_generate(0,45),
						robot.action_pair_generate(0,45),
						robot.action_pair_generate(0,45),
						robot.action_pair_generate(0,45),
						robot.action_pair_generate(0,30)]

	for a in action_pair_array:
		robot.gt_state = robot.next_state(robot.gt_state,a,False)
		print(robot.gt_state)
		print(robot.generate_observation(robot.gt_state,False))
		print('\n')
		#robot.timeupdate
		#robot.observationupdate

def test_case2(robot,state):
	robot.gt_state = state
	action_pair_array = [robot.action_pair_generate(30,0),
						robot.action_pair_generate(0,90),
						robot.action_pair_generate(30,0),
						robot.action_pair_generate(30,0),
						robot.action_pair_generate(0,-90),
						robot.action_pair_generate(30,0)]

	for a in action_pair_array:
		robot.gt_state = robot.next_state(robot.gt_state,a,True)
		robot.time_update(a)
		for b in range(1):
			robot.observation_update()

		#robot.timeupdate
		#robot.observationupdate
		print(robot.gt_state)
		print(robot.state_mean)
		print(robot.state_cov)
		print('\n')

###
if __name__ == '__main__':

	state = np.array([[375]
			,[250],
			[np.pi]])
	state_cov = np.zeros((3,3))
	state_cov[0][0], state_cov[1][1], state_cov[2][2] = 1, 1, np.pi
	robot = robot(state,state_cov)
	test_case2(robot,state)

# if __name__ == '__main__':
# 	state_mean, state_cov, gt_state = np.zeros((3,1)), np.zeros((3,3)), np.zeros((3,1))
# 	state_mean[0][0], state_mean[1][0], state_mean[2][0] = 20, 20, np.pi/6
# 	#state_cov[0][0], state_cov[1][1], state_cov[2][2] = 2, 2, 0.5
# 	state_cov[0][0], state_cov[1][1], state_cov[2][2] = 10000, 10000, np.pi/2
# 	gt_state[0][0], gt_state[1][0], gt_state[2][0] = 100, 100, np.pi/2
# 	action = np.zeros((2,1))
# 	#action[0][0] = 0
# 	robot = robot(state_mean, state_cov, gt_state)
# 	i = 0
# 	while True:
# 		robot.observation_update()
# 		i += 1
# 		if i % 100 == 0:
# 			print(robot.state_cov, '\r\n', robot.state_mean)
# 			pdb.set_trace()

