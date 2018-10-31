import numpy as np
import pdb

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
		# transistion model covariance matrix
		self.transistion_cov = np.zeros((2,2))
		self.transistion_cov[0][0], self.transistion_cov[1][1]  = 3.65**2, 0.086**2
		# observation model covariance matrix
		self.ob_cov = np.zeros((3,3))
		self.ob_cov[0][0], self.ob_cov[1][1], self.ob_cov[2][2] = 9**2, 9**2, 0.0021**2 
		self.Dx, self.Dy = 750, 500

	def time_update_state_matrix(self, action):
		matrix = np.zeros((3,3))
		vt, _ = action[0][0]
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

		if 0 <= theta <= theta1 or 2*np.pi-theta2 <= theta <= 2*pi:
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

	def ob_update_state_matrix(self, state):
		x, y, theta = state[0][0], state[1][0], state[2][0]
		region = self.find_region(state)
		Dx, Dy = self.Dx, self.Dy

		state_matrix = {}
		if region == 1:
			state_matrix[1] = np.zeros((3,3))
			state_matrix[1][0][0] = 1/np.cos(theta)
			state_matrix[1][0][2] = x*np.sin(theta)/(np.cos(theta)**2)
			state_matrix[1][1][1] = 1/np.sin(theta)
			state_matrix[1][1][2] = (Dy-y)*np.cos(theta)/(np.sin(theta)**2)
			state_matrix[1][2][2] = 1.0
		elif region == 2:
			state_matrix[2] = np.zeros((3,3))
			state_matrix[2][1][0] = 1/np.cos(theta)
			state_matrix[2][1][2] = x*np.sin(theta)/(np.cos(theta)**2)
			state_matrix[2][0][1] = 1/np.sin(theta)
			state_matrix[2][0][2] = -y*np.cos(theta)/(np.sin(theta)**2)	
			state_matrix[2][2][2] = 1.0	
		elif region == 3:
			state_matrix[3] = np.zeros((3,3))
			state_matrix[3][0][0] = 1/np.cos(theta)
			state_matrix[3][0][2] = -(Dx-x)*np.sin(theta)/(np.cos(theta)**2)
			state_matrix[3][1][1] = 1/np.sin(theta)
			state_matrix[3][1][2] = -y*np.cos(theta)/(np.sin(theta)**2)
			state_matrix[3][2][2] = 1.0
		else:
			state_matrix[4] = np.zeros((3,3))
			state_matrix[4][1][0] = 1/np.cos(theta)
			state_matrix[4][1][2] = -(Dx-x)*np.sin(theta)/(np.cos(theta)**2)
			state_matrix[4][0][1] = 1/np.sin(theta)
			state_matrix[4][0][2] = (Dy-y)*np.cos(theta)/(np.sin(theta)**2)
			state_matrix[4][2][2] = 1.0

		return state_matrix[region]

	def ob_update_noise_matrix(self):
		return np.identiy(3)

if __name__ == '__main__':
	state_mean, state_cov, gt_state = np.zeros((3,1)), np.zeros((3,3)), np.zeros((3,1))
	state_mean[0][0], state_mean[1][0], state_mean[2][0] = 50, 50, 0
	state_cov[0][0], state_cov[1][1], state_cov[2][2] = 2, 2, 0.5
	gt_state[0][0], gt_state[1][0], gt_state[2][0] = 50, 50, 0
	action = np.zeros((2,1))
	action[0][0] = 5
	robot = robot(state_mean, state_cov, gt_state)
	robot.distance_function(robot.gt_state)
	robot.ob_update_state_matrix(robot.gt_state)
	pdb.set_trace()