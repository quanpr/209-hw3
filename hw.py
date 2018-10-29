import numpy as np
import pdb

class robot:
	def __init__(self):
		# x, y, theta
		# theta within the range of [0, 2pi)
		self.state_mean = np.zeros((3,1))
		self.state_cov = np.zeros((3,3))
		# x, y, theta
		self.gt_state = np.zeros((3,1))
		#self.time_update_noise_cov = np.zeros((3,3))
		#self.ob_update_noise_cov = np.zeros((3,3))
		self.time_duration = 1
		self.Dx, self.Dy = 750, 500

	def time_update_state_matrix(self, action):
		matrix = np.zeros((3,3))
		vt, _ = action
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
		x, y, theta = state
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
		x, y, theta = state
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
		return function[region](x,y)

	def ob_update

