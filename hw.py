import numpy as np
import pdb

class robot:
	def __init__(self):
		# x mm, y mm, theta radius
		# theta within the range of [0, 2pi)
		self.state_mean = np.zeros((3,1))
		self.state_cov = np.zeros((3,3))
		# x, y, theta
		self.gt_state = np.zeros((3,1))
		#self.time_update_noise_cov = np.zeros((3,3))
		#self.ob_update_noise_cov = np.zeros((3,3))
		self.time_duration = 1
		self.Dx, self.Dy = 750, 500
		self.action = np.zeros((2,1))   #Wr, Wl

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

	def next_state(self,state,action, add_noise = False):
		#old_state = self.gt_state
		new_state= np.zeros((3,1),dtype = float)
		noise = np.zeros((2,1))

		if(add_noise == True):
			noise[0] += np.random.normal(0,3.65)  #set 0.05 percent error speed to 3*sigma 73mm/step
			noise[1] += np.random.normal(0,0.36) #bia mean

		v = (action[0]+action[1]) /2 * 40
		phi = (action[0]-action[1]) * 20/85 #radius

		print(state)
		# print(state[0][0] -(v +noise[0]) * np.cos(state[2]))
		# print(state[1][0] - (v +noise[0]) * np.sin(state[2]))
		# print(state[2][0] + (phi + noise[1]))
		new_state[0][0] = state[0] - (v +noise[0]) * np.cos(state[2])
		new_state[1][0] = state[1] - (v +noise[0]) * np.sin(state[2])
		new_state[2][0] = state[2] + (phi + noise[1])  #rotation ratial

		while (new_state[2] >= 2 * np.pi ):
			new_state[2] -= 2 * np.pi
		while (new_state[2] <= 0 ):
			new_state[2] += 2 * np.pi

		if(new_state[0][0] <= 0.0):  # +42.5
			new_state[0][0] = 0.0
		if(new_state[0][0] >= 750.0): # -42.5
			new_state[0][0] = 750.0

		if(new_state[1][0] <= 0): # +42.5
			new_state[1][0] = 0.0
		if(new_state[1][0] >= 500.0): # -42.5
			new_state[1][0] = 500.0
		#time += 1
		#self.gt_state += [x_move,y_move,theta_turn]
		return new_state
	def observatoin()
	def ob_update_state_matrix(self, state):
		x, y, theta = state
		region = self.find_region(state)
		Dx, Dy = self.Dx, self.Dy

		state_matrix = {}
		state_matrix[1] = np.zeros((3,3))
		state_matrix[1][0][0] = 1/np.cos(theta)
		state_matrix[1][0][2] = x*np.sin(theta)/(np.cos(theta)**2)
		state_matrix[1][1][1] = 1/np.sin(theta)
		state_matrix[1][1][2] = -y*np.cos(theta)/(np.sin(theta)**2)

		state_matrix[2] = np.zeros((3,3))
		state_matrix[2][1][0] = 1/np.cos(theta)
		state_matrix[2][1][2] = x*np.sin(theta)/(np.cos(theta)**2)
		state_matrix[2][0][1] = 1/np.sin(theta)
		state_matrix[2][0][2] = -y*np.cos(theta)/(np.sin(theta)**2)

		state_matrix[3] = np.zeros((3,3))
		state_matrix[3][0][0] = -1/np.cos(theta)
		state_matrix[3][0][2] = (Dx-x)*np.sin(theta)/(np.cos(theta)**2)
		state_matrix[3][1][1] = 1/np.sin(theta)
		state_matrix[3][1][2] = -y*np.cos(theta)/(np.sin(theta)**2)

		state_matrix[4] = np.zeros((3,3))
		state_matrix[4][1][0] = 1/np.cos(theta)
		state_matrix[4][1][2] = -(Dx-x)*np.sin(theta)/(np.cos(theta)**2)
		state_matrix[4][0][1] = 1/np.sin(theta)
		state_matrix[4][0][2] = (Dy-y)*np.cos(theta)/(np.sin(theta)**2)

		return state_matrix[region]

	def ob_update_noise_matrix(self):
		return np.identiy(3)

if __name__ == '__main__':
	robot = robot()
	state = [0.0,0.0,np.pi]
	degree =370
	wr = degree * np.pi / 180.0*85/40

	speed = 50.0 # 50mm
	wl = speed / 40.0

	new_state = robot.next_state(state,[wr,-wr],False)
	print(new_state)
