import numpy as np
import cv2
from itertools import combinations_with_replacement
from collections import defaultdict
from numpy.linalg import inv

'''
J : haze free image
I : hazy image
t : 0 for haze free image and 1 for maximum hazy image
'''

class Dehazer:
	def __init__(self,img,omega=0.95,t0=0.1,w=5,w_refine=40,epsilon=1e-2,refine=True,threshold_percentage=0.001,show_intermediate=False):
		self.image = img # Corresponds to I in the paper
		self.omega = omega
		self.t0 = t0
		self.w = w
		self.w_refine = w_refine
		self.epsilon = epsilon
		self.refine = refine
		self.show_intermediate = show_intermediate
		self.threshold_percentage = threshold_percentage

	def get_dark_channel(self):
		'''
		w : window size
		Add docstring
		'''
		w = self.w
		self.dark_channel = np.zeros((self.image.shape[0],self.image.shape[1],1))
		padded_image = np.pad(self.image,((int(w/2),int(w/2)),(int(w/2),int(w/2)),(0,0)),'edge')

		for i in range(int(w/2),padded_image.shape[0] - int(w/2)):
			for j in range(int(w/2),padded_image.shape[1] - int(w/2)):
				block = padded_image[i-int(w/2):i+int(w/2)+1,j-int(w/2):j+int(w/2)+1,:]
				self.dark_channel[i-int(w/2),j-int(w/2),0] = np.min(block)

		# if self.show_intermediate is True:
		# 	cv2.imshow('Dark Channel Image',self.dark_channel)
		# 	cv2.waitKey(0)

	def get_A(self):
		'''
		Add docstring
		'''
		# TODO : See the computationally efficient implementation

		self.get_dark_channel()

		# Calculate distribution 
		frequencies = np.zeros(256).astype(int)
		for i in range(0,self.dark_channel.shape[0]):
			for j in range(0,self.dark_channel.shape[1]):
				frequencies[int(self.dark_channel[i,j,0])]+=1

		assert sum(frequencies) == self.image.shape[0]*self.image.shape[1]
		
		# print('Frequencies : {}'.format(frequencies))

		# Find the top 1% brightest pixels in the dark channel image
		threshold = self.threshold_percentage*sum(frequencies)
		# print('Threshold : {}'.format(threshold))

		rev_freq = np.flip(frequencies,axis=0)
		cum_sum = np.cumsum(rev_freq)
		index = np.array(np.where(cum_sum <= threshold)).reshape(-1) # Stores the gray values which are in the top threshold percentage

		top_gray_values = []
		sum_so_far = 0

		for i in range(len(cum_sum)):
			if rev_freq[i] == 0:
				continue
			sum_so_far+=rev_freq[i]
			top_gray_values.append(255 - i)
			if sum_so_far > threshold:
				break

		# # Seperate gray values not present in the dark channel image
		# for i in index:
		# 	if rev_freq[i] == 0:
		# 		continue
		# 	top_gray_values.append(255 - i)
		
		# print('Top Gray Values : {}'.format(top_gray_values))
		A = np.zeros((1,3))

		max_in_b = max_in_g = max_in_r = -1.0*float('inf')

		# Find maximum for each channel across these values
		for tg in top_gray_values:
			indices = np.where(self.dark_channel == tg)
			iterater_ = zip(indices[0],indices[1])
	
			for i,j in iterater_:
				c = 0 # Channel under consideration
				if(self.image[i,j,c] > max_in_b):
					max_in_b = self.image[i,j,c]
				c = 1
				if(self.image[i,j,c] > max_in_g):
					max_in_g = self.image[i,j,c]
				c = 2
				if(self.image[i,j,c] > max_in_r):
					max_in_r = self.image[i,j,c]

		A[0,0] = max_in_b
		A[0,1] = max_in_g
		A[0,2] = max_in_r

		self.A = A.reshape(1,1,3)

		if self.show_intermediate is True:
			print(A)

	def get_transmission_map(self):
		self.get_A()
		normalized_haze_image = self.image/self.A
		w = self.w

		self.transmission_map = np.zeros((normalized_haze_image.shape[0],normalized_haze_image.shape[1],1))
		padded_image = np.pad(normalized_haze_image,((int(w/2),int(w/2)),(int(w/2),int(w/2)),(0,0)),'edge')

		for i in range(int(w/2),padded_image.shape[0] - int(w/2)):
			for j in range(int(w/2),padded_image.shape[1] - int(w/2)):
				block = padded_image[i-int(w/2):i+int(w/2)+1,j-int(w/2):j+int(w/2)+1,:]
				self.transmission_map[i-int(w/2),j-int(w/2),0] = 1.0 - self.omega*np.min(block)

		if self.show_intermediate is True:
			print(self.transmission_map)
			cv2.imshow('Transmission Map',self.transmission_map)
			cv2.waitKey(0)

	def mean_filter(self,I, w):
		kernel = np.ones((w,w),np.float32)/(1.0*w*w)
		return cv2.filter2D(I,-1,kernel)

	def guided_filter(self,I, p):
	    """Refine a filter under the guidance of another (RGB) image.
	    Parameters
	    -----------
	    I:   an M * N * 3 RGB image for guidance.
	    p:   the M * N filter to be guided
	    r:   the radius of the guidance
	    eps: epsilon for the guided filter
	    Return
	    -----------
	    The guided filter.
	    """
	    R,G,B = 0, 1, 2
	    w = self.w_refine
	    eps = self.epsilon

	    p = p.reshape(p.shape[0],p.shape[1])
	    M, N = self.image.shape[0],self.image.shape[1]
	    # base = self.mean_filter(np.ones((M, N)), w)

	    # print('Base : {}'.format(base))
	    # means = [self.mean_filter(I[:, :, i], w) / base for i in range(3)]
	    # print(means)

	    # each channel of I filtered with the mean filter
	    means = [self.mean_filter(I[:, :, i], w) for i in range(3)] # mu_k
	    # print(means)

	    # p filtered with the mean filter
	    mean_p = self.mean_filter(p, w) # p_k_bar
	    # filter I with p then filter it with the mean filter
	    means_IP = [self.mean_filter(I[:, :, i] * p, w) for i in range(3)] # First term in second bracket of equation 14
	    # covariance of (I, p) in each local patch
	    covIP = [means_IP[i] - means[i] * mean_p for i in range(3)] # Second term in equation 14

	    '''
	    Equation 14 describes an image as the second term...the summation is nothing but a mean filter applied to every pixel
	    '''

	    # variance of I in each local patch: the matrix Sigma in eq.14
	    var = defaultdict(dict)
	    for i, j in combinations_with_replacement(range(3), 2):
	        var[i][j] = self.mean_filter(
	            I[:, :, i] * I[:, :, j], w) - means[i] * means[j] # Matrix Sigma terms for each pixel

	    a = np.zeros((M, N, 3))
	    for y, x in np.ndindex(M, N):
	        #         rr, rg, rb
	        # Sigma = rg, gg, gb
	        #         rb, gb, bb
	        Sigma = np.array([[var[R][R][y, x], var[R][G][y, x], var[R][B][y, x]],
	                          [var[R][G][y, x], var[G][G][y, x], var[G][B][y, x]],
	                          [var[R][B][y, x], var[G][B][y, x], var[B][B][y, x]]])
	        cov = np.array([c[y, x] for c in covIP])
	        a[y, x] = np.dot(cov, inv(Sigma + eps * np.eye(3)))  # eq 14
	    
	    b = mean_p - a[:, :, R] * means[R] - a[:, :, G] * means[G] - a[:, :, B] * means[B] # Equation 15

	    q = (self.mean_filter(a[:, :, R], w) * I[:, :, R] + self.mean_filter(a[:, :, G], w) *
	         I[:, :, G] + self.mean_filter(a[:, :, B], w) * I[:, :, B] + self.mean_filter(b, w)) # Equations 7,8,16

	    self.refined_transmission_map = q.reshape(q.shape[0],q.shape[1],1)

	def refine_transmission_map(self):
		normalized_image = (self.image - self.image.min())/(self.image.max() - self.image.min())
		self.guided_filter(normalized_image,self.transmission_map)

		if self.show_intermediate is True:
			print(self.refined_transmission_map)
			cv2.imshow('Refined Transmission Map',self.refined_transmission_map)
			cv2.waitKey(0)

	def dehaze(self):
		self.get_transmission_map()

		if self.refine is True:
			self.refine_transmission_map()

		# Normalizing I and A
		self.image = (self.image - np.min(self.image))/(np.max(self.image) - np.min(self.image))
		self.A/=255.0

		if self.refine is True:
			self.J = (self.image - self.A)/(np.maximum(self.refined_transmission_map,self.t0)) + self.A
		else:
			self.J = (self.image - self.A)/(np.maximum(self.transmission_map,self.t0)) + self.A

		if self.show_intermediate is True:
			print(self.J)
			cv2.imshow('Dehazed Image',self.J)
			cv2.waitKey(0)

		return self.J*255.0