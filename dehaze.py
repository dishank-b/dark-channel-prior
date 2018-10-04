import numpy as np
import cv2

'''
J : haze free image
I : hazy image
t : 0 for haze free image and 1 for maximum hazy image
'''

class Dehazer:
	def __init__(self,img,omega=0.95,t0=0.1,w=5,threshold_percentage=0.001,show_intermediate=False):
		self.image = img # Corresponds to I in the paper
		self.omega = omega
		self.t0 = t0
		self.w = w
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

		if self.show_intermediate is True:
			cv2.imshow('Dark Channel Image',self.dark_channel)
			cv2.waitKey(0)

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
		
		# Find the top 1% brightest pixels in the dark channel image
		threshold = self.threshold_percentage*sum(frequencies)
		rev_freq = np.flip(frequencies,axis=0)
		cum_sum = np.cumsum(rev_freq)
		index = np.array(np.where(cum_sum <= threshold)).reshape(-1) # Stores the gray values which are in the top threshold percentage

		top_gray_values = []
		# Seperate gray values not present in the dark channel image
		for i in index:
			if rev_freq[i] == 0:
				continue
			top_gray_values.append(255 - i)
		
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

	def refine_transmission_map(self):
		gray = self.image.min(axis=2) #.reshape(self.image.shape[0],self.image.shape[1],1)
		print(gray.shape)
		self.get_transmission_map()
		t = (self.transmission_map*255.0).astype(np.uint8)
		self.refined_transmission_map = cv2.ximgproc.guidedFilter(gray,self.transmission_map, 40, 1e-2)

	def dehaze(self):
		self.refine_transmission_map()

		# Normalizing I and A
		self.image = (self.image - np.min(self.image))/(np.max(self.image) - np.min(self.image))
		self.A/=255.0

		self.J = (self.image - self.A)/(np.maximum(self.transmission_map,self.t0)) + self.A

		if self.show_intermediate is True:
			print(self.J)
			cv2.imshow('Dehazed Image',self.J)
			cv2.waitKey(0)

		return self.J

def main():
	dehazer = Dehazer(cv2.imread('test.jpeg',1),w=3,show_intermediate=False)
	dehazed = dehazer.dehaze()
	cv2.imshow('Dehazed Image',dehazed)
	cv2.waitKey(0)
main()
