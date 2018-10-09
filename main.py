from dehaze import Dehazer
import argparse
import os 
import cv2

def parse_arguments():
	## Parse Arguments ###
	parser = argparse.ArgumentParser()
	parser.add_argument('path',type=str,help='Path for the image or folder of images',default='')
	args = parser.parse_args()
	print(args.path)
	return args

def main():
	args = parse_arguments()
	path = args.path 

	is_single_image = False
	image_extetnsions = ['.png','.jpg','.jpeg','.ppm','.gif']
	for extension in image_extetnsions:
		if extension in path: #Means a single image is given as the argument
			is_single_image = True
			try:
				dehazer = Dehazer(cv2.imread(path,1),w=3,show_intermediate=False)
			except:
				print('Error Reading The Image From The Path')

			dehazed = dehazer.dehaze() #Get the dehazed image
			
			if not os.path.exists('Single_Image_Output/'):
				os.mkdir('Single_Image_Output')

			# Seperate the image name from the sepcified path
			last_index_of_backlashash = 0

			for pos,char in enumerate(path):
				if char == '/':
					if pos > last_index_of_backlashash:
						last_index_of_backlashash = pos
			# print(pos)
			image_name = path[last_index_of_backlashash:]
			cv2.imwrite('Single_Image_Output/' + image_name.strip('/'),dehazed)
			print('Written the image to the Single_Image_Output directory')
			break

	if is_single_image is False: # Means a directory was passed as the argument
		if not os.path.exists('Outputs'):
			os.mkdir('Outputs')

		for j in os.listdir(path):
			try:
				img = cv2.imread(path+j,1)
				dehazer = Dehazer(img,w=3,show_intermediate=True)
			except:
				print('Error Dehazing The Images')

			dehazed = dehazer.dehaze()
			cv2.imwrite('Outputs/' + j,dehazed)

		print('Written the Images to the Outputs directory in the specified path')

main()