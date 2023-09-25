import os
import cv2
import configparser
import numpy as np
import random

class DataReader():
	"""docstring for DataReader"""
	def __init__(self):
		super(DataReader, self).__init__()
		config = configparser.ConfigParser()
		config.read('config.ini')

		self.width = int(config['Image']['width'])
		self.height = int(config['Image']['height'])
		self.m = int(config['Image']['m'])
		self.n = int(config['Image']['n'])
		self.num_channels = int(config['Image']['num_channels'])
		self.data_dir = config['Data']['data_dir']
		self.train = config['Data']['train']
		self.test = config['Data']['test']
		self.classes = eval(config['Data']['classes'])
		self.epochs = int(config['Train']['epochs'])
		self.batch_size = int(config['Train']['batch_size'])
		self.checkpoint = config['Train']['checkpoints']
		
		# training data
		self.all_train_data = []
		path = self.train
		with open(path) as my_file:
			for line in my_file:
				line = line.strip()
				split = line.split(',')
				self.all_train_data.append((split[0], split[1]))

		# testing data
		self.all_test_data = []
		path = self.test
		with open(path) as my_file:
			for line in my_file:
				line = line.strip()
				split = line.split(',')
				self.all_test_data.append((split[0], split[1]))


	def read_in_batch(self, training=True):
		if training:
			all_data = self.all_train_data
		else:
			all_data = self.all_test_data

		batches = random.sample(range(0, len(all_data)), self.batch_size)
		imgs = np.zeros((self.batch_size, self.m, self.n, self.height, self.width, self.num_channels))
		labels = np.array([self.classes.index(all_data[batch][1]) for batch in batches])

		for i,batch in enumerate(batches):
			for row in range(self.m):
				for col in range(self.n):
					path = os.path.join(self.data_dir, str(all_data[batch][0]), f"{row}_{col}.png")
					imgs[i][row][col] = cv2.imread(path)

		return imgs, labels
