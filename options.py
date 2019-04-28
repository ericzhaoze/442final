import argparse

class init_option():
	def __init__(self):
		self.initialized = False
	def initialize(self):
		parser = argparse.ArgumentParser()
		parser.add_argument("--num_epochs", type=int, default=200, help="number of epochs of training")
		parser.add_argument("--dataroot", type=str, default="cap_data", help="directory of the dataset")
		parser.add_argument("--batch_size", type=int, default=1, help="define batch size")
		parser.add_argument("--lr", type=float, default=0.0002, help="learning rate for optimizer")
		self.initialized = True
		opt = parser.parse_args()
		
		print(opt)

		return opt