# Originated from https://towardsdatascience.com/hopfield-networks-neural-memory-machines-4c94be821073

#for MNIST fetch
import requests, gzip, os, hashlib
import numpy as np
import matplotlib.pyplot as plt
import pygame

from .hopfield_network import HopfieldNetwork

#Fetch MNIST dataset from the ~SOURCE~
def fetch_MNIST(url):
	assert os.path.exists(url), "File not found"

	with open(url, "rb") as f:
		dat = f.read()

	return np.frombuffer(dat, dtype=np.uint8).copy()

def MNIST_Hopfield():
	#test out the Hopfield_Network object on some MNIST data
	#fetch MNIST dataset for some random memory downloads
	#data from https://www.kaggle.com/datasets/hojjatk/mnist-dataset

	X = fetch_MNIST(
		"./Dataset/MNIST/train-images-idx3-ubyte"
		)[0x10:].reshape((-1,784))
	print("Fetched MNIST dataset", X.shape)

	#convert to binary
	X_binary = np.where(X>20, 1, 0)

	#Snag a memory from computer brain
	memories_list = np.array([X_binary[np.random.randint(len(X))], X_binary[np.random.randint(len(X))]])
	#initialize Hopfield object
	H_Net = HopfieldNetwork(memories_list)
	H_Net.learn()


	#Draw it all out, updating board each update iteration
	cellsize = 20

	pygame.init() #initialize pygame
	#set dimensions of board and cellsize -  28 X 28  ~ special display surface
	surface = pygame.display.set_mode((28*cellsize,28*cellsize))
	pygame.display.set_caption("   ")


	#kill pygame if user exits window
	Running = True
	#main animation loop
	while Running:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				Running = False

				#plot weights matrix
				plt.figure("weights", figsize=(10,7))
				plt.imshow(H_Net.connections,cmap='RdPu') #
				plt.xlabel("Each row/column represents a neuron, each square a connection")

				plt.title(" 4096 Neurons - 16,777,216 unique connections",fontsize=15)
				plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])

				#plot energies
				plt.figure("Energy",figsize=(10,7))
				x = np.arange(len(H_Net.energy_logs))
				plt.scatter(x,np.array(H_Net.energy_logs),s=1,color='red')
				plt.xlabel("Generation")
				plt.ylabel("Energy")
				plt.title("Network Energy over Successive Generations",fontsize=15)
				plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])

		cells = H_Net.state.reshape(28,28).T

		#fills surface with color
		surface.fill((211,211,211))

		#loop through network state array and update colors for each cell
		for r, c in np.ndindex(cells.shape): #iterates through all cells in cells matrix
			if cells[r,c] == -1:
				col = (135,206,250)

			elif cells[r,c] == 1:
				col = (0,0,128)

			else:
				col = (255,140,0)
			pygame.draw.rect(surface, col, (r*cellsize, c*cellsize, \
												cellsize, cellsize)) #draw new cell_

		#update network state
		H_Net.update_states(0.07)
		H_Net.compute_energy()
		pygame.display.update() #updates display from new .draw in update function
		pygame.time.wait(50)

	#quit pygame
	pygame.quit()

if __name__ == "__main__":
	MNIST_Hopfield()
	plt.show()
