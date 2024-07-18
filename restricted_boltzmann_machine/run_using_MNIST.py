import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid

from Dataset import fetch_MNIST

from .rbm import RBM


def train(rbm: RBM, v_0, num_gibbs: int, batch_size: int):
	# last update of hidden units should use the probability itself
	v_k = v_0
	p_h_0, h_k = rbm.sample_hidden(v_0)

	# gibbs sampling
	for _ in range(num_gibbs):
		_, v_k = rbm.sample_visible(h_k)
		p_h_k, h_k = rbm.sample_hidden(v_k)

	rbm.update(v_0, p_h_0, v_k, p_h_k, batch_size=batch_size)

	# train_loss = torch.mean(torch.abs(v_0[v_0>=0] - v_k[v_0>=0]))
	# print('loss: '+str(train_loss))

	return v_k

def show(orig, recon):
	plt.figure()
	_, axs = plt.subplots(1, 2)
	axs[0].imshow(np.transpose(orig.numpy(), (1, 2, 0)))
	axs[1].imshow(np.transpose(recon.numpy(), (1, 2, 0)))
	plt.show()

def MNIST_RBM(batch_size: int = 32, num_epoch: int = 1000):
	X = fetch_MNIST()
	X_binary = np.where(X>20, 1, 0)

	rbm = RBM(784, 128)
	for _ in range(num_epoch):
		v_0 = torch.Tensor(X_binary[np.random.randint(low=0, high=len(X), size=(batch_size))])
		v_k = train(rbm, v_0, num_gibbs=10, batch_size=batch_size)

	show(
		make_grid(v_0.view(-1, 1, 28, 28).data),
		make_grid(v_k.view(-1, 1, 28, 28).data)
	)

if __name__ == "__main__":
	MNIST_RBM()
