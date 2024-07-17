import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid

from Dataset import fetch_MNIST

from .rbm import RBM


def train(rbm: RBM, v_0, num_gibbs: int, num_epoch: int = 10):
	v_k = None
	for _ in range(num_epoch):
		# last update of hidden units should use the probability itself
		p_h_0, _ = rbm.sample_hidden(v_0)
		v_k = v_0

		# gibbs sampling
		for _ in range(num_gibbs):
			_, h_k = rbm.sample_hidden(v_k)
			_, v_k = rbm.sample_visible(h_k)

		# last update of hidden units should use the probability itself
		p_h_k, _ = rbm.sample_hidden(v_k)
		rbm.update(v_0, p_h_0, v_k, p_h_k)

	return v_k

def show(img):
	plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
	plt.show()

def MNIST_RBM(num_v: int = 20, n: int = 1):
	X = fetch_MNIST()
	X_binary = np.where(X>20, 1, 0)

	v_0 = torch.Tensor(X_binary[np.random.randint(low=0, high=len(X), size=(num_v))])

	rbm = RBM(784, 128)
	v_k = train(rbm, v_0, n, num_epoch=30)

	show(make_grid(v_0.view(-1, 1, 28, 28).data))
	show(make_grid(v_k.view(-1, 1, 28, 28).data))

if __name__ == "__main__":
	MNIST_RBM()
