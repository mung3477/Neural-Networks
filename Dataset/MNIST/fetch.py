import os

import numpy as np


def fetch_MNIST(url: str = "./Dataset/MNIST/train-images-idx3-ubyte"):
	assert os.path.exists(url), "File not found"

	with open(url, "rb") as f:
		dat = f.read()

	return (np.frombuffer(dat, dtype=np.uint8).copy())[0x10:].reshape((-1,784))
