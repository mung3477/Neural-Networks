"""
	Python implementation of the [Hopfield Network]
		: (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC346238/pdf/pnas00447-0135.pdf)
	This code is slightly modified from
			https://towardsdatascience.com/hopfield-networks-neural-memory-machines-4c94be821073
"""

import numpy as np;
class HopfieldNetwork:
	def __init__(self, input: np.ndarray) -> None:

		if input.ndim == 2:
			self.memories = input;
			self.n = self.memories.shape[1];
		elif input.ndim == 1:
			self.memories = np.array([input])
			self.n = input.shape[0];
		else:
			raise ValueError('Input must be an flatten image or a list of flatten images.');

		self.state = np.random.randint(0, 2, (self.n, 1))
		self.connections = np.zeros((self.n, self.n))
		self.energy_logs = list()

	def learn(self) -> None:
		# Storage prescription from the paper
		bipolarized_memories = self.memories * 2 - 1
		self.connections = (1 / self.memories.shape[0]) * bipolarized_memories.T @ bipolarized_memories
		np.fill_diagonal(self.connections, 0)

	def update_states(self, W)->None:
		# Generate random numbers for each neuron
		attempt_probs = np.random.random(self.n)

		# Determine which neurons attempt an update
		# Each neuron i readjusts its state randomly in time but with a mean attempt rate W,
		update_mask = attempt_probs < W
		for neuron in np.nonzero(update_mask)[0]:
			activation = np.dot(self.connections[neuron, :], self.state)
			if activation < 0:
				self.state[neuron] = 0
			else:
				self.state[neuron] = 1

	def compute_energy(self)->None:
		# Compute the energy of the current state
		energy = -0.5 * np.dot(self.state.T, self.connections @ self.state)
		self.energy_logs.append(energy)

