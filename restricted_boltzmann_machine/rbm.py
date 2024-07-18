import torch

# From https://medium.com/machine-learning-researcher/boltzmann-machine-c2ce76d94da5

class RBM:
	def __init__(self, num_visible, num_hidden):
		self.weights = torch.randn(num_visible, num_hidden) * 0.1
		self.visible_bias = torch.zeros(1, num_visible)
		self.hidden_bias = torch.zeros(1, num_hidden)
		self.weight_momentum = torch.zeros(num_visible, num_hidden)
		self.visible_bias_momentum = torch.zeros(num_visible)
		self.hidden_bias_momentum = torch.zeros(num_hidden)

	def sample_hidden(self, v: torch.Tensor):
		prob_h_is_on_given_v = torch.sigmoid(torch.mm(v, self.weights) + self.hidden_bias)
		h = torch.bernoulli(prob_h_is_on_given_v)
		return prob_h_is_on_given_v, h

	def sample_visible(self, h: torch.Tensor):
		prob_v_is_on_given_h = torch.sigmoid(torch.mm(h, self.weights.T) + self.visible_bias)
		v = torch.bernoulli(prob_v_is_on_given_h)
		return prob_v_is_on_given_h, v


	# Since we are using batch gradient descent, we need to divide the gradients by the batch size
	# To increase the speed of training, we use momentum of 0.9
	def update(self, v_0: torch.Tensor, p_h_0: torch.Tensor, v_k: torch.Tensor, p_h_k: torch.Tensor, batch_size: int, momentum=0.9, weight_decay=0.001):
		weight_update = (torch.mm(v_0.T, p_h_0) - torch.mm(v_k.T, p_h_k)) / batch_size - weight_decay * self.weights
		self.weight_momentum = self.weight_momentum * momentum + weight_update
		self.weights += self.weight_momentum

		# https://stats.stackexchange.com/a/191119
		visible_bias_update = torch.sum((v_0 - v_k), 0) / batch_size
		self.visible_bias_momentum = self.visible_bias_momentum * momentum + visible_bias_update
		self.visible_bias += self.visible_bias_momentum

		hidden_bias_update = torch.sum((p_h_0 - p_h_k), 0) / batch_size
		self.hidden_bias_momentum = self.hidden_bias_momentum * momentum + hidden_bias_update
		self.hidden_bias += self.hidden_bias_momentum

