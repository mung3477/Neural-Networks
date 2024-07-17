import torch

# From https://medium.com/machine-learning-researcher/boltzmann-machine-c2ce76d94da5

class RBM:
	def __init__(self, num_visible, num_hidden):
		self.num_visible = num_visible
		self.num_hidden = num_hidden
		self.weights = torch.randn(num_hidden, num_visible) * 1e-2
		self.visible_bias = torch.zeros(1, num_visible)
		self.hidden_bias = torch.zeros(1, num_hidden)

	def sample_hidden(self, v: torch.Tensor):
		activation = torch.mm(v, self.weights.T) + self.hidden_bias
		prob_h_is_on_given_v = torch.sigmoid(activation)
		return prob_h_is_on_given_v, torch.bernoulli(prob_h_is_on_given_v)

	def sample_visible(self, h: torch.Tensor):
		activation = torch.mm(h, self.weights) + self.visible_bias
		prob_v_is_on_given_h = torch.sigmoid(activation)
		return prob_v_is_on_given_h, torch.bernoulli(prob_v_is_on_given_h)

	def update(self, v: torch.Tensor, p_h: torch.Tensor, v_k: torch.Tensor, p_h_k: torch.Tensor):
		self.weights += torch.mm(p_h.T, v) - torch.mm(p_h_k.T, v_k)

		# https://stats.stackexchange.com/a/191119
		self.visible_bias += torch.sum(v - v_k)
		self.hidden_bias += torch.sum(p_h - p_h_k)

