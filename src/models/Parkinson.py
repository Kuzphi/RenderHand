import torch.nn as nn
import torch

# print net.params
__all__ = ['parkinson']
class Parkinson(nn.Module):
	def __init__(self, length, num_class):
		super(Parkinson, self).__init__()
		self.conv1 = nn.Conv1d(3, 128, kernel_size = 7, padding = 3)
		self.conv2 = nn.Conv1d(128, 128, kernel_size = 7, padding = 3)
		self.conv3 = nn.Conv1d(128, 128, kernel_size = 7, padding = 3)
		self.conv4 = nn.Conv1d(128, 128, kernel_size = 7, padding = 3)
		self.conv5 = nn.Conv1d(128, 8, kernel_size = 1)

		self.relu = nn.ReLU()
		self.tanh = nn.Tanh()
		self.pool = nn.MaxPool1d(kernel_size = 2)
		self.fc = nn.Linear(length / 16 * 8, num_class)
	def forward(self, x):
		out = self.relu(self.conv1(x))
		out = self.pool(out)
		out = self.relu(self.conv2(out))
		out = self.pool(out)
		out = self.relu(self.conv3(out))
		out = self.pool(out)
		out = self.relu(self.conv4(out))
		out = self.pool(out)
		out = self.relu(self.conv5(out))
		out = out.view(out.size(0), -1)
		out = self.tanh(self.fc(out))
		return out


def parkinson(length = 2000, num_classes = 5, **kwargs):
	return Parkinson(length, num_classes)
