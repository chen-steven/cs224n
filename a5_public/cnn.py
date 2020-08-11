#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
class CNN(nn.Module):
	def __init__(self, reshape_size, embed_size, max_word_length, kernel_size=5):
		super(CNN, self).__init__()
		self.convolution = nn.Conv1d(in_channels=reshape_size, out_channels=embed_size, kernel_size=kernel_size, bias=True)
		self.max_pool = nn.MaxPool1d(kernel_size = max_word_length-kernel_size+1)
	def forward(self, x):
		x = F.relu(self.convolution(x))
		x = self.max_pool(x).squeeze()
		return x


### END YOUR CODE

