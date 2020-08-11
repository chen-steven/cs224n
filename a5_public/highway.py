#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
class Highway(nn.Module):
	def __init__(self, embedding_length):
		super(Highway, self).__init__()
		self.embedding_length = embedding_length
		self.proj_layer = nn.Linear(in_features=embedding_length, out_features=embedding_length, bias=True)
		self.gate_layer = nn.Linear(in_features=embedding_length, out_features=embedding_length, bias=True)
	def forward(self, x):
		x_proj = F.relu(self.proj_layer(x))
		x_gate = torch.sigmoid(self.gate_layer(x))

		x_highway = x_proj*x_gate + (1-x_gate)*x
		return x_highway

### END YOUR CODE 
	

