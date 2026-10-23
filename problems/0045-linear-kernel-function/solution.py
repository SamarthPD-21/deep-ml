import numpy as np

def kernel_function(x1, x2):
	# Your code here
	if len(x1) != len(x2):
		return -1
	Ans=0.0
	for i in range(len(x1)):
		Ans = Ans+ x1[i]*x2[i]

	return Ans
	pass
