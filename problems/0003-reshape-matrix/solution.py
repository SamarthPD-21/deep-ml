import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	#Write your code here and return a python list after reshaping by using numpy's tolist() method
	n1 = len(a)*len(a[0])
	n2 = new_shape[0]*new_shape[1]

	if n2!=n1 :
		return []
	Ans = np.reshape(a, new_shape)
	return Ans