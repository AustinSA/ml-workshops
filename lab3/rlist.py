import random

def random_list(n, secure=True):
	random_floats = []
	if secure:
		crypto = random.System.Random()
		random_float = crypto.random
	else:
		random_float=random.random
	for i in range(n):
		random_floats.append(random_float())
	return random_floats
print(random_list(10, secure=False))
