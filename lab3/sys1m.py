from random import SystemRandom

crypto = SystemRandom()
x=[crypto.random() for _ in range (1000000)]
