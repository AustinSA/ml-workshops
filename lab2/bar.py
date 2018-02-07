from matplotlib import pyplot as plt

beer = ["Belgian", "British", "American", "German", "Mexican"]
num_votes = [5,11,3,8,10]
xs = [i + 0.1 for i, _ in enumerate(beer)]
print xs

plt.bar(xs, num_votes)
plt.ylabel("# votes")
plt.title("Favorite beer country")
plt.xticks([ i + 0.1 for i, _ in enumerate(beer)], beer)
plt.show()
