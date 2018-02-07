![RX-M LLC][RX-M LLC]


# Machine Learning


## Lab 7 – Nearest Neighbors

In pattern recognition, the k-nearest neighbors algorithm (k-NN) is a non-parametric method used for classification and
regression. In both cases, the input consists of the k closest training examples in the feature space. The output
depends on whether k-NN is used for classification or regression:

In k-NN classification, the output is a class membership. An object is classified by a majority vote of its neighbors,
with the object being assigned to the class most common among its k nearest neighbors (k is a positive integer,
typically small). If k = 1, then the object is simply assigned to the class of that single nearest neighbor. In k-NN
regression, the output is the property value for the object. This value is the average of the values of its k nearest
neighbors. k-NN is a type of instance-based learning, or lazy learning, where the function is only approximated locally
and all computation is deferred until classification.

Both for classification and regression, a useful technique can be to assign weight to the contributions of the
neighbors, so that the nearer neighbors contribute more to the average than the more distant ones. For example, a common
weighting scheme consists in giving each neighbor a weight of 1/d, where d is the distance to the neighbor.

The neighbors are taken from a set of objects for which the class (for k-NN classification) or the object property value
(for k-NN regression) is known. This can be thought of as the training set for the algorithm, though no explicit
training step is required.

In this lab we'll create a k-NN model to classify cities by programming language preference. This lab is an "under the
hood" lab designed to help you see how algorithms and model work under the covers. We will use our own Python code
to perform all of the operations in the lab except for standard libraries like math and random and matplotlib.


### 1. Data setup

We will use a sample set of data which makes (wild) claims about which programming language a given city prefers. We
will organize the cities by location using latitude and longitude.

Here's the data set:

```python
cities = [(-86.75,33.5666666666667,'Python'),(-88.25,30.6833333333333,'Python'),(-112.016666666667,33.4333333333333,'Java'),(-110.933333333333,32.1166666666667,'Java'),(-92.2333333333333,34.7333333333333,'R'),(-121.95,37.7,'R'),(-118.15,33.8166666666667,'Python'),(-118.233333333333,34.05,'Java'),(-122.316666666667,37.8166666666667,'R'),(-117.6,34.05,'Python'),(-116.533333333333,33.8166666666667,'Python'),(-121.5,38.5166666666667,'R'),(-117.166666666667,32.7333333333333,'R'),(-122.383333333333,37.6166666666667,'R'),(-121.933333333333,37.3666666666667,'R'),(-122.016666666667,36.9833333333333,'Python'),(-104.716666666667,38.8166666666667,'Python'),(-104.866666666667,39.75,'Python'),(-72.65,41.7333333333333,'R'),(-75.6,39.6666666666667,'Python'),(-77.0333333333333,38.85,'Python'),(-80.2666666666667,25.8,'Java'),(-81.3833333333333,28.55,'Java'),(-82.5333333333333,27.9666666666667,'Java'),(-84.4333333333333,33.65,'Python'),(-116.216666666667,43.5666666666667,'Python'),(-87.75,41.7833333333333,'Java'),(-86.2833333333333,39.7333333333333,'Java'),(-93.65,41.5333333333333,'Java'),(-97.4166666666667,37.65,'Java'),(-85.7333333333333,38.1833333333333,'Python'),(-90.25,29.9833333333333,'Java'),(-70.3166666666667,43.65,'R'),(-76.6666666666667,39.1833333333333,'R'),(-71.0333333333333,42.3666666666667,'R'),(-72.5333333333333,42.2,'R'),(-83.0166666666667,42.4166666666667,'Python'),(-84.6,42.7833333333333,'Python'),(-93.2166666666667,44.8833333333333,'Python'),(-90.0833333333333,32.3166666666667,'Java'),(-94.5833333333333,39.1166666666667,'Java'),(-90.3833333333333,38.75,'Python'),(-108.533333333333,45.8,'Python'),(-95.9,41.3,'Python'),(-115.166666666667,36.0833333333333,'Java'),(-71.4333333333333,42.9333333333333,'R'),(-74.1666666666667,40.7,'R'),(-106.616666666667,35.05,'Python'),(-78.7333333333333,42.9333333333333,'R'),(-73.9666666666667,40.7833333333333,'R'),(-80.9333333333333,35.2166666666667,'Python'),(-78.7833333333333,35.8666666666667,'Python'),(-100.75,46.7666666666667,'Java'),(-84.5166666666667,39.15,'Java'),(-81.85,41.4,'Java'),(-82.8833333333333,40,'Java'),(-97.6,35.4,'Python'),(-122.666666666667,45.5333333333333,'Python'),(-75.25,39.8833333333333,'Python'),(-80.2166666666667,40.5,'Python'),(-71.4333333333333,41.7333333333333,'R'),(-81.1166666666667,33.95,'R'),(-96.7333333333333,43.5666666666667,'Python'),(-90,35.05,'R'),(-86.6833333333333,36.1166666666667,'R'),(-97.7,30.3,'Python'),(-96.85,32.85,'Java'),(-95.35,29.9666666666667,'Java'),(-98.4666666666667,29.5333333333333,'Java'),(-111.966666666667,40.7666666666667,'Python'),(-73.15,44.4666666666667,'R'),(-77.3333333333333,37.5,'Python'),(-122.3,47.5333333333333,'Python'),(-89.3333333333333,43.1333333333333,'R'),(-104.816666666667,41.15,'Java')]
```

We will use a list comprehension to reorg the data into a list of tuples, where each tuple contains two elements, a list
of the lat/long and the language. Something like this:

cities = [([longitude, latitude], language) for longitude, latitude, language in cities]

Create a simple Python program that includes the data and the comprehension. Then add a loop to display the data to
verify the new cities data structure is ready for use.

Note that we are using lists not numpy arrays. The down side is numpy arrays are faster and more compact, the upside is
Python lists are more flexible allowing us to append data, store non numbers and collections, etc.


### 2. The k-NN Algorithm

Now let's create the k-NN function. Our function will need to accept the k value (how many neighbors to consider) the
labeled points to pull the neighbors from and the new point to assess.

The body of this function will need to order the labeled points by distance from the current point so that we can
select the closest k points to examine. Then we will need to find the majority label (most common) from the nearest
neighbors. The code might look like this:

```python
def knn_classify(k, labeled_points, new_point):
    """each labeled point should be a pair (point, label)"""

    # order the labeled points from nearest to farthest
    by_distance = sorted(labeled_points,
                         key=lambda (point, _): distance(point, new_point))

    # find the labels for the k closest
    k_nearest_labels = [label for _, label in by_distance[:k]]

    # and let them vote
    return majority_vote(k_nearest_labels)
```

This classification function depends on the distance() function to create sort keys (distances) and the majority_vote()
function for selecting the most popular label (programming language).

The distance function set might look like this:

```python
def distance(v, w):
   return math.sqrt(squared_distance(v, w))

def squared_distance(v, w):
    return sum_of_squares(vector_subtract(v, w))

def sum_of_squares(v):
    """v_1 * v_1 + ... + v_n * v_n"""
    return dot(v, v)

def vector_subtract(v, w):
    """subtracts two vectors componentwise"""
    return [v_i - w_i for v_i, w_i in zip(v,w)]

def dot(v, w):
    """v_1 * w_1 + ... + v_n * w_n"""
    return sum(v_i * w_i for v_i, w_i in zip(v, w))
```

The majority vote function might look like this:

```python
def majority_vote(labels):
    """assumes that labels are ordered from nearest to farthest"""
    vote_counts = Counter(labels)
    winner, winner_count = vote_counts.most_common(1)[0]
    num_winners = len([count
                       for count in vote_counts.values()
                       if count == winner_count])

    if num_winners == 1:
        return winner                     # unique winner, so return it
    else:
        return majority_vote(labels[:-1]) # try again without the farthest
```

Add these functions to your python sample data program from step 1.


### 3. Create a solution

Write a Python program that accepts a lat/long on the command line and uses the data from step 1 and the functions from
step 2 to display the k-NN class (programming language) for the entered lat/long using a k of 5.


### 4. Test values of k

Modify your program to output the k-NN classes for 10 points selected from your exiting data. Display the results for
values of k including: 1, 3, 5, 7

Report the accuracy of each k by comparing the k-NN result to the label for the point. Don't forget to remove the city
you are testing from the neighbors set.


## CHALLANGE STEP

Add a plot of the city locations from step 4. For extra credit use the programming language to label the city's point.

Hint: matplotlib scatter plots work well in this capacity:

```python
plt.scatter(x, y, color=colors[language], marker=markers[language], label=language, zorder=10)
```


<br>

Congratulations you have successfully completed the lab!

<br>

_Copyright (c) 2013-2017 RX-M LLC, Cloud Native Consulting, all rights reserved_

[RX-M LLC]: http://rx-m.io/rxm-cnc.svg "RX-M LLC"
