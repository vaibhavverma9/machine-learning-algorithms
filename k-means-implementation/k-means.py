from matplotlib import pyplot as plt
import numpy as np 
import random
import math 

########################################################################
########################################################################
### The following functions are the implementations of k-means and k-means++


### k_means algorithm
def k_means(data, number_of_clusters):
	### array of distortion keeps track of distortion over time
	distortion_by_iteration = []


	### initializing an updated set of cluster points 
	cluster_centers = initialize_random_cluster_centers(data, number_of_clusters)
	distortion_by_iteration.append(distortion_value(data, cluster_centers))

	### updating data and then updating centers 
	data = assign_data_to_cluster_points(data, cluster_centers)
	updated_centers = update_cluster_centers(data, cluster_centers)
	distortion_by_iteration.append(distortion_value(data, cluster_centers))

	### updating centers continually until a set of cluster sets does not update
	while updated_centers != cluster_centers:
		cluster_centers = updated_centers
		data = assign_data_to_cluster_points(data, cluster_centers)
		updated_centers = update_cluster_centers(data, cluster_centers)
		distortion_by_iteration.append(distortion_value(data, cluster_centers))
	print distortion_value(data, cluster_centers)
	return (data, distortion_by_iteration)

def k_means_plusplus(data, number_of_clusters):
	distortion_by_iteration = []


	### initializing an updated set of cluster points 
	cluster_centers = initialize_plusplus_cluster_centers(data, number_of_clusters)
	distortion_by_iteration.append(distortion_value(data, cluster_centers))

	### updating data and then updating centers 
	data = assign_data_to_cluster_points(data, cluster_centers)
	updated_centers = update_cluster_centers(data, cluster_centers)
	distortion_by_iteration.append(distortion_value(data, cluster_centers))

	### updating centers continually until a set of cluster sets does not update
	while updated_centers != cluster_centers:
		cluster_centers = updated_centers
		data = assign_data_to_cluster_points(data, cluster_centers)
		updated_centers = update_cluster_centers(data, cluster_centers)
		distortion_by_iteration.append(distortion_value(data, cluster_centers))
	return (data, distortion_by_iteration)

color_dict = {0 : 'bo',
				1 : 'go',
				2 : 'ro'}

### parse text file data given a file name
def parse_data(file_name):
	data = []
	f = open(file_name, 'r')
	lines = f.readlines()

	for line in lines:
		line = line.strip()
		if line:
			arr = line.split(' ')
			temp_arr = []
			for i in arr:
				try: 
					i = float(i)
					temp_arr.append(i)
				except: 
					continue
			### the following keeps track of which cluster the data point is assigned to
			### initialized to 0
			temp_arr.append(0) 
			data.append(temp_arr)
	return data

########################################################################
########################################################################
### The following are various plot functions

### plots data using matplotlib library 
def plot_data(data):
	for arr in data:
		X = arr[0]
		Y = arr[1]
		Z = arr[2]		
		plt.plot(X, Y, color_dict[Z])
	plt.show()

### plot distortion values by iteration for each run 
def plot_distortion(distortions):
	for arr in distortions:
		plt.plot(range(0, len(arr)), arr)
	plt.show()


########################################################################
########################################################################
### The following are initialization functions for cluster centers 


### function initializes cluster centers randomly using data
### xmin, ymin and xmax, ymax serve as bounds for random selection
def initialize_random_cluster_centers(data, number_of_clusters):
	xmin = min_x(data)
	ymin = min_y(data)
	xmax = max_x(data)
	ymax = max_y(data)

	initialized_cluster_points = []

	while number_of_clusters > 0:
		random_x = random.uniform(xmin, xmax)
		random_y = random.uniform(ymin, ymax)
		initialized_cluster_points.append([random_x, random_y])
		number_of_clusters -= 1

	return initialized_cluster_points 

### function initializes cluster centers 
### using k-means initialization algorithm for cluster centers 
def initialize_plusplus_cluster_centers(data, number_of_clusters):
	cluster_centers = []
	cluster_centers.append(random.choice(data)) 

	### determine probabilities 
	for i in xrange(1, number_of_clusters):
		point = identify_plusplus_cluster_center(data, cluster_centers)
		cluster_centers.append(point)
	return cluster_centers

########################################################################
########################################################################
### The following two functions support the k-means++ algorithm


### identifies a cluster center for k-means++ initialization
def identify_plusplus_cluster_center(data, cluster_centers):
	probabilities = determine_probabilities(data, cluster_centers)
	if (len(probabilities) != len(data)):
		return "error: data and probabilities do not match up in k-means++"
	
	# generate a random number between 0 and 1
	rand = random.uniform(0, 1)

	# walk the list substracting the probability of each item from number
	# pick item that, after substraction, took the number down to 0 or below.
	for i in range(0, len(probabilities)):
		if(rand < 0):
			return data[i]
		else: 
			rand -= probabilities[i]

def determine_probabilities(data, cluster_centers):
	probabilities = []
	sum_cost = sum_min_cost(data, cluster_centers)
	for data_point in data:
		prob = float(min_cost(data_point, cluster_centers)) / float(sum_cost)
		probabilities.append(prob)
	return probabilities 

########################################################################
########################################################################
### The following two functions determine the min cost 
### between a data point and cluster centers 


### given a data point and a set of cluster centers
### function returns minimum cost between the data point and a cluster center 
def min_cost(data_point, cluster_centers):
	min_val = float("inf")
	for i in cluster_centers:
		if(distance_squared(data_point, i) < min_val):
			min_val = distance_squared(data_point, i)
	return min_val 

### given a data set and a set of cluster centers
### function returns sum of min cost for each data point 
def sum_min_cost(data, cluster_centers):
	total_cost = 0 
	for data_point in data: 
		total_cost += min_cost(data_point, cluster_centers)
	return total_cost 

########################################################################
########################################################################
### The following functions contribute to the correct assignment 
### of data points to cluster points


### given a data set and cluster centers
### function assigns each data point a cluster based on distance
def assign_data_to_cluster_points(data, cluster_centers):
	updated_data = []
	for i in data:
		X = i[0]
		Y = i[1]
		point = [X, Y]		
		cluster_index = closest_cluster_center(point, cluster_centers)
		i[2] = cluster_index
		updated_data.append(i)
	return updated_data

### given a data point and cluster_centers 
### function retuns which the index of the closest cluster centers
def closest_cluster_center(point, cluster_centers):
	min_dist = float("inf")
	cluster_index = float("inf")
	for i in range(0, len(cluster_centers)):
		dist = distance_squared(point, cluster_centers[i])
		if(dist < min_dist):
			cluster_index = i
			min_dist = dist
	return cluster_index 

def update_cluster_centers(data, cluster_centers):
	updated_clusters = []
	for i in range(0, len(cluster_centers)):
		xtotal = 0.0
		ytotal = 0.0
		count = 0.0
		for j in data:
			if(j[2] == i):
				xtotal += j[0]
				ytotal += j[1]
				count += 1
		if(count != 0):
			X = float(xtotal) / count
			Y = float(ytotal) / count
			updated_clusters.append([X, Y])
	return updated_clusters


########################################################################
########################################################################
### The following function calculates the distortion value


def distortion_value(data, cluster_centers):
	cost = 0
	for i in data:
		point = [i[0], i[1]]
		index = i[2]
		dist = distance_squared(point, cluster_centers[index])
		cost += dist
	return cost 
	### to be continued 

########################################################################
########################################################################
### The following two functions run multiple iterations on the data
### k-means vs. k-means++ 


def run_iterations(number, data, number_of_clusters):
	distortions = []
	while number > 0:
		(data, distortion_by_iteration) = k_means(data, number_of_clusters)
		number -= 1
		distortions.append(distortion_by_iteration)
	# plot_distortion(distortions)

def run_iterations_plusplus(number, data, number_of_clusters):
	distortions = []
	while number > 0:
		(data, distortion_by_iteration) = k_means_plusplus(data, number_of_clusters)
		number -= 1
		distortions.append(distortion_by_iteration)
	plot_distortion(distortions)

########################################################################
########################################################################
### The following functions are min, max functions for x and y respectively
### along with distance functions 

def distance(pointA, pointB):
	dist = math.hypot(pointB[0] - pointA[0], pointB[1] - pointA[1])
	return dist

### determines the distance between two points using math library
def distance_squared(pointA, pointB):
	dist = math.hypot(pointB[0] - pointA[0], pointB[1] - pointA[1])
	dist = dist ** 2
	return dist

def min_x(data):
	min_val = float("inf")
	for arr in data:
		X = arr[0]
		if X < min_val:
			min_val = X
	return min_val

def max_x(data):
	max_val = -float("inf")
	for arr in data:
		X = arr[0]
		if X > max_val:
			max_val = X
	return max_val

def min_y(data):
	min_val = float("inf")
	for arr in data:
		Y = arr[1]
		if Y < min_val:
			min_val = Y
	return min_val

def max_y(data):
	max_val = -float("inf")
	for arr in data:
		Y = arr[0]
		if Y > max_val:
			max_val = Y
	return max_val

### main function is where this python script starts 
### vary the main function based on what you want to output
def main():
	number_of_clusters = 3
	file_name = 'toydata.txt'
	data = parse_data(file_name)
	###	run_iterations(20, data, number_of_clusters)
	(data, distortion_by_iteration) = k_means_plusplus(data, number_of_clusters)	
	plot_data(data)

if __name__ == "__main__":
	main()