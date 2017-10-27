import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance 
from operator import itemgetter

color_dict = {1 : 'g',
				2 : 'y',
				3 : 'b',
				4 : 'r'}

### given a file location
### returns datatable of text file 
def read_file(file_loc):
	df = pd.read_table(file_loc, delim_whitespace=True, names=('d1', 'd2', 'd3', 'c'))
	return df

def initialize_matrix(n):
	matrix = np.full((n, n), float('inf'))
	return matrix 

def initialize_dist_data(df, n):
	dist_data = []
	for i in range(n):
		pointI = (df['d1'][i], df['d2'][i], df['d3'][i])
		distances = []
		for j in range(n):
			pointJ = (df['d1'][j], df['d2'][j], df['d3'][j])
			dist = distance.euclidean(pointI, pointJ)
			distances.append([i, j, dist])
		distances.sort(key=itemgetter(2))
		distances = distances[0:11]
		dist_data.append(distances)
	return dist_data

def update_matrix_with_dist(init_matrix, dist_data):
	for i in dist_data:
		for j in i:
			X = int(j[0])
			Y = int(j[1])
			Z = float(j[2])
			init_matrix[X][Y] = Z
	return init_matrix

def floyd_warshall(matrix, n):
	for k in range(0, n):
		for i in range(0, n):
			for j in range(0, n):
				if(matrix[i][j] > matrix[i][k] + matrix[k][j]):
					matrix[i][j] = matrix[i][k] + matrix[k][j]
	return matrix

### citation: http://www.nervouscomputer.com/hfs/cmdscale-in-python/
def cmds(matrix, n):
	n = len(matrix)
	H = np.eye(n) - np.ones((n, n))/n
	B = -H.dot(matrix**2).dot(H)/2
	eig_vals, eig_vecs = np.linalg.eigh(B)
	eval_1, eval_2, eig_pairs = sort_eig_pairs(eig_vals, eig_vecs)
	eig_vec_mat = construct_eig_vec_matrix(eig_pairs)
	eig_vec_mat = np.array(eig_vec_mat)

	eig_vec_mat_T = eig_vec_mat.T[::1]

	eig_val_mat = np.zeros((n, 2))
	eig_val_mat[0][0] = eval_1
	eig_val_mat[1][1] = eval_2
	result = eig_vec_mat_T.dot(eig_val_mat)
	return result 

### sort the eigenvalue, eigenvector pairs by greatest to smallest eigenvalue
def sort_eig_pairs(eig_vals, eig_vecs):
	eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
	eig_pairs.sort(key=lambda x: x[0], reverse=True)
	eval_1 = eig_pairs[0][0]
	eval_2 = eig_pairs[1][0]
	return eval_1, eval_2, eig_pairs 

def construct_eig_vec_matrix(sorted_eig_pairs):
	eig_vec_mat = []
	for eig_val, eig_vec in sorted_eig_pairs:
		eig_vec_mat.append(eig_vec.tolist())
	return eig_vec_mat

def plot_data(result, last_col):
	for i in range(0, len(result)): 
		j = result[i]
		X = j[0]
		Y = j[1]
		plt.scatter(X, Y, c=color_dict[int(last_col[i])])
	plt.show()

### main function is where this python script starts 
### vary the main function based on what you want to output
def main():
	### setting up the data 
	file_loc = './3Ddata.txt'
	df = read_file(file_loc)
	last_col = df.iloc[:,-1]
	n = len(df)

	init_matrix = initialize_matrix(n)
	dist_data = initialize_dist_data(df, n)
	matrix = update_matrix_with_dist(init_matrix, dist_data)
	matrix = floyd_warshall(matrix, n)
	result = cmds(matrix, n)	
	plot_data(result, last_col)

if __name__ == "__main__":
	main()