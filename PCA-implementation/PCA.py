import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


color_dict = {1 : 'g', 2 : 'y', 3 : 'b', 4 : 'r'}

### given a file location
### returns datatable of text file 
def read_file(file_loc):
	df = pd.read_table('./3Ddata.txt', delim_whitespace=True, names=('d1', 'd2', 'd3', 'c'))
	return df

### given a dataset and number of lines
### returns covariance matrix 
def create_cov_matrix(df, n):
	df = df.drop('c', 1)
	cov_matrix = [[0, 0, 0]] * 3
	mean_vec = np.mean(df, axis=0)
	cov_mat = (df - mean_vec).T.dot((df - mean_vec)) / (df.shape[0])
	return cov_mat

### eigendecomposition of the covariance matrix
### returns eigenvalues and eigenvectors 
def eigendecomposition(cov_matrix):
	eig_vals, eig_vecs = np.linalg.eig(cov_matrix)
	return eig_vals, eig_vecs

### sort the eigenvalue, eigenvector pairs by greatest to smallest eigenvalue
def sort_eig_pairs(eig_vals, eig_vecs):
	eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
	eig_pairs.sort(key=lambda x: x[0], reverse=True)
	return eig_pairs 	

### removes dimension with lowest eigenvalue 
def reduced_matrix(eig_pairs):
	matrix_w = np.hstack((eig_pairs[0][1].reshape(3, 1), eig_pairs[1][1].reshape(3, 1)))
	return matrix_w

### transform onto the new subspace 
def projection(df, matrix_w):
	last_col = df.iloc[:,-1]
	df_dim = df.drop('c', 1)
	df = df_dim.dot(matrix_w)
	df['c'] = last_col 
	plot_data(df)

def plot_data(df):
	for index, row in df.iterrows():
		X = row[0]
		Y = row[1]
		Z = row[2]
		plt.scatter(X, Y, c=color_dict[int(Z)])
	plt.show()

### main function is where this python script starts 
### vary the main function based on what you want to output
def main():
	### setting up the data 
	n = 500
	file_loc = './3Ddata.txt'
	df = read_file(file_loc)
	
	### functions to produce output for #4
	cov_matrix = create_cov_matrix(df, n)
	eig_vals, eig_vecs = eigendecomposition(cov_matrix)
	sorted_eig_pairs = sort_eig_pairs(eig_vals, eig_vecs)
	matrix_w = reduced_matrix(sorted_eig_pairs)
	projection(df, matrix_w)

if __name__ == "__main__":
	main()
