import numpy as np

class PCA:
    """Expects input data to be a matrix where is row is an element"""
    def __init__(self,):
        pass
    def preprocess_image(self,image):
        """Expects and image to be (c,w,h)"""
        

    def compute(self,data):
        n,m = data.shape
        mean_centered_data = data - np.tile(np.mean(data,axis=1),(n,1)).T
        U, S, VT = np.linalg.svd(mean_centered_data,full_matrices=False)

    def _compute_covariance_matrix(self,mean_centered_data):
        return np.matmul(np.transpose(mean_centered_data),mean_centered_data)
    
    def _compute_eigen(self,cov_matrix):
        eigen_values = np.linalg.eigvals(cov_matrix)
        return eigen_values
    
    def _compute_principal_components(self,mean_centered_data,eigen_values):
        principal_components = np.matmul(mean_centered_data,eigen_values)
        return principal_components