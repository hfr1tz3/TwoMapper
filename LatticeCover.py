import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from gtda.utils.validation import validate_params
from gtda.utils.intervals import Interval
import warnings
from gtda.mapper.utils._cover import _remove_empty_and_duplicate_intervals

## Taken from:
## https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points
def cartesian_product(arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la])
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

class LatticeCover(BaseEstimator, TransformerMixin):
    # Parameters
    _hyperparameters = {
        'n_intervals': {'type': int, 'in': Interval(1, np.inf, closed='left')},
        'overlap_frac': {'type': float, 'in': Interval(0, 1, closed = 'neither')},
        'special': {'type': bool}
    }
    ''' 
    Attributes
    -----------
    dim : int
        Number of features in the data set computed in :meth: `fit`.
    left_limits_ : ndarray of shape (dim,)
        Minimum limits of the bounding box in all dimensions computed in :meth: `fit`. 
    right_limits_ : ndarray of shape (dim,)
    ball_centers_ : ndarray of shape ( large product see notes , dim + 1)
        Centers of every ball in the cover. The number of balls are dependent on `n_intervals`, `left_limits_`, and `right_limits_`.
    cover_radius_ : float
        The radius for each ball in the cover.
    '''

    def __init__(self, n_intervals = 10, overlap_frac = 0.3, special = False):
        self.n_intervals = n_intervals
        self.overlap_frac = overlap_frac
        self.special = special
        
    def _fit(self, X):
        self.dim = self._check_dim(X)
        if self.special is False:
            X = self._hyperplane_embed(X)
        self.left_limits_, self.right_limits_ = self._find_bounding_box(X,
                                                                   self.dim,
                                                                   self.n_intervals, 
                                                                   self.special
                                                                  )
        self.ball_centers_, self.ball_radius_ = self._lattice_cover_limits(self, 
                                                                           self.left_limits_,
                                                                           self.right_limits_,
                                                                           self.dim,
                                                                           self.n_intervals,
                                                                           self.overlap_frac,
                                                                           self.special
                                                                          )
        return self

    def fit(self, X, y=None):
        X = check_array(X)
        validate_params(self.get_params(), self._hyperparameters)
        if self.overlap_frac <= 1e-8:
            warmings.warn("`overlap_frac` is close to zero, "
                          "which might cause numerical issues and errors.",
                          RuntimeWarning)
        fitter = self._fit
        return fitter(X)

    def _transform_data(self, X):
        data_bools = np.full((self.ball_centers_.shape[0]), False)
        for i in range(X.shape[0]):
            cover_check = np.linalg.norm(self.ball_centers_ - X[i], axis = 1) < self.ball_radius_
            if np.any(cover_check):
                data_bools = np.vstack([data_bools, cover_check])
        return data_bools[1:]
            
    def _transform_centers(self, X):
        cover_bools = np.full((X.shape[0],), False)
        for i in range(self.ball_centers_.shape[0]):
            cover_check = np.linalg.norm(X - self.ball_centers_[i], axis = 1) < self.ball_radius_
            if np.any(cover_check):
                cover_bools = np.vstack([cover_bools, cover_check])
        data_bools = cover_bools[1:].T
        return data_bools
        
    def _transform(self, X):
        if self.ball_centers_.shape[0] < X.shape[0]:
            data_bools = self._transform_data(X)
        if X.shape[0] < self.ball_centers_.shape[0]:
            data_bools = self._transform_centers(X)
        return data_bools

    def transform(self, X, y=None):
        check_is_fitted(self)
        Xt = check_array(X)
        Xt = self._transform(Xt)
        Xt = _remove_empty_and_duplicate_intervals(Xt)
        return Xt

    def _fit_transform(self, X):
        if self.special:
            Xt = self._fit(X)._transform(X)
        if self.special is False:
            Xt = self._fit(X)._transform(self._hyperplane_embed(X))
        return Xt

    def fit_transform(self, X, y=None, **fit_params):
        Xt = check_array(X)
        validate_params(self.get_params(), self._hyperparameters)
        Xt = self._fit_transform(Xt)
        Xt = _remove_empty_and_duplicate_intervals(Xt)
        return Xt

    def _check_dim(self, X):
        if X.shape[1] > 5:
            warnings.warn("Using an incredibly high dimensional (dim {X.shape[1]}) can be dangerous; Kernel destroying, in fact. Proceed with Caution")
        return X.shape[1]
    
    'Embeds our data X\sub R^{dim} \righthookarrow R^{dim+1}'
    def _hyperplane_embed(self, X):
        embed = -np.sum(X,axis=1).T
        return np.c_[X, embed]
    'Finds bounds for each coordinate over data set X'
    'Outputs a (dim+1,2) array'
    def _find_bounding_box(self, X, dim, n_intervals, special=False):
        if special and (dim not in {2,3}):
            raise ValueError(f'We only have special lattice representations in dimensions 2 and 3.')
        coord_array = np.zeros((dim+1,2)) # Embed image into R^{dim+1}
        for i in range(dim+1):
            coord_array[i,0] = np.min(X[:,i]) # Minimum value in i-th coord
            coord_array[i,1] = np.max(X[:,i]) # Maximum value in i-th coord
        only_one_pt = all( _ == coord_array.ravel()[0] for _ in coord_array.ravel())
        if only_one_pt and n_intervals > 1:
            raise ValueError(
                f"Only one unique filter value found, cannot fit"
                f"{n_intervals} > 1 intervals.")
        if special: # We have special representations for A* in dimensions 2 and 3.
            return coord_array[:dim, 0], coord_array[:dim, 1]
        else:
            return coord_array[:,0], coord_array[:,1]

    @staticmethod
    def _get_generator_matrix(dim, special):
        if dim == 2 and special:
            basis_vectors = np.array([1,0,-1/2,np.sqrt(3)/2]).reshape((2,2))
        if dim == 3 and special:
            basis_vectors = np.array([2,0,0,0,2,0,1,1,1]).reshape((3,3))
        else:
            basis_vectors = np.zeros((dim, dim+1))
            basis_vectors[dim-1, 0] = -dim/(dim+1)
            basis_vectors[dim-1, dim] = 1/(dim+1)
            for i in range(dim-1):
                basis_vectors[i,0] = 1
                basis_vectors[i,i+1] = -1
                basis_vectors[dim-1,i+1] = 1/(dim+1)
        generator_matrix = np.asmatrix(basis_vectors)
        return generator_matrix

    @staticmethod
    def _lattice_cover_limits(self, left_limits, right_limits, dim, n_intervals, overlap_frac, special):
        generating_matrix = self._get_generator_matrix(dim, special)
        cover_radius = np.sqrt(dim * (dim + 2)/(12* (dim + 1)))
        ldim = generating_matrix.shape[1]
        assert left_limits.shape[0] == right_limits.shape[0] == ldim
        bound_vector = np.abs(right_limits - left_limits)
        scale = np.max([np.max(bound_vector/n_intervals), 1])
        # Create bounds for lattice points
        scaled_min_bound, scaled_max_bound = np.zeros(dim), np.zeros(dim)
        scaled_min_bound[:dim-1] = np.asarray([np.floor((left_limits[dim] - right_limits[j+1])/scale) for j in range(dim-1)])
        scaled_min_bound[dim-1] = np.floor((ldim) * left_limits[dim] / scale)
        scaled_max_bound[:dim-1] = np.asarray([np.ceil((right_limits[dim] - left_limits[j+1])/scale) for j in range(dim-1)])
        scaled_max_bound[dim-1] = np.ceil((ldim) * right_limits[dim] / scale)
        xi_coord_arrays = [np.arange(start = scaled_min_bound[k], stop = scaled_max_bound[k], step = 1) for k in range(dim)]
        xi_vectors = cartesian_product(xi_coord_arrays) # all possible integer vectors to generate lattice points
        lattice_points = scale * xi_vectors @ generating_matrix
        scaled_cover_radius = scale * cover_radius * (1 + overlap_frac) # radius for balls at each lattice point
        return lattice_points, scaled_cover_radius