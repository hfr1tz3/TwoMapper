import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from gtda.utils.validation import validate_params
from gtda.utils.intervals import Interval
import warnings
from gtda.mapper.utils._cover import _check_has_one_column, _remove_empty_and_duplicate_intervals

'Add to _cover.py'
def _check_has_two_columns(X):
    if X.shape[1] != 2:
        raise ValueError(f"X must have two columns. Currently X has {X.shape[1]} columns.")

class CircularCover(BaseEstimator, TransformerMixin):
    # Parameters
    _hyperparameters = {
        'kind': {'type': str, 'in': ['uniform', 'balanced']},
        'n_intervals': {'type': int, 'in': Interval(1, np.inf, closed='left')},
        'overlap_frac': {'type': float, 'in': Interval(0, 1, closed='neither')}
        }
    '''
    Attributes
    ----------
    coord1_limits_ : ndarray of shape (2,)
        Limits of the bounding box on the coord1 (x) axis computed in :meth: `fit`.
    
    coord2_limits_ : ndarray of shape (2,)
        Limits of the bounding box on the coord2 (y) axis computed in :meth: `fit`.
    
    ball_centers_ : ndarray of shape #Not exactly correct
    (n_intervals*(2*n_intervals/3)*(coord2_limits_[1]-coord2_limits_[0])/(coord1_limits_[1]-coord1_limits_[0]),)
        Centers of every ball in the cover. The number of balls in the cover are
        dependent on `overlap_frac`, `n_intervals`, `coord1_limits_`, and `coord2_limits_`
    
    ball_radius_ : float
        The radius for each ball in the circular cover.
        
    '''
    
    def __init__(self, kind = 'uniform', n_intervals = 10, overlap_frac = 0.1):
        self.kind = kind
        self.n_intervals = n_intervals
        self.overlap_frac = overlap_frac
    
    def _fit_uniform(self, X):
        self.coord1_limits_, self.coord2_limits_ = self._find_bounding_box(
            X, self.n_intervals, is_uniform = True)
        self.ball_centers_, self.ball_radius_ = self._circular_cover_limits(
            self.coord1_limits_[0], self.coord1_limits_[1], self.coord2_limits_[0],
            self.coord2_limits_[1], self.n_intervals, self.overlap_frac)
        return self
    
    def fit(self, X, y=None):
        
        X = check_array(X)
        validate_params(self.get_params(), self._hyperparameters)
        if self.overlap_frac <= 1e-8:
            warnings.warn("`overlap_frac` is close to zero, "
                          "which might cause numerical issues and errors.",
                          RuntimeWarning)
        if X.ndim == 2:
            _check_has_two_columns(X)
            
        is_uniform = self.kind == 'uniform'
        # Can add `balanced` version later
        if is_uniform:
            fitter = self._fit_uniform 
        return fitter(X)
    
    # def _transform(self, X):
    #     Xt = []
    #     for data in X:
    #         dist_bools = [np.linalg.norm(data-center) < self.ball_radius_ for center in self.ball_centers_]
    #         Xt.append(dist_bools)
    #     Xt = np.asarray(Xt)
    #     return Xt
    
    def _transform(self, X):
        data_vec = np.repeat([X], self.ball_centers_.shape[0], axis = 1).reshape(
            (X.shape[0], self.ball_centers_.shape[0], 2)
        )
        ball_vec = np.repeat([self.ball_centers_], X.shape[0], axis = 0)
        data_bools = np.linalg.norm(data_vec - ball_vec, axis = 2) < self.ball_radius_
        return data_bools
    
    def transform(self, X, y=None):
        check_is_fitted(self)
        Xt = check_array(X)
        
        if Xt.ndim == 2:
            _check_has_two_columns(Xt)
        
        Xt = self._transform(Xt)
        Xt = _remove_empty_and_duplicate_intervals(Xt)
        return Xt
    
    def _fit_transform(self, X):
        if self.kind == 'uniform':
            Xt = self._fit_uniform(X)._transform(X)
            return Xt
    
    def fit_transform(self, X, y=None, **fit_params):
        Xt = check_array(X)
        validate_params(self.get_params(), self._hyperparameters)
        
        if Xt.ndim == 2:
            _check_has_two_columns(Xt)
        Xt = self._fit_transform(Xt)
        Xt = _remove_empty_and_duplicate_intervals(Xt)
        return Xt
    
    def _find_bounding_box(self, X, n_intervals, is_uniform=True):
        if is_uniform:
            coord1_min_val, coord1_max_val = np.min(X[:,0]), np.max(X[:,0])
            coord2_min_val, coord2_max_val = np.min(X[:,1]), np.max(X[:,1])
            coord_list = [coord1_min_val, coord1_max_val, coord2_min_val, coord2_max_val]
            only_one_pt = all( _ == coord_list[0] for _ in coord_list)
        else:
            raise ValueError("Circular cover can only be uniformly computed at this time")
        if only_one_pt and n_intervals > 1:
            raise ValueError(
                f"Only one unique filter value found, cannot fit "
                f"{n_intervals} > 1 intervals.")
        
        return np.asarray(coord_list[:2]), np.asarray(coord_list[2:4])
        
    
    @staticmethod
    def _circular_cover_limits(x_min, x_max, y_min, y_max, n_intervals, overlap_frac):
        ''' We build a triangular lattice which will be the centers of each circle in the cover
            This lattice allows us to have distinct 3-fold intersections of of open sets in our
            cover.
        '''
        # Compute the radius, dx, and dy
        x_len = (x_max - x_min)/n_intervals
        y_len = float((np.sqrt(3)/2)*x_len)
        radius = (0.5 + overlap_frac)*x_len
        
        # Find ball centers on the even rows of the lattice
        even_rows = np.arange(start= x_min - 0.5*x_len, stop = x_max + 0.5*x_len, step = x_len)
        even_cols = np.arange(start = y_min, stop = y_max + y_len, step = 2*y_len)
        xeven, yeven = np.meshgrid(even_rows, even_cols)
        even_center_list = np.vstack(list(zip(xeven.ravel(), yeven.ravel())))
        
        # Find ball centers on the odd rows of the lattice
        odd_rows = np.arange(start = x_min, stop = x_max + x_len, step = x_len)
        odd_cols = np.arange(start = y_min + y_len, stop = y_max + y_len, step = 2*y_len)
        xodd, yodd = np.meshgrid(odd_rows, odd_cols)
        odd_center_list = np.vstack(list(zip(xodd.ravel(),yodd.ravel())))
        
        # Combine and sort the centers list via the 2-norm
        center_list = np.concatenate((even_center_list,odd_center_list), axis = 0)
        sorted_index = np.argsort(np.linalg.norm(center_list, axis = 1))
        center_list = center_list[sorted_index]
        
        return center_list, radius