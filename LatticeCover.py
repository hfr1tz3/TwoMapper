import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from gtda.utils.validation import validate_params
from gtda.utils.intervals import Interval
import warnings
from gtda.mapper.utils._cover import _remove_empty_and_duplicate_intervals

def cartesian_product(arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la])
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

class LatticeCover(BaseEstimator, TransformerMixin):
    ''' Cover data with balls centered at points from a chosen root lattice.
    In :meth: `fit`, given a training array `X` representing a collection of
    real numbers, a cover of the plane by balls or hypercubes centered at a
    root lattice is constructed based on the distribution of values in `X`.
    In :meth:  `transform`, the cover is applied to a new array `X'` to yield
    a cover of `X'`.

    Two kinds of cover are currently available: "triangular" and "cubical". The cubical cover is constructed from hypercubes whose centers lie on a scaled integer root lattice :math: `\mathbb{Z}^n`. This will yield hypercubes of equal length in all dimensions. The triangular cover is generated from the dual root lattice :math: `A_n^*`, also known as the *triangular lattice* in dimension 2 and the *body-centered cubic lattice* in dimension 3. The centers of all spheres from both types of covers are constructed from scaled generating matrices :math: `M`, and the number of points in each dimension of the lattice is determined by :math: `\max_{1\leq i \leq N+1}\frac{1}{K} |M_i - m_i|`, where :math:`K` is the parameter `n_intervals`.

    Parameters
    ----------
    n_intervals : int, optional, default: 10. 
        The number of intervals to construct the lattice.
    overlap_frac : float, optional, default 0.3. 
        The overlap fraction between cover sets.
    kind : ``'triangular'`` | ``'cubical'``, default ``'triangular'``.
        Determines the underlying root lattice for center points, with ``'triangular'`` constructing :math: `A_n^*` and ``'cubical'`` constructing :math: `\mathbb{Z}^n`.
    special : bool, default False. If `True`, `kind = 'triangular'`, and the filtered data is 2 or 3 dimensional, then the cover centers will be constructed with the special generating matrices
      $$M = 
    \begin{bmatrix}
    1 & 0 \\
    \frac{-1}{2} & \frac{\sqrt{3}}{2} \\
    \end{bmatrix},\qquad
    M =
    \begin{bmatrix}
    2 & 0 & 0 \\
    0 & 2 & 0 \\
    1 & 1 & 1 \\
    \end{bmatrix}
    $$
 
    Attributes
    -----------
    dim : int
        Number of features in the data set computed in :meth: `fit`.
    left_limits_ : ndarray of shape (dim,)
        Minimum limits of the bounding box in all dimensions computed in :meth: `fit`. 
    right_limits_ : ndarray of shape (dim,)
    ball_centers_ : ndarray of shape at most ( large product see notes , dim + 1)
        For ``'triangular'`` cover only.
        Centers of every ball in the cover. The number of balls are dependent on `n_intervals`, `left_limits_`, and `right_limits_`.
    ball_radius_ : float
        The radius for each ball in the cover. For ``'triangular'`` cover only.
    left_interval_limits_ : ndarray of shape (c, dim)
        For ``'cubical'`` cover only.
        Left limits for each hypercube in the cover. The number of hypercubes 
        are dependent on `n_intervals`, `left_limits_` and `right_limits_`.
    right_interval_limits_ : ndarray of shape (c, dim)
        For ``'cubical'`` cover only. The number of hypercubes 
        are dependent on `n_intervals`, `left_limits_` and `right_limits_`.
        
    '''
    # Parameters
    _hyperparameters = {
        'kind': {'type': str, 'in': ['triangular', 'cubical']},
        'n_intervals': {'type': int, 'in': Interval(1, np.inf, closed='left')},
        'overlap_frac': {'type': float, 'in': Interval(0, 1, closed = 'neither')},
        'special': {'type': bool}
    }
    def __init__(self, n_intervals = 10, overlap_frac = 0.3, kind = 'triangular', special = False):
        self.n_intervals = n_intervals
        self.overlap_frac = overlap_frac
        self.special = special
        self.kind = kind
        
    def _fit_triangular(self, X):
        self.dim = self._check_dim(X)
        if self.special is False:
            X = self._hyperplane_embed(X)
        self.left_limits_, self.right_limits_ = self._find_bounding_box(X,
                                                                   self.dim,
                                                                   self.n_intervals, 
                                                                   self.special,
                                                                   self.kind
                                                                  )
        self.ball_centers_, self.ball_radius_ = self._triangular_lattice_cover_limits(self, 
                                                                           self.left_limits_,
                                                                           self.right_limits_,
                                                                           self.dim,
                                                                           self.n_intervals,
                                                                           self.overlap_frac,
                                                                           self.special
                                                                          )
        return self

    def _fit_cubical(self, X):
        self.dim = self._check_dim(X)
        self.left_limits_, self.right_limits_ = self._find_bounding_box(X,
                                                                      self.dim,
                                                                      self.n_intervals,
                                                                      self.special,
                                                                      self.kind
                                                                     )
        self.interval_centers_, self.left_interval_limits_, self.right_interval_limits_ = self._cubical_lattice_cover_limits(self,
                                                                                                                             self.left_limits_,
                                                                                                                             self.right_limits_,
                                                                                                                             self.dim,
                                                                                                                             self.n_intervals,
                                                                                                                             self.overlap_frac,
                                                                                                                             self.special
                                                                                                                             )
        return self

    def fit(self, X, y=None):
        """Compute all cover interval limits according to `X`
        and store them in :attr:`left_limits_`, :attr:`right_limits_`.
        For the ``'triangular'`` cover we additionally compute the 
        ball centers and radii and store them in
        :attr:`ball_centers_`, and :attr:`ball_radius_`.
        For the ``'cubical'`` cover we additionally compute each hypercube
        interval limits and store them in :attr:`left_interval_limits_` and
        :attr:`right_interval_limits_`. 
        Then return the estimator.

        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray of shape (n_samples,) or (n_samples,1)
            Input data.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        ----------
        self : object
        
        """
        X = check_array(X)
        validate_params(self.get_params(), self._hyperparameters)
        if self.overlap_frac <= 1e-8:
            warmings.warn("`overlap_frac` is close to zero, "
                          "which might cause numerical issues and errors.",
                          RuntimeWarning)
        is_triangular = self.kind == 'triangular'
        fitter = self._fit_triangular if is_triangular else self._fit_cubical
        return fitter(X)

    def _triangular_transform_data(self, X):
        data_bools = np.full((self.ball_centers_.shape[0],), False)
        for i in range(X.shape[0]):
            cover_check = np.linalg.norm(self.ball_centers_ - X[i], axis = 1) < self.ball_radius_
            if np.any(cover_check):
                data_bools = np.vstack([data_bools, cover_check])
            else:
                print('cover check failed at point')
                print('point', i, X[i])
                print('radius', self.ball_radius_)
                print(np.linalg.norm(self.ball_centers_ - X[i], axis=1))
                print(cover_check)
        return data_bools[1:]
            
    def _triangular_transform_centers(self, X):
        cover_bools = np.full((X.shape[0],), False)
        for i in range(self.ball_centers_.shape[0]):
            cover_check = np.linalg.norm(X - self.ball_centers_[i], axis = 1) < self.ball_radius_
            if np.any(cover_check):
                cover_bools = np.vstack([cover_bools, cover_check])
        data_bools = cover_bools[1:].T
        return data_bools
        
    def _triangular_transform(self, X):
        if self.ball_centers_.shape[0] <= X.shape[0]:
            data_bools = self._triangular_transform_data(X)
        if X.shape[0] < self.ball_centers_.shape[0]:
            data_bools = self._triangular_transform_centers(X)
        return data_bools

    def _cubical_transform_data(self, X):
        data_bools = np.full((self.left_interval_limits_.shape[0],), False)
        for i in range(X.shape[0]):
            cover_check = np.all(np.logical_and(X[i] > self.left_interval_limits_, 
                                         X[i] < self.right_interval_limits_), axis = 1)
            if np.any(cover_check):
                data_bools = np.vstack([data_bools, cover_check])
            else:
                print('cover check failed at point')
                print('point', i, X[i])
                print('left limits', self.left_interval_limits_)
                print('right limits', self.right_interval_limits_)
                print(np.logical_and(X[i] > self.left_interval_limits_, 
                                     X[i] < self.right_interval_limits_))
                print(cover_check)
        return data_bools[1:]

    def _cubical_transform(self, X):
        return self._cubical_transform_data(X)

    def transform(self, X, y=None):
        """Compute a cover of `X` according to the cover of the space
        computed in :meth:`fit`, and return it as a (n_samples, num_coversets)
        boolean array. Each column indicates the location of the entries in
        `X` corresponding to a common cover set.

        Parameters
        -----------
        X : ndarray of shape (n_samples, dim)
            Input data.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API require this parameter.

        Returns
        --------
        Xt : ndarray of shape (n_samples, num_coversets)
            Encoding the cover of `X` as a boolean array. In general,
            `num_coversets` is less than or equal to `n_intervals`**dim
            as empty or duplicated cover sets are removed.

        """
        check_is_fitted(self)
        Xt = check_array(X)
        is_triangular = self.kind == 'triangular'
        Xt = self._triangular_transform(Xt) if is_triangular else self._cubical_transform(Xt)
        Xt = _remove_empty_and_duplicate_intervals(Xt)
        return Xt

    def _fit_transform(self, X):
        is_triangular = self.kind == 'triangular'
        if is_triangular:
            if self.special:
                Xt = self._fit_triangular(X)._triangular_transform(X)
            else:
                Xt = self._fit_triangular(X)._triangular_transform(self._hyperplane_embed(X))
        else:
            Xt = self._fit_cubical(X)._cubical_transform(X)
        return Xt

    def fit_transform(self, X, y=None, **fit_params):
        """Fit the data, then transform it.

        Parameters
        ----------
        X : ndarray of shape (n_samples, dim)
            Input data.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        --------
        Xt : ndarray of shape (n_samples, num_coversets)
            Encoding the cover of `X` as a boolean array. In general,
            `num_coversets` is less than or equal to `n_intervals`**dim
            as empty or duplicated cover sets are removed.
            
        """
        Xt = check_array(X)
        validate_params(self.get_params(), self._hyperparameters)
        Xt = self._fit_transform(Xt)
        Xt = _remove_empty_and_duplicate_intervals(Xt)
        return Xt

    def _check_dim(self, X):
        if X.shape[1] > 5:
            warnings.warn("Using an incredibly high dimensional (dim {X.shape[1]}) can be dangerous; Kernel destroying, in fact. Proceed with Caution")
        return X.shape[1]
    
    def _hyperplane_embed(self, X):
        '''Embeds our data X\sub R^{dim} \righthookarrow R^{dim+1}.
        Used when ``kind = 'triangular'`` and ``special = False``.
        
        '''
        embed = -np.sum(X,axis=1).T
        return np.c_[X, embed]

    def _find_bounding_box(self, X, dim, n_intervals, special, kind):
        """Finds bounds for each coordinate over data set X.
            Outputs a ndarray of shape (dim+1,2) if ``special = True`` 
            and ``kind = 'triangular'`` or a ndarray of shape (dim,2)
            if ``kind = 'cubical`` or (``kind = 'triangular'`` and 
            ``special = True``).
        """
        is_cubical = kind == 'cubical'
        if special and (dim not in {2,3}):
            raise ValueError(f'We only have special lattice representations in dimensions 2 and 3.')
        coord_array = np.zeros((dim+1,2)) # Embed image into R^{dim+1}
        for i in range(dim+1):
            if (special or is_cubical) and i == dim:
                coord_array[i] = [0,0]
            else:
                coord_array[i,0] = np.min(X[:,i]) # Minimum value in i-th coord
                coord_array[i,1] = np.max(X[:,i]) # Maximum value in i-th coord
        only_one_pt = all( _ == coord_array.ravel()[0] for _ in coord_array.ravel())
        if only_one_pt and n_intervals > 1:
            raise ValueError(
                f"Only one unique filter value found, cannot fit"
                f"{n_intervals} > 1 intervals.")
        if special or is_cubical: # We have special representations for A* in dimensions 2 and 3.
            return coord_array[:dim, 0], coord_array[:dim, 1]
        else:
            return coord_array[:,0], coord_array[:,1]

    @staticmethod
    def _get_generator_matrix(self, dim, special):
        """Create generating matrix M for ``kind = 'triangular'`` covers.
        M is has two special representations chosen when ``special = True``
        and `X` has dimension 2 or 3. In general, M will be (dim, dim + 1)
        matrix.
        """
        if dim == 2 and special:
            basis_vectors = np.array([1,0,-1/2,np.sqrt(3)/2]).reshape((2,2))
        if dim == 3 and special:
            basis_vectors = np.array([2,0,0,0,2,0,1,1,1]).reshape((3,3))
        if dim not in {2,3} or not special:
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
    def _triangular_lattice_cover_limits(self, left_limits, right_limits, dim, n_intervals, overlap_frac, special):
        """Contruct the scaled :math:`A_n^*` lattice and output sphere centers
        (lattice points) and common radius for each ball.
        """
        generating_matrix = self._get_generator_matrix(self, dim, special)
        ldim = generating_matrix.shape[1]
        assert left_limits.shape[0] == right_limits.shape[0] == ldim, f'{left_limits.shape[0]} != {right_limits.shape[0]} != {ldim}'
        bound_vector = np.abs(right_limits - left_limits)
        scale = np.max([np.average(bound_vector/n_intervals), 1])
        # Create bounds for lattice points
        scaled_min_bound, scaled_max_bound = np.zeros(dim), np.zeros(dim)
        if special and dim == 2:
            cover_radius = 1 / np.sqrt(3)
            scaled_min_bound[0] = np.floor((left_limits[0] + left_limits[1]/np.sqrt(3))/scale)
            scaled_min_bound[1] = np.floor(2 * left_limits[1] / (scale *np.sqrt(3)))
            scaled_max_bound[0] = np.ceil((right_limits[0] + (right_limits[1]/np.sqrt(3)))/ scale)
            scaled_max_bound[1] = np.ceil(2 * right_limits[1] / (scale * np.sqrt(3)))
        if special and dim == 3:
            cover_radius = np.sqrt(5) / 2
            scaled_min_bound[2] = left_limits[2] / scale
            scaled_max_bound[2] = right_limits[2] / scale
            for i in range(2):
                scaled_min_bound[i] = (left_limits[i] - right_limits[2]) / (2 * scale)
                scaled_max_bound[i] = (right_limits[i] - left_limits[2]) / (2 * scale)
        if not special:
            cover_radius = np.sqrt(dim * (dim + 2)/(12* (dim + 1)))
            scaled_min_bound[:dim-1] = np.asarray([np.floor((left_limits[dim] - right_limits[j+1])/scale) for j in range(dim-1)])
            scaled_min_bound[dim-1] = np.floor((ldim) * left_limits[dim] / scale)
            scaled_max_bound[:dim-1] = np.asarray([np.ceil((right_limits[dim] - left_limits[j+1])/scale) for j in range(dim-1)])
            scaled_max_bound[dim-1] = np.ceil((ldim) * right_limits[dim] / scale)
        xi_coord_arrays = [np.arange(start = scaled_min_bound[k], stop = scaled_max_bound[k]+1, step = 1) for k in range(dim)]
        xi_vectors = cartesian_product(xi_coord_arrays) # all possible integer vectors to generate lattice points
        lattice_points = scale * xi_vectors @ generating_matrix
        # remove unnecessary lattice points:
        pt_list = np.asarray(lattice_points)
        scaled_cover_radius = scale * cover_radius * (1 + overlap_frac) # radius for balls at each lattice point
        mask = np.all(np.logical_and(pt_list > left_limits - scaled_cover_radius, pt_list < right_limits + scaled_cover_radius), axis = 1)
        
        return pt_list[mask], scaled_cover_radius

    @staticmethod
    def _cubical_lattice_cover_limits(self, left_limits, right_limits, dim, n_intervals, overlap_frac, special):
        """Construct scaled integer lattice :math:`\mathbb{Z}^n` and return
        uniform hypercube bounds for the cover sets.
        """
        
        # we don't need generating matrix since it is the identity
        # generating_matrix = np.ma.identity(dim)
        cover_radius = np.sqrt(dim)/2
        assert left_limits.shape[0] == right_limits.shape[0] == dim, f'dim = {dim}, left {left_limits.shape}, right {right_limits.shape}'
        bound_vector = np.abs(right_limits - left_limits)
        scale = np.max([np.average(bound_vector/n_intervals), 1])
        # create bounds for lattice points
        scaled_min_bound = np.floor(left_limits/scale)
        scaled_max_bound = np.ceil(right_limits/scale)
        xi_coord_arrays = [np.arange(start = scaled_min_bound[k], stop = scaled_max_bound[k]+1, step = 1) for k in range(dim)]
        xi_vectors = cartesian_product(xi_coord_arrays) # all possible integer vectors to generate lattice points
        lattice_points = scale * xi_vectors
        pt_list = np.asarray(lattice_points)
        mask = np.all(np.logical_and(pt_list > left_limits - 0.5 * scale / (1 - overlap_frac), pt_list < right_limits + 0.5 * scale / (1 - overlap_frac)), axis = 1)
        left_interval_limits = pt_list[mask] - 0.5 * scale / (1 - overlap_frac)
        right_interval_limits = pt_list[mask] + 0.5 * scale / (1 - overlap_frac)
        assert left_interval_limits.shape == right_interval_limits.shape
        return pt_list[mask], left_interval_limits, right_interval_limits