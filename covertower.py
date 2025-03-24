import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from gtda.utils.validation import validate_params
from gtda.utils.intervals import Interval
from extendedcover import ExtendedCubicalCover, ExtendedLatticeCover
from copy import deepcopy
from utils._covertower import _remove_duplicate_covers

class CubicalCoverTower(BaseEstimator, TransformerMixin):
    '''
    Sequence of Cubical Covers used to produce a Multiscale Mapper object. 
    Initialize a `extendedcover.ExtendedCubicalCover`cover with parameters
    `n_intervals = n_intervals` and `overlap_frac = start_overlap_frac`.
    Store this in :attr:`initial_cover`.
    Then we expand each cover set in the initial cover by increasing 
    the overlap fraction `tower_size` - 1 times, creating a filtration
    of covers across the `overlap_frac` parameter.
    The list of covers in the filtration is stored in :attr:`cover_sets`.
    Each cover in the filtration is a `extendedcover.ExtendedCubicalCover`
    object.

    Parameters
    ----------
    tower_size : int, optional, default: ``10``.
        The number of cover objects in the tower. Each subsequent cover
        contains intervals which are larger in size than the previous cover.
        
    n_intervals : int, optional, default: ``10``.
        The number of intervals for each feature dimension
        calculated in :meth:`fit`.

    start_overlap_frac : float, optional, default: ``0.1``.
        The fractional overlap between consecutive intervals in the covers of
        each feature domain calculated in :meth:`fit` for the initial cover.

    end_overlap_frac : float, optional, default: ``0.9``.
        The upper bound on the fractional overlap between consecutive intervals
        in the covers of each feature domain calculated in :meth:`fit` for the 
        final cover in the Tower.

    kind : str, optional, default: ``uniform``.
        The kind of CubicalCover to use. Options are ``uniform`` or ``balanced``.

    Attributes
    ----------

    initial_cover : `extendedcover.ExtendedCubicalCover`
        initiated with paramters `overlap_frac = start_overlap_frac`
        and `n_intervals = n_intervals`.

    cover_sets : list 
        The list of covers in the tower of covers. 
        Each cover is a `extendedcover.ExtendedCubicalCover` object.

    cover_set_volumes : ndarray of shape (tower_size, 1).
        The list of the volume for every set in each cover of the tower of covers. The i-th value is
        is the volume of each cover set in the ith cover in the cover tower.

    cover_overlap_fracs: ndarray of shape (tower_size, 1).
        The sequence of overlap_fracs used to create each cover in the tower of covers.
    
    '''

    _hyperparameters = {
        'tower_size': {'type': int, 'in': Interval(1, np.inf, closed='left')},
        'n_intervals': {'type': int, 'in': Interval(1, np.inf, closed='left')},
        'start_overlap_frac': {'type': float, 'in': Interval(0, 1, closed='neither')},
        'end_overlap_frac': {'type': float, 'in': Interval(0, 1, closed='neither')},
        'kind': {'type': str, 'in': ['uniform', 'balanced']},
    }

    def __init__(self, 
                 tower_size = 10, 
                 n_intervals = 10,
                 start_overlap_frac = 0.1,
                 end_overlap_frac = 0.9,
                 kind = 'uniform'):
        
        self.tower_size = tower_size
        self.n_intervals = n_intervals
        self.start_overlap_frac = start_overlap_frac
        self.end_overlap_frac = end_overlap_frac
        self.kind = kind
        self.initial_cover = ExtendedCubicalCover(kind = self.kind,
                                                  tower_index = 0,
                                                  n_intervals = self.n_intervals,
                                                  overlap_frac = self.start_overlap_frac)

        assert self.start_overlap_frac < self.end_overlap_frac, \
        f"Must have ending overlap {self.end_overlap_frac} greater than starting overlap {self.start_overlap_frac}."
        
    def _fit_single_cover(self, X, init_cover, gain, cover_set_index, tower_index):
        ''' Fit a single cover in the tower of covers '''
        new_cover = deepcopy(init_cover)
        new_cover.cover_set_index = cover_set_index
        new_cover.overlap_frac = gain
        new_cover.tower_index = tower_index
        new_cover.fit_extended(X)
        return new_cover, new_cover.cover_set_volume_

    def _fit(self, X):
        self.initial_cover.fit(X)._transform(X)
        cover_set_index = self.initial_cover.cover_set_index
        gains = np.linspace(self.start_overlap_frac, self.end_overlap_frac, num = self.tower_size, endpoint = True)
        self.cover_overlap_fracs = gains
        self.cover_sets = []
        self.cover_set_volumes = []
        for k in range(self.tower_size):
            cover, set_vol = self._fit_single_cover(X,
                                       self.initial_cover,  
                                       gains[k],
                                       cover_set_index,
                                       tower_index = k
                                      )
            self.cover_sets.append(cover)
            self.cover_set_volumes.append(set_vol)
        return self

    def fit(self, X, y=None):
        """Compute a tower of covers by computing all cover interval limits 
        for every cover set in the tower of covers according to `X`.
        Store each fitted cover in the cover tower in :attr:`cover_sets`.
        We additionally compute the overlap_frac and volume of each
        cover and store them in :attr:`cover_overlap_fracs`.
        The tower of covers is a filtration and we store the volume of each 
        cover set in :attr:`cover_set_volumes` as a list.

        This method is here to implement the usual scikit-learn API, and 
        hence works in pipelines.

        Parameters
        -----------
        X : ndarry of shape (n_samples,) or (n_samples,1)
            Input data.

        y : None
            There is no need for a target in a transform, yet the pipeline
            API requires this parameter.

        Returns
        ---------
        self : object
        
        """
        validate_params(self.get_params(), self._hyperparameters)
        return self._fit(X)

    def _transform(self, X):
        transformed_covers = [
            self.cover_sets[k].transform(X) for k in range(self.tower_size)
        ]
        return transformed_covers

    def transform(self, X, y = None):
        """Compute a cover tower of `X` according to the the cover tower of 
        the space computed in :meth:`fit`, and return it as a list of 
        transformed covers where each transformed cover is a boolean array
        (n_samples, num_coversets).

        Parameters
        -----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        y : None
            There is no need for a target in a transformer, yet the pipeline 
            API requires this parameter.

        Returns
        ---------
        Xt : list of length tower_size.
            Each object in the list is a fit and transformed 
            `extendedcover.ExtendedCubicalCover` object,
            ie. an ndarray of shape (n_samples, num_coversets), encoding 
            a cover of `X` as a boolean array.

        """
        Xt = check_array(X, ensure_2d = False)
        for k in range(self.tower_size):
            check_is_fitted(self.cover_sets[k], '_coverers')
        Xt_masks = self._transform(Xt)
        Xt_masks = _remove_duplicate_covers(Xt_masks)
        return Xt_masks

    def _fit_transform(self, X):
        Xt = self._fit(X)._transform(X)
        return Xt

    def fit_transform(self, X):
        """ Fit the data, then transform it.

        Parameters
        -----------
        X : ndarray of shape(n_samples, n_features)

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        ----------
        Xt : list of length tower_size.
            Each object in the list is a fit and transformed 
            `extendedcover.ExtendedCubicalCover`,
            ie. an ndarray of shape (n_samples, num_coversets), encoding 
            a cover of `X` as a boolean array.
        """
        Xt = check_array(X, ensure_2d = False)
        validate_params(self.get_params(), self._hyperparameters)
        Xt = self._fit_transform(Xt)
        Xt = _remove_duplicate_covers(Xt)
        return Xt

class LatticeCoverTower(BaseEstimator, TransformerMixin):
    ''' Sequence of Lattice Covers used to produce a Multiscale Mapper object. 
    Initialize a `extendedcover.ExtendedLatticeCover` with parameters
    `n_intervals = n_intervals` and `overlap_frac = start_overlap_frac`.
    Store this in :attr:`initial_cover`.
    Then we expand each cover set in the initial cover by increasing 
    the overlap fraction `tower_size` - 1 times, creating a filtration
    of covers across the `overlap_frac` parameter.
    The list of covers in the filtration is stored in :attr:`cover_sets`.

    Parameters
    ----------
    tower_size : int, optional, default: ``10``.
        The number of cover objects in the tower. Each subsequent cover
        contains cover sets which are larger in size than the previous cover.
        
    n_intervals : int, optional, default: ``10``.
        The number of intervals for each feature dimension
        calculated in :meth:`fit`.

    start_overlap_frac : float, optional, default: ``0.1``.
        The fractional overlap between consecutive intervals in the covers of
        each feature domain calculated in :meth:`fit` for the initial cover.

    end_overlap_frac : float, optional, default: ``0.5``.
        The upper bound on the fractional overlap between consecutive intervals
        in the covers of each feature domain calculated in :meth:`fit` for the 
        final cover in the Tower.

    kind : str, default ``triangular``. 
        The type of lattice used to construct the cover. 
        Options: ``cubical`` or ``triangular``.
        ``cubical`` produces hypercube cover sets centered at points
        on the integer lattice I^n.
        ``triangular`` produced spherical cover sets centered at points
        on the dual root lattice A_n^*.

    special : bool, default ``False``.
        Special generating matrices for the ``triangular`` lattice construction.
        Only can be set to ``True`` if ``kind`` is ``triangular`` and
        the dimension of the data set is 2 or 3.

    Attributes
    ----------
    initial_cover : `extendedcover.ExtendedLatticeCover` object initiated with paramters `overlap_frac = start_overlap_frac`,
    `n_intervals = n_intervals`, `kind = kind`, and `special = special`.

    cover_sets : list 
        The list of covers in the tower of covers. Each cover is a `extendedcover.ExtendedLatticeCover` object.

    cover_set_volumes : ndarray of shape (tower_size, 1)
        The list of the volume for every set in each cover of the tower of covers. The i-th value is
        is the volume of each cover set in the ith cover in the cover tower.

    cover_overlap_fracs: ndarray of shape (tower_size, 1).
        The sequence of overlap_fracs used to create each cover in the tower of covers.
    
    '''

    _hyperparameters = {
        'tower_size': {'type': int, 'in': Interval(1, np.inf, closed='left')},
        'n_intervals': {'type': int, 'in': Interval(1, np.inf, closed='left')},
        'start_overlap_frac': {'type': float, 'in': Interval(0, 1, closed='neither')},
        'end_overlap_frac': {'type': float, 'in': Interval(0, 1, closed='neither')},
        'kind': {'type': str, 'in': ['triangular', 'cubical']},
        'special': {'type': bool},
    }

    def __init__(self, tower_size = 10, n_intervals = 10,
                 start_overlap_frac = 0.1, end_overlap_frac = 0.5,
                 kind = 'triangular', special = False):
        self.tower_size = tower_size
        self.n_intervals = n_intervals
        self.start_overlap_frac = start_overlap_frac
        self.end_overlap_frac = end_overlap_frac
        self.kind = kind
        self.special = special
        self.initial_cover = ExtendedLatticeCover(special=self.special,
                                                  kind = self.kind,
                                                  tower_index = 0,
                                                  n_intervals = self.n_intervals,
                                                  overlap_frac = self.start_overlap_frac)
        assert self.start_overlap_frac < self.end_overlap_frac, \
        f"Must have ending overlap {self.end_overlap_frac} greater than starting overlap {self.start_overlap_frac}."

    def _fit_single_cover(self, X, init_cover, gain, cover_set_index, tower_index, centers):
        is_cubical = self.kind == 'cubical'
        is_triangular = self.kind == 'triangular'
        new_cover = deepcopy(init_cover)
        new_cover.cover_set_index = cover_set_index
        new_cover.overlap_frac = gain
        new_cover.tower_index = tower_index
        new_cover.cover_centers = centers
        new_cover.fit_extended(X)
        return new_cover, new_cover.cover_set_volume_

    def _fit_all(self, X):
        is_triangular = self.kind == 'triangular'
        is_cubical = self.kind == 'cubical'
        self.initial_cover.fit(X)._transform(X)
        if is_triangular:
            cover_centers = self.initial_cover.ball_centers_
        if is_cubical:
            cover_centers = self.initial_cover.interval_centers_
        cover_set_index = self.initial_cover.cover_set_index
        gains = np.linspace(self.start_overlap_frac, self.end_overlap_frac, num = self.tower_size, endpoint = True)
        self.cover_overlap_fracs = gains
        self.cover_sets = []
        cover_set_volumes = []
        for k in range(self.tower_size):
            cover, set_vol = self._fit_single_cover(X,
                                       self.initial_cover,  
                                       gains[k],
                                       cover_set_index,
                                       tower_index = k,
                                       centers = cover_centers,
                                      )
            self.cover_sets.append(cover)
            cover_set_volumes.append(set_vol)
        self.cover_set_volumes = np.array(cover_set_volumes)
        return self

    def fit(self, X, y=None):
        """Compute a tower of covers by computing all attributes 
        for every cover set in the tower of covers according to `X`.
        Store each fitted cover in the cover tower in :attr:`cover_sets`.
        We additionally store the overlap_frac for
        each cover in :attr:`cover_overlap_fracs`.
        The tower of covers is a filtration and we store the volume of each 
        cover set in :attr:`cover_set_volumes` as a list.

        This method is here to implement the usual scikit-learn API, and 
        hence works in pipelines.

        Parameters
        -----------
        X : ndarry of shape (n_samples,) or (n_samples,1)
            Input data.

        y : None
            There is no need for a target in a transform, yet the pipeline
            API requires this parameter.

        Returns
        ---------
        self : object
        
        """
        validate_params(self.get_params(), self._hyperparameters)
        return self._fit_all(X)

    def _transform(self, X):
        transformed_covers = [
            self.cover_sets[k].transform(X) for k in range(self.tower_size)
        ]
        return transformed_covers

    def transform(self, X, y = None):
        """Compute a cover tower of `X` according to the the cover tower of 
        the space computed in :meth:`fit`, and return it as a list of 
        transformed covers where each transformed cover is a boolean array
        (n_samples, num_coversets).

        Parameters
        -----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        y : None
            There is no need for a target in a transformer, yet the pipeline 
            API requires this parameter.

        Returns
        ---------
        Xt : list of length tower_size.
            Each object in the list is a fit and transformed 
            `ExtendedCubicalCover`,
            ie. an ndarray of shape (n_samples, num_coversets), encoding 
            a cover of `X` as a boolean array.

        """
        Xt = check_array(X, ensure_2d = False)
        for k in range(self.tower_size):
            check_is_fitted(self.cover_sets[k], 'left_limits_')
        Xt_masks = self._transform(Xt)
        Xt_masks = _remove_duplicate_covers(Xt_masks)
        return Xt_masks

    def _fit_transform(self, X):
        Xt = self._fit_all(X)._transform(X)
        return Xt

    def fit_transform(self, X):
        """ Fit the data, then transform it.

        Parameters
        -----------
        X : ndarray of shape(n_samples, n_features)

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        ----------
        Xt : list of length tower_size.
            Each object in the list is a fit and transformed 
            `ExtendedLatticeCover`,
            ie. an ndarray of shape (n_samples, num_coversets), encoding 
            a cover of `X` as a boolean array.
        """
        Xt = check_array(X, ensure_2d = False)
        validate_params(self.get_params(), self._hyperparameters)
        Xt = self._fit_transform(Xt)
        Xt = _remove_duplicate_covers(Xt)
        return Xt