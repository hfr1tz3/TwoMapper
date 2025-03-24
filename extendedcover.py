import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from gtda.utils.validation import validate_params
from gtda.utils.intervals import Interval
from gtda.mapper.cover import CubicalCover
from latticecover import LatticeCover
from itertools import product

class ExtendedLatticeCover(LatticeCover):
    '''LatticeCover constructed for LatticeCoverTower. 
    This cover is a subclass of the `latticecover.LatticeCover` object, where
    we additionally book keep filtration value and an index list for the cover sets
    for the cover using :attr:`tower_index` and :attr:`cover_set_index`.

    All methods are identical to the methods for `latticecover.LatticeCover` object,
    except that to fit a `ExtendedLatticeCover` object you should use the :meth:`fit_extended`
    instead of :meth:`fit`. See `latticecover.LatticeCover` for more information on how 
    a lattice cover is constructed.

    Parameters
    -----------
    n_intervals : int, optional, default: ``10``
        The number of interval in the cover computed in :meth:`fit_extended`.

    overlap_frac : float, optional, default: ``0.1``
        The proportional intersection length between two consecutive cover sets
        in the cover.

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

    tower_index : int, optional, default: ``0``
        The index of the cover used when the extended cover is within a 
        `covertower.LatticeCoverTower` object.

    cover_set_index : ndarray, optional, default: ``None``
        The index labelling for each cover set in the ExtendedLatticeCover.
        This is used to book keep cover sets in between covers within a
        `covertower.LatticeCoverTower` object.

    cover_centers: ndarray, optional, default: ``None``
        The list of center points to be used in the ExtendedLatticeCover.
        This is used to fix the cover sets in between covers within a 
        `covertower.LatticeCoverTower` object to the same as the initial
        cover in the tower of covers.

    Attributes
    -----------
    This is a subclass of `latticecover.LatticeCover` and hence has its attributes:
    
    dim : int
        Number of features in the data set computed in :meth: `fit`.

    left_limits_ : ndarray of shape (dim,)
        Minimum limits of the bounding box in all dimensions computed in :meth: `fit`. 

    right_limits_ : ndarray of shape (dim,)

    ball_centers_ : ndarray of shape at most ( large product see notes , dim + 1)
        Centers of every ball in the cover. 
        The number of balls are dependent on `n_intervals`, `left_limits_`, and `right_limits_`.
        Attribute only when ``kind = 'triangular'``.

    ball_radius_ : float
        The radius for each ball in the cover. Attribute only when ``kind = 'triangular'``.

    left_interval_limits_ : ndarray of shape (num_intervals, dim)
        The left limits of intervals in each dimension of the cover.
        For each dimension the interval limits are bounded between `left_limits_` and `right_limits_`.
        Attribute only when ``kind = 'cubical'``.

    right_interval_limits_ : ndarray of shape (num_intervals, dim)
        The right limits of intervals in each dimension of the cover.
        For each dimension the interval limits are bounded between `left_limits_` and `right_limits_`.
        Attribute only when ``kind = 'cubical'``.

    interval_centers_ : ndarray of shape (num_intervals, dim)
        The center points for each hypercube in the cover.
        Attribute only when `kind` is ``cubical``.

    cover_set_volume_ : float
        The volume of each cover set in the cover.
    '''

    _hyperparameters = {
    'n_intervals': {'type': int, 'in': Interval(1, np.inf, closed='left')},
    'overlap_frac': {'type': float, 'in': Interval(0, 1, closed='neither')},
    'tower_index': {'type': int, 'in': Interval(0, np.inf, closed='left')},
    'cover_set_index': {'type': np.ndarray},
    'cover_centers' : {'type': np.ndarray},
    'kind': {'type': str, 'in': ['triangular', 'cubical']},
    'special': {'type': bool},
    }

    def __init__(self, n_intervals=10, overlap_frac=0.1, tower_index=0, cover_set_index=None,
                 cover_centers = None, kind='triangular', special=False, **kwargs):
        super(ExtendedLatticeCover, self).__init__(n_intervals=n_intervals,
                                                   overlap_frac=overlap_frac, 
                                                   kind=kind, 
                                                   special=special, 
                                                   **kwargs)
        self.tower_index = tower_index
        if cover_set_index is None:
            self.cover_set_index = np.full((n_intervals, 1), True, dtype = bool)
        else:
            self.cover_set_index = cover_set_index
        if cover_centers is None:
            self.cover_centers = np.array([-1])
        else:
            self.cover_centers = cover_centers

    def _fit_extended(self, X):
        is_cubical = self.kind == 'cubical'
        is_triangular = self.kind == 'triangular'
        self.fit(X)
        # Centers must be fixed between covers in the tower for Lattice covers.
        # This is because the lattice covers define the viable point set wrt 
        # overlap_frac and as we filter over overlap_frac this tends to increase
        # the number of viable points which is BAD.
        # instead we fix the cover center set based on the initial cover in the tower.
        if is_cubical:
            if self.tower_index == 0:
                self.cover_centers = self.interval_centers_
            if self.tower_index != 0:
                self.interval_centers_ = self.cover_centers
                bound_vector = np.abs(self.right_limits_ - self.left_limits_)
                scale = np.max([np.average(bound_vector/self.n_intervals), 1])
                self.right_interval_limits_ = self.cover_centers + 0.5 * scale / (1 - self.overlap_frac)
                self.left_interval_limits_ = self.cover_centers - 0.5 * scale / (1 - self.overlap_frac)
            interval_lens = [self.right_interval_limits_[0][i] - self.left_interval_limits_[0][i] for i in range(self.dim)]
            set_vol = np.multiply.reduce(interval_lens)
        if is_triangular:
            if self.tower_index == 0:
                self.cover_centers = self.ball_centers_
            if self.tower_index != 0:
                self.ball_centers_ = self.cover_centers
            set_vol = 2 * np.pi * self.ball_radius_
        self.cover_set_volume_ = set_vol
        return self
            
    def fit_extended(self, X, y=None):
        """Compute all cover interval limits or ball centers,
        and ball radii according to `X` as in `latticecover.LatticeCover`.

        Parameters
        -----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        y : None
            There is no need for a target with a transformer.
            Required for pipeline API.

        Returns
        ---------
        self : object
        """
        # fit usual LatticeCover object
        # without the removing of duplicates from
        # the latticecover.LatticeCover class
        X = check_array(X, ensure_2d=False)
        validate_params(self.get_params(), self._hyperparameters)
        self._fit_extended(X)
        return self

    def _prune_cover(self, masks):
        # Remove empty cover sets only if the tower index is 0
        # otherwise they must be preserved for the filtration
        # in `covertower.LatticeCoverTower` object
        if self.tower_index == 0:
            cover_set_index = np.where(np.any(masks, axis = 0))[0]
            self.cover_set_index = cover_set_index
        pruned_masks = masks[:, self.cover_set_index]
        return pruned_masks

    def _transform(self, X):
        is_cubical = self.kind == 'cubical'
        is_triangular = self.kind == 'triangular'
        if is_cubical:
            Xt = self._cubical_transform(X)
        if is_triangular:
            if self.special is False:
                X = self._hyperplane_embed(X)
            Xt = self._triangular_transform(X)
        Xt = self._prune_cover(Xt)
        return Xt

    def transform(self, X, y=None):
        """Compute cover of `X` according to cover computed with 
        :meth:`fit_extended`. Constructs a 2-dimensional 
        boolean array with each column indicating the location of 
        entries in `X` with respect to the cover.

        Parameters
        -----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        y : None
            Not needed, just required for scikit pipeline API.
        """
        check_is_fitted(self)
        Xt = check_array(X , ensure_2d=False)
        Xt = self._transform(Xt)
        return Xt
        
    def fit_transform(self, X, y=None, **fit_params):
        """Fit a cover to the data, then transform it.

        Parameters
        -----------
        X : ndarray of shape (n_samples, n_features)
            Input data

        y : None
            Not needed, just required for scikit pipeline API.
        """
        Xt = check_array(X)
        Xt = self.fit_extended(Xt)._transform(Xt)
        return Xt
        

class ExtendedCubicalCover(CubicalCover):
    '''CubicalCover constructed for CubicalCoverTower. 
    This cover is a subclass of the `gtda.mapper.cover.CubicalCover` object, where
    we additionally book keep filtration value and an index list for the cover sets
    for the cover using :attr:`tower_index` and :attr:`cover_set_index`.
    We additionally add the attribute :attr:`cover_set_volume_` to identify the 
    change in resolution when computing a `multiscale2mapper.Multiscale2Mapper` object.

    All methods are identical to the methods for `gtda.mapper.cover.CubicalCover` object,
    except that to fit a `ExtendedCubicalCover` object you should use the :meth:`fit_extended`
    instead of :meth:`fit`.

    Parameters
    -----------
    n_intervals : int, optional, default: ``10``
        The number of interval in the cover computed in :meth:`fit_extended`.

    overlap_frac : float, optional, default: ``0.1``
        The proportional intersection length between two consecutive intervals 
        in the cover.

    tower_index : int, optional, default: ``0``
        The index of the cover used when the extended cover is within a 
        `covertower.CubicalCoverTower` object.

    cover_set_index : ndarray, optional, default: ``None``
        The index labelling for each cover set in the ExtendedCubicalCover.
        This is used to book keep cover sets inbetween covers within a
        `covertower.CubicalCoverTower` object.

    Attributes
    -----------
    This is a subclass of `gtda.mapper.cover.CubicalCover` and hence has its attributes:
    
    left_limits_ : ndarray of shape (n_intervals,)
        Left limits of the cover intervals computed in :meth:`fit`.
        See `gtda.cover.CubicalCover` Notes.

    right_limits_ : ndarray of shape (n_intervals,)
        Right limits of the cover intervals computed in :meth:`fit`. 
        See the `gtda.cover.CubicalCover` Notes.

    cover_set_volume_ : float
        The volume of each cover set in the cover.
    '''

    _hyperparameters = {
        'n_intervals': {'type': int, 'in': Interval(1, np.inf, closed='left')},
        'overlap_frac': {'type': float, 'in': Interval(0, 1, closed='neither')},
        'tower_index': {'type': int, 'in': Interval(0, np.inf, closed='left')},
        'cover_set_index': {'type': np.ndarray}
    }

    def __init__(self, n_intervals=10, overlap_frac=0.1, tower_index=0, cover_set_index = None, **kwargs):
        super(ExtendedCubicalCover, self).__init__(n_intervals=n_intervals, overlap_frac=overlap_frac, **kwargs)
        self.tower_index = tower_index
        self.n_intervals = n_intervals
        self.overlap_frac = overlap_frac
        if cover_set_index is None:
            self.cover_set_index = np.full((n_intervals, 1), True, dtype = bool)
        else:
            self.cover_set_index = cover_set_index

    def fit_extended(self, X, y=None):
        """Compute all cover interval limits according to `X`
        as in `gtda.mapper.cover.CubicalCover.

        Parameters
        -----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        y : None
            There is no need for a target with a transformer.
            Required for pipeline API.

        Returns
        ---------
        self : object
        """
        # fit usual CubicalCover object
        # without the removing of duplicates from
        # the gtda.mapper.cover.CubicalCover class
        X = check_array(X, ensure_2d=False)
        validate_params(self.get_params(), self._hyperparameters)
        self.fit(X)
        interval_lens = [self._coverers[k].right_limits_[1] - self._coverers[k].left_limits_[1] for k in range(len(self._coverers))]
        self.cover_set_volume_ = np.multiply.reduce(interval_lens)
        return self

    def _prune_cover(self, masks):
        # Remove empty cover sets only if the tower index is 0
        # otherwise they must be preserved for the filtration
        # in `covertower.CubicalCoverTower` object
        if self.tower_index == 0:
            cover_set_index = np.where(np.any(masks, axis = 0))[0]
            self.cover_set_index = cover_set_index                            
        pruned_masks = masks[:, self.cover_set_index]
        return pruned_masks

    def _transform(self, X):
        covers = [coverer._transform(X[:, [i]])
                  for i, coverer in enumerate(self._coverers)]
        Xt = self._index_constant_combine_one_dim_covers(covers)
        Xt = self._prune_cover(Xt)
        return Xt
        
    def transform(self, X, y=None):
        """Compute cover of `X` according to cover computed with 
        :meth:`fit_extended`. Constructs a 2-dimensional 
        boolean array with each column indicating the location of 
        entries in `X` with respect to the cover.

        Parameters
        -----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        y : None
            Not needed, just required for scikit pipeline API.
        """
        check_is_fitted(self, '_coverers')
        Xt = check_array(X , ensure_2d=False)
        Xt = self._transform(Xt)
        return Xt
        
    def fit_transform(self, X, y=None, **fit_params):
        """Fit a cover to the data, then transform it.

        Parameters
        -----------
        X : ndarray of shape (n_samples, n_features)
            Input data

        y : None
            Not needed, just required for scikit pipeline API.
        """
        Xt = self.fit_extended(X)._transform(X)
        return Xt

    @staticmethod
    def _index_constant_combine_one_dim_covers(covers):
        # Similar to the function in gtda, however we remove
        # the function which cleans the cover.
        intervals = (
            [cover[:, i] for i in range(cover.shape[1])] for cover in covers
        )

        Xt = np.array([np.logical_and.reduce(t) for t in product(*intervals)]).T

        return Xt