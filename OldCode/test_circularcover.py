from functools import reduce

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis.extra.numpy import arrays, array_shapes
from hypothesis.strategies import floats, integers, booleans, composite
from numpy.testing import assert_almost_equal
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from CircularCover import CircularCover

'This composite function does not work very well for my tests'
# @composite
# def get_filter_values(draw, shape=None):
#     """Generate a 2d array of floats, of a given shape. If the shape is not
#     given, generate a shape of at least (4,2)."""
#     if shape is None:
#         shape = array_shapes(min_dims=1, max_dims=1, min_side=4)
#     x = draw(arrays(dtype=float,
#                     elements=floats(allow_nan=False,
#                                     allow_infinity=False,
#                                     min_value=-1e5,
#                                     max_value=1e5),
#                     shape=shape, unique=True))
#     y = draw(arrays(dtype=float,
#                     elements=floats(allow_nan=False,
#                                     allow_infinity=False,
#                                     min_value=-1e5,
#                                     max_value=1e5),
#                     shape=shape, unique=True))
#     pts = [[x[i],y[i]] for i in range(x.shape[0])]
#     return np.asarray(pts)    


@composite
def get_nb_intervals(draw):
    return draw(integers(min_value=3, max_value=20))


@composite
def get_overlap_fraction(draw):
    return draw(floats(allow_nan=False,
                       allow_infinity=False,
                       min_value=1e-5, exclude_min=True,
                       max_value=1., exclude_max=True))



@given(n_intervals=get_nb_intervals(),
       overlap_frac=get_overlap_fraction())
@settings(deadline=1000, max_examples=100)
def test_fit_transform_against_fit_and_transform(
        n_intervals, overlap_frac
        ):
    """Fitting and transforming should give the same result as fit_transform"""
    pts = 1000*np.random.random((100,2))
    pts = np.unique(pts, axis=0)
    cover = CircularCover(n_intervals=n_intervals,
                                overlap_frac=overlap_frac)
    x_fit_transf = cover.fit_transform(pts)

    cover2 = CircularCover(n_intervals=n_intervals,
                                 overlap_frac=overlap_frac)
    cover2 = cover2.fit(pts)
    x_fit_and_transf = cover2.transform(pts)
    assert_almost_equal(x_fit_transf, x_fit_and_transf)
    
@given(n_intervals=get_nb_intervals())
@settings(deadline=1000, max_examples=100)
def test_filter_values_covered(n_intervals):
    """Test that each value is at least in one interval.
    (that is, the cover is a true cover)."""
    filter_values = 1000*np.random.random((100,2))
    filter_values = np.unique(filter_values, axis=0)
    cover = CircularCover(n_intervals=n_intervals)
    interval_masks = cover.fit_transform(filter_values)
    intervals = []
    for i in range(interval_masks.shape[1]):
        intervals.extend(filter_values[interval_masks[:, i]])
    unique_points = np.unique(intervals, axis=0)
    pt_sort = np.argsort(np.linalg.norm(unique_points, axis=1))
    fil_sort = np.argsort(np.linalg.norm(filter_values, axis=1))
    
    # print('-----------------------------------')
    # print('DATA: ', filter_values)
    # print('CENTERS: ', cover.fit(filter_values).ball_centers_)
    # print('RADIUS: ', cover.fit(filter_values).ball_radius_)
    # print('COVER OUTPUT: ', interval_masks)
    # print('POINTS IN EACH COVER: ', intervals)
    # print('UNIQUE POINTS: ', unique_points)
    # print('--------------------------------')

    assert_almost_equal(unique_points[pt_sort], filter_values[fil_sort])

