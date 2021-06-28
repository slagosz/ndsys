import numpy as np
import pytest

from ndsys.features import VolterraFeatures, prepare_data


def test_prepare_data():
    x = np.vstack([1, 2, 3])
    y = np.vstack([10, 11, 12])

    x_out, y_out = prepare_data(x, y, (1, 1), None)
    assert (x_out == np.vstack([1, 2, 3])).all()
    assert (y_out == np.vstack([10, 11, 12])).all()

    x_out, y_out = prepare_data(x, y, (2, 1), None)
    assert (x_out == np.vstack([1, 2, 3])).all()
    assert (y_out == np.vstack([11, 12])).all()

    x_out, y_out = prepare_data(x, y, (3, 1), None)
    assert (x_out == np.vstack([1, 2, 3])).all()
    assert (y_out == np.vstack([12])).all()

    x_out, y_out = prepare_data(x, y, (1, 1), 'zeros')
    assert (x_out == np.vstack([1, 2, 3])).all()
    assert (y_out == np.vstack([10, 11, 12])).all()

    x_out, y_out = prepare_data(x, y, (2, 1), 'zeros')
    assert (x_out == np.vstack([0, 1, 2, 3])).all()
    assert (y_out == np.vstack([10, 11, 12])).all()

    x_out, y_out = prepare_data(x, y, (3, 1), 'zeros')
    assert (x_out == np.vstack([0, 0, 1, 2, 3])).all()
    assert (y_out == np.vstack([10, 11, 12])).all()

    x_out, y_out = prepare_data(x, y, (2, 1),  np.vstack([-1]))
    assert (x_out == np.vstack([-1, 1, 2, 3])).all()
    assert (y_out == np.vstack([10, 11, 12])).all()

    x_out, y_out = prepare_data(x, y, (3, 1),  np.vstack([-2, -1]))
    assert (x_out == np.vstack([-2, -1, 1, 2, 3])).all()
    assert (y_out == np.vstack([10, 11, 12])).all()


def test_volterra():
    x = np.vstack([1, 2, 3])

    features = lambda kernel: VolterraFeatures(kernel, include_bias=False).fit_transform(x)
    features_bias = lambda kernel: VolterraFeatures(kernel, include_bias=True).fit_transform(x)

    # np.testing.assert_array_equal(features([1]), [[1], [2], [3]])
    # np.testing.assert_array_equal(features_bias([1]), [[1, 1], [1, 2], [1, 3]])
    np.testing.assert_array_equal(features([2]), [[2, 1], [3, 2]])
    np.testing.assert_array_equal(features([3]), [[3, 2, 1]])

    np.testing.assert_array_equal(features([1, 1]), [[1, 1], [2, 4], [3, 9]])
    np.testing.assert_array_equal(features_bias([1, 1]), [[1, 1, 1], [1, 2, 4], [1, 3, 9]])
    np.testing.assert_array_equal(features([1, 2]), [[2, 4, 2, 1], [3, 9, 6, 4]])
    np.testing.assert_array_equal(features([0, 1]), [[1], [4], [9]])
    np.testing.assert_array_equal(features([0, 2]), [[4, 2, 1], [9, 6, 4]])





# def test_volterra_function_basic(self):
#     x = [2, 3]
#     self.assertEqual(volterra_function([0], x), 3)
#     self.assertEqual(volterra_function([1], x), 2)
#     self.assertEqual(volterra_function([0, 0], x), 3 * 3)
#     self.assertEqual(volterra_function([1, 1], x), 2 * 2)
#     self.assertEqual(volterra_function([0, 1], x), 3 * 2)
#     self.assertEqual(volterra_function([1, 0], x), 3 * 2)
#
# def test_volterra_function_time_shift(self):
#     x = [1, 2, 3, 4]
#     self.assertEqual(volterra_function([0], x, 0), 1)
#     self.assertEqual(volterra_function([0], x, 1), 2)
#     self.assertEqual(volterra_function([0], x, 2), 3)
#     self.assertEqual(volterra_function([0], x, 3), 4)
#     self.assertEqual(volterra_function([1], x, 1), 1)
#     self.assertEqual(volterra_function([2], x, 2), 1)
#     self.assertEqual(volterra_function([2], x, 3), 2)
#
# def test_volterra_function_too_short_input(self):
#     x = [1]
#     self.assertRaises(IndexError, volterra_function, [1], x)
