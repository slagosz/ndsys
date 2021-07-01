ndsys
=====
|RTD|

*ndsys* is a scikit-learn-compatible package for nonlinear dynamical systems modelling

The package allows to experiment with SISO (single input, single output) dynamical systems, *i.e.* systems possessing memory.

At the moment it contains implementations of:

*  Volterra (polynomial-like) features
*  an entropic variant of the Dual Averaging algorithm

The original purpose of above implementations were to illustrate our theoretical results concerning the estimation error of Dual Averaging methods in the non-i.i.d. setting (specifically, when the estimated system has a notion of memory). However, the implementation of the Dual Averaging algorithm is not limited to dynamical systems.

.. |RTD| image:: https://readthedocs.org/projects/ndsys/badge/?version=latest
    :target: https://ndsys.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status