"""
Unit and regression test for the ballistico package.
"""

# Import package, test suite, and other packages as needed
from ballistico.finitedifference import FiniteDifference
import numpy as np
from ballistico.phonons import Phonons
from ballistico.conductivity import Conductivity
import pytest


@pytest.yield_fixture(scope="session")
def phonons():
    print ("Preparing phonons object.")
    finite_difference = FiniteDifference.from_folder(folder='ballistico/tests/si-crystal',
                                                     supercell=[3, 3, 3],
                                                     format='eskm')
    phonons = Phonons(finite_difference=finite_difference,
                      kpts=[5, 5, 5],
                      is_classic=False,
                      temperature=300,
                      is_tf_backend=True,
                      folder='temp',
                      storage='memory')
    return phonons


def test_sc_conductivity(phonons):
    cond = np.abs(np.mean(Conductivity(phonons=phonons, method='sc', max_n_iterations=71).conductivity
                          .sum(axis=0).diagonal()))
    np.testing.assert_approx_equal(cond, 255, significant=3)


def test_qhgk_conductivity(phonons):
    cond = Conductivity(phonons=phonons, method='qhgk').conductivity.sum(axis=0)
    cond = np.abs(np.mean(cond.diagonal()))
    np.testing.assert_approx_equal(cond, 230, significant=3)


def test_rta_conductivity(phonons):
    cond = np.abs(np.mean(Conductivity(phonons=phonons, method='rta').conductivity.sum(axis=0).diagonal()))
    np.testing.assert_approx_equal(cond, 226, significant=3)


def test_inverse_conductivity(phonons):
    cond = np.abs(np.mean(Conductivity(phonons=phonons, method='inverse').conductivity.sum(axis=0).diagonal()))
    np.testing.assert_approx_equal(cond, 256, significant=3)