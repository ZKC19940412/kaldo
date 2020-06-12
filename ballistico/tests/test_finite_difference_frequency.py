import numpy as np
from ase.build import bulk
from ase.calculators.emt import EMT
from ballistico.finitedifference import FiniteDifference
from ballistico.phonons import Phonons
import pytest
from tempfile import TemporaryDirectory

frequency = np.array([[-1.25649072e-04, -8.34000390e-05, -3.59057257e-05],
       [ 1.97491784e+00,  1.97491785e+00,  4.42161612e+00],
       [ 3.14745764e+00,  3.14745765e+00,  7.47699178e+00],
       [ 3.14745764e+00,  3.14745765e+00,  7.47699178e+00],
       [ 1.97491784e+00,  1.97491785e+00,  4.42161612e+00],
       [ 1.97491785e+00,  1.97491785e+00,  4.42161612e+00],
       [ 3.14904947e+00,  3.14904947e+00,  4.18064810e+00],
       [ 4.06983868e+00,  4.35881352e+00,  6.75710303e+00],
       [ 3.83740093e+00,  5.28122332e+00,  7.52582925e+00],
       [ 2.65294463e+00,  4.59239797e+00,  5.96657231e+00],
       [ 3.14745764e+00,  3.14745765e+00,  7.47699178e+00],
       [ 4.06983868e+00,  4.35881352e+00,  6.75710303e+00],
       [ 5.03817448e+00,  5.03817448e+00,  7.48345258e+00],
       [ 4.92046371e+00,  6.09817578e+00,  7.58799483e+00],
       [ 3.83740093e+00,  5.28122332e+00,  7.52582925e+00],
       [ 3.14745764e+00,  3.14745765e+00,  7.47699178e+00],
       [ 3.83740093e+00,  5.28122332e+00,  7.52582925e+00],
       [ 4.92046371e+00,  6.09817578e+00,  7.58799483e+00],
       [ 5.03817448e+00,  5.03817448e+00,  7.48345258e+00],
       [ 4.06983868e+00,  4.35881352e+00,  6.75710303e+00],
       [ 1.97491785e+00,  1.97491785e+00,  4.42161612e+00],
       [ 2.65294463e+00,  4.59239797e+00,  5.96657231e+00],
       [ 3.83740093e+00,  5.28122332e+00,  7.52582925e+00],
       [ 4.06983868e+00,  4.35881352e+00,  6.75710303e+00],
       [ 3.14904947e+00,  3.14904947e+00,  4.18064810e+00],
       [ 1.97491785e+00,  1.97491785e+00,  4.42161612e+00],
       [ 3.14904947e+00,  3.14904947e+00,  4.18064810e+00],
       [ 4.06983868e+00,  4.35881352e+00,  6.75710303e+00],
       [ 3.83740093e+00,  5.28122332e+00,  7.52582925e+00],
       [ 2.65294464e+00,  4.59239797e+00,  5.96657231e+00],
       [ 3.14904946e+00,  3.14904947e+00,  4.18064809e+00],
       [ 1.97491784e+00,  1.97491785e+00,  4.42161612e+00],
       [ 2.65294464e+00,  4.59239797e+00,  5.96657231e+00],
       [ 3.83740093e+00,  5.28122332e+00,  7.52582925e+00],
       [ 4.06983868e+00,  4.35881352e+00,  6.75710303e+00],
       [ 4.06983868e+00,  4.35881352e+00,  6.75710303e+00],
       [ 2.65294464e+00,  4.59239797e+00,  5.96657231e+00],
       [ 4.06983868e+00,  4.35881352e+00,  6.75710303e+00],
       [ 4.77058454e+00,  6.18436289e+00,  7.12511829e+00],
       [ 4.77058454e+00,  6.18436289e+00,  7.12511829e+00],
       [ 3.83740093e+00,  5.28122332e+00,  7.52582925e+00],
       [ 3.83740093e+00,  5.28122332e+00,  7.52582925e+00],
       [ 4.77058454e+00,  6.18436289e+00,  7.12511829e+00],
       [ 4.92046371e+00,  6.09817579e+00,  7.58799483e+00],
       [ 4.77058454e+00,  6.18436289e+00,  7.12511829e+00],
       [ 2.65294464e+00,  4.59239797e+00,  5.96657231e+00],
       [ 4.06983868e+00,  4.35881352e+00,  6.75710303e+00],
       [ 4.77058454e+00,  6.18436289e+00,  7.12511829e+00],
       [ 4.77058454e+00,  6.18436289e+00,  7.12511829e+00],
       [ 4.06983868e+00,  4.35881352e+00,  6.75710303e+00],
       [ 3.14745764e+00,  3.14745765e+00,  7.47699178e+00],
       [ 4.06983868e+00,  4.35881352e+00,  6.75710303e+00],
       [ 5.03817448e+00,  5.03817448e+00,  7.48345258e+00],
       [ 4.92046371e+00,  6.09817578e+00,  7.58799483e+00],
       [ 3.83740093e+00,  5.28122332e+00,  7.52582925e+00],
       [ 4.06983868e+00,  4.35881352e+00,  6.75710303e+00],
       [ 2.65294464e+00,  4.59239797e+00,  5.96657231e+00],
       [ 4.06983868e+00,  4.35881352e+00,  6.75710303e+00],
       [ 4.77058454e+00,  6.18436289e+00,  7.12511829e+00],
       [ 4.77058454e+00,  6.18436289e+00,  7.12511829e+00],
       [ 5.03817448e+00,  5.03817448e+00,  7.48345258e+00],
       [ 4.06983868e+00,  4.35881352e+00,  6.75710303e+00],
       [ 3.14745765e+00,  3.14745765e+00,  7.47699178e+00],
       [ 3.83740093e+00,  5.28122332e+00,  7.52582925e+00],
       [ 4.92046371e+00,  6.09817579e+00,  7.58799483e+00],
       [ 4.92046371e+00,  6.09817578e+00,  7.58799483e+00],
       [ 4.77058454e+00,  6.18436289e+00,  7.12511829e+00],
       [ 3.83740093e+00,  5.28122332e+00,  7.52582925e+00],
       [ 3.83740093e+00,  5.28122332e+00,  7.52582925e+00],
       [ 4.77058454e+00,  6.18436289e+00,  7.12511829e+00],
       [ 3.83740093e+00,  5.28122332e+00,  7.52582925e+00],
       [ 4.77058454e+00,  6.18436289e+00,  7.12511829e+00],
       [ 4.92046371e+00,  6.09817578e+00,  7.58799483e+00],
       [ 4.77058454e+00,  6.18436289e+00,  7.12511829e+00],
       [ 3.83740093e+00,  5.28122332e+00,  7.52582925e+00],
       [ 3.14745764e+00,  3.14745765e+00,  7.47699178e+00],
       [ 3.83740093e+00,  5.28122332e+00,  7.52582925e+00],
       [ 4.92046371e+00,  6.09817578e+00,  7.58799483e+00],
       [ 5.03817448e+00,  5.03817448e+00,  7.48345258e+00],
       [ 4.06983868e+00,  4.35881352e+00,  6.75710303e+00],
       [ 3.83740093e+00,  5.28122332e+00,  7.52582925e+00],
       [ 3.83740093e+00,  5.28122332e+00,  7.52582925e+00],
       [ 4.77058454e+00,  6.18436289e+00,  7.12511829e+00],
       [ 4.92046371e+00,  6.09817578e+00,  7.58799483e+00],
       [ 4.77058454e+00,  6.18436289e+00,  7.12511829e+00],
       [ 4.92046371e+00,  6.09817578e+00,  7.58799483e+00],
       [ 4.77058454e+00,  6.18436289e+00,  7.12511829e+00],
       [ 3.83740093e+00,  5.28122332e+00,  7.52582925e+00],
       [ 3.83740093e+00,  5.28122332e+00,  7.52582925e+00],
       [ 4.77058454e+00,  6.18436289e+00,  7.12511829e+00],
       [ 5.03817448e+00,  5.03817448e+00,  7.48345258e+00],
       [ 4.92046371e+00,  6.09817579e+00,  7.58799483e+00],
       [ 3.83740093e+00,  5.28122332e+00,  7.52582925e+00],
       [ 3.14745765e+00,  3.14745765e+00,  7.47699178e+00],
       [ 4.06983868e+00,  4.35881352e+00,  6.75710303e+00],
       [ 4.06983868e+00,  4.35881352e+00,  6.75710303e+00],
       [ 4.77058454e+00,  6.18436289e+00,  7.12511829e+00],
       [ 4.77058454e+00,  6.18436289e+00,  7.12511829e+00],
       [ 4.06983868e+00,  4.35881352e+00,  6.75710303e+00],
       [ 2.65294464e+00,  4.59239797e+00,  5.96657231e+00],
       [ 1.97491785e+00,  1.97491785e+00,  4.42161612e+00],
       [ 2.65294464e+00,  4.59239797e+00,  5.96657231e+00],
       [ 3.83740093e+00,  5.28122332e+00,  7.52582925e+00],
       [ 4.06983868e+00,  4.35881352e+00,  6.75710303e+00],
       [ 3.14904947e+00,  3.14904947e+00,  4.18064810e+00],
       [ 2.65294464e+00,  4.59239797e+00,  5.96657231e+00],
       [ 4.06983868e+00,  4.35881352e+00,  6.75710303e+00],
       [ 4.77058454e+00,  6.18436289e+00,  7.12511829e+00],
       [ 4.77058454e+00,  6.18436289e+00,  7.12511829e+00],
       [ 4.06983868e+00,  4.35881352e+00,  6.75710303e+00],
       [ 3.83740093e+00,  5.28122332e+00,  7.52582925e+00],
       [ 4.77058454e+00,  6.18436289e+00,  7.12511829e+00],
       [ 4.92046371e+00,  6.09817579e+00,  7.58799483e+00],
       [ 4.77058454e+00,  6.18436289e+00,  7.12511829e+00],
       [ 3.83740093e+00,  5.28122332e+00,  7.52582925e+00],
       [ 4.06983868e+00,  4.35881352e+00,  6.75710303e+00],
       [ 4.77058454e+00,  6.18436289e+00,  7.12511829e+00],
       [ 4.77058454e+00,  6.18436289e+00,  7.12511829e+00],
       [ 4.06983868e+00,  4.35881352e+00,  6.75710303e+00],
       [ 2.65294464e+00,  4.59239797e+00,  5.96657231e+00],
       [ 3.14904946e+00,  3.14904947e+00,  4.18064809e+00],
       [ 4.06983868e+00,  4.35881352e+00,  6.75710303e+00],
       [ 3.83740093e+00,  5.28122332e+00,  7.52582925e+00],
       [ 2.65294464e+00,  4.59239797e+00,  5.96657231e+00],
       [ 1.97491784e+00,  1.97491785e+00,  4.42161612e+00]])



@pytest.yield_fixture(scope="session")
def phonons():
    print ("Preparing phonons object.")
    # Setup crystal and EMT calculator
    atoms = bulk('Al', 'fcc', a=4.05)
    with TemporaryDirectory() as td:
        finite_difference = FiniteDifference(atoms=atoms,
                                             supercell=[5, 5, 5],
                                             folder=td)

        finite_difference.calculate_second(calculator=EMT())
        is_classic = False
        phonons_config = {'kpts': [5, 5, 5],
                          'is_classic': is_classic,
                          'temperature': 300,
                          'is_tf_backend':False,
                          'storage':'memory'}
        phonons = Phonons(finite_difference=finite_difference, **phonons_config)
        return phonons


def test_frequency_with_finite_difference(phonons):
    calculated_frequency = phonons.frequency
    np.testing.assert_equal(np.round(calculated_frequency - frequency, 2), 0)