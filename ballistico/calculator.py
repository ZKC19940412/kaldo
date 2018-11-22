import ballistico.constants as constants
import numpy as np
from ballistico.logger import Logger
import spglib as spg
import ballistico.atoms_helper as atoms_helper


# DIAGONALIZATION_ALGORITHM = scipy.linalg.lapack.zheev
DIAGONALIZATION_ALGORITHM = np.linalg.eigh

DELTA_THRESHOLD = 2
# DELTA_CORRECTION = scipy.special.erf (DELTA_THRESHOLD / np.sqrt (2))
DELTA_CORRECTION = 1

def calculate_density_of_states(frequencies, k_mesh, delta, num):
    n_modes = frequencies.shape[-1]
    frequencies = frequencies.reshape ((k_mesh[0], k_mesh[1], k_mesh[2], n_modes))
    n_k_points = np.prod (k_mesh)
    # increase_factor = 3
    omega_kl = np.zeros ((n_k_points, n_modes))
    for mode in range (n_modes):
        omega_kl[:, mode] = frequencies[..., mode].flatten ()
    # Energy axis and dos
    omega_e = np.linspace (0., np.amax (omega_kl) + 5e-3, num=num)
    dos_e = np.zeros_like (omega_e)
    # Sum up contribution from all q-points and branches
    for omega_l in omega_kl:
        diff_el = (omega_e[:, np.newaxis] - omega_l[np.newaxis, :]) ** 2
        dos_el = 1. / (diff_el + (0.5 * delta) ** 2)
        dos_e += dos_el.sum (axis=1)
    dos_e *= 1. / (n_k_points * np.pi) * 0.5 * delta
    return omega_e, dos_e

def diagonalize_second_order_single_k(qvec, atoms, second_order, list_of_replicas, replicated_atoms):
    list_of_replicas = list_of_replicas
    geometry = atoms.positions
    cell_inv = np.linalg.inv (atoms.cell)
    kpoint = 2 * np.pi * (cell_inv).dot (qvec)
    
    # TODO: remove this copy()
    second_order = second_order[0].copy()

    n_particles = geometry.shape[0]
    n_replicas = list_of_replicas.shape[0]

    mass = np.sqrt(atoms.get_masses ())
    second_order /= mass[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
    second_order /= mass[np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]
    
    chi_k = np.zeros (n_replicas).astype (complex)
    for id_replica in range (n_replicas):
        chi_k[id_replica] = np.exp (1j * list_of_replicas[id_replica].dot (kpoint))
    dyn_s = np.einsum('ialjb,l->iajb', second_order, chi_k)
    dxij = atoms_helper.apply_boundary (replicated_atoms, geometry[:, np.newaxis, np.newaxis] - (
            geometry[np.newaxis, :, np.newaxis] + list_of_replicas[np.newaxis, np.newaxis, :]))
    ddyn_s = 1j * np.einsum('ijla,l,ibljc->aibjc',
                       dxij,
                       chi_k,
                       second_order, optimize='greedy')
    prefactor = 1 / (constants.charge_of_electron) / constants.rydbergoverev * \
                (constants.bohroverangstrom ** 2) * 2 * constants.electron_mass * 1e4
    dyn = prefactor * dyn_s.reshape (n_particles * 3, n_particles * 3)
    ddyn = prefactor * ddyn_s.reshape (3, n_particles * 3, n_particles * 3) / constants.bohroverangstrom
    out = DIAGONALIZATION_ALGORITHM (dyn.reshape (n_particles * 3, n_particles * 3))
    eigenvals, eigenvects = out[0], out[1]
    # idx = eigenvals.argsort ()
    # eigenvals = eigenvals[idx]
    # eigenvects = eigenvects[:, idx]
    frequencies = np.abs (eigenvals) ** .5 * np.sign (eigenvals) / (np.pi * 2.)
    velocities = np.zeros ((frequencies.shape[0], 3), dtype=np.complex)
    vel = np.einsum('ki,aij,jq->akq',eigenvects.conj().T, ddyn, eigenvects, optimize='greedy')
    for alpha in range (3):
        for mu in range (n_particles * 3):
            if frequencies[mu] != 0:
                velocities[mu, alpha] = vel[alpha, mu, mu] / (2 * (2 * np.pi) * frequencies[mu])
    return frequencies * constants.toTHz, eigenvals, eigenvects, velocities * constants.toTHz * constants.bohr2nm

def calculate_second_all_grid(k_points, atoms, second_order, list_of_replicas, replicated_atoms):
    n_unit_cell = second_order.shape[1]
    n_k_points = k_points.shape[0]
    frequencies = np.zeros ((n_k_points, n_unit_cell * 3))
    eigenvalues = np.zeros ((n_k_points, n_unit_cell * 3))
    eigenvectors = np.zeros ((n_k_points, n_unit_cell * 3, n_unit_cell * 3)).astype (np.complex)
    velocities = np.zeros ((n_k_points, n_unit_cell * 3, 3)).astype(np.complex)
    for index_k in range (n_k_points):
        freq, eval, evect, vels = diagonalize_second_order_single_k (k_points[index_k], atoms, second_order, list_of_replicas, replicated_atoms)
        frequencies[index_k, :] = freq
        eigenvalues[index_k, :] = eval
        eigenvectors[index_k, :, :] = evect
        velocities[index_k, :, :] = vels
    return frequencies, eigenvalues, eigenvectors, velocities

def calculate_broadening(velocity, cellinv, k_size):
    # we want the last index of velocity (the coordinate index to dot from the right to rlattice vec
    # 10 = armstrong to nanometers
    delta_k = cellinv / k_size * 2 * np.pi
    base_sigma = ((np.tensordot (velocity * 10., delta_k, [-1, 1])) ** 2).sum (axis=-1)
    base_sigma = np.sqrt (base_sigma / 6.)
    return base_sigma / (2 * np.pi)


def gaussian_delta(params):
    # alpha is a factor that tells whats the ration between the width of the gaussian and the width of allowed phase space
    delta_energy = params[0]
    # allowing processes with width sigma and creating a gaussian with width sigma/2 we include 95% (erf(2/sqrt(2)) of the probability of scattering. The erf makes the total area 1
    sigma = params[1]
    # correction = scipy.special.erf(DELTA_THRESHOLD / np.sqrt(2))
    correction = 1
    return 1 / np.sqrt (2 * np.pi * sigma ** 2) * np.exp (- delta_energy ** 2 / (2 * sigma ** 2)) / correction

def calculate_gamma(atoms, frequencies, velocities, density, k_size, eigenvectors, list_of_replicas, third_order, sigma_in):
    prefactor = 1e-3 / (
            4. * np.pi) ** 3 * constants.avogadro ** 3 * constants.charge_of_electron ** 2 * constants.hbar
    coeff = 1000 * constants.hbar / constants.charge_of_electron
    
    
    nptk = np.prod (k_size)
    n_particles = atoms.positions.shape[0]
    cell_inv = np.linalg.inv (atoms.cell)
    masses = atoms.get_masses ()
    
    Logger ().info ('Lifetime calculation')
    n_modes = n_particles * 3
    ps = np.zeros ((2, np.prod (k_size), n_modes))
    
    # TODO: remove acoustic sum rule
    frequencies[0, :3] = 0
    
    n_replicas = list_of_replicas.shape[0]
    rlattvec = cell_inv * 2 * np.pi
    chi = np.zeros ((nptk, n_replicas), dtype=np.complex)
    for index_k in range (np.prod (k_size)):
        i_k = np.array (np.unravel_index (index_k, k_size, order='F'))
        k_point = i_k / k_size
        realq = np.matmul (rlattvec, k_point)
        for l in range (n_replicas):
            chi[index_k, l] = np.exp (1j * list_of_replicas[l].dot (realq))
    scaled_potential = third_order[0] / np.sqrt \
        (masses[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis, \
         np.newaxis, np.newaxis, np.newaxis])
    scaled_potential /= np.sqrt (masses[np.newaxis, np.newaxis, \
                                 np.newaxis, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis])
    scaled_potential /= np.sqrt (masses[np.newaxis, np.newaxis, \
                                 np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, np.newaxis])
    scaled_potential = scaled_potential.reshape (n_modes, n_replicas, n_modes, n_replicas, n_modes)
    Logger ().info ('Projection started')
    gamma = np.zeros ((2, nptk, n_modes))
    gamma_tensor = np.zeros ((2, nptk, n_modes, nptk * n_modes))
    n_modes = n_particles * 3
    nptk = np.prod (k_size)
    mapping, grid = spg.get_ir_reciprocal_mesh (k_size,
                                                atoms,
                                                is_shift=[0, 0, 0])
    unique_points, degeneracy = np.unique (mapping, return_counts=True)
    list_of_k = unique_points
    Logger ().info ('n_irreducible_q_points = ' + str (int (len (unique_points))) + ' : ' + str (unique_points))
    third_eigenv_np = eigenvectors.conj ()
    n_phonons = nptk * n_modes

    third_eigenv_tf = third_eigenv_np.swapaxes (1, 2).reshape (n_phonons, n_modes)


    for is_plus in (1, 0):
        if is_plus:
            Logger ().info ('\nCreation processes')
            second_eigenv_np = eigenvectors
            second_chi = chi
        else:
            Logger ().info ('\nAnnihilation processes')
            second_eigenv_np = eigenvectors.conj ()
            second_chi = chi.conj ()
        second_eigenv_tf = second_eigenv_np.swapaxes (1, 2).reshape (n_phonons, n_modes)
        for index_k in (list_of_k):
            i_k = np.array (np.unravel_index (index_k, k_size, order='F'))
            
            for mu in range (n_modes):
    
                first = eigenvectors[index_k, :, mu]
                first_projected_potential = np.einsum ('wlitj,w->litj', scaled_potential, first, optimize='greedy')
                # TODO: add a threshold instead of 0
                if frequencies[index_k, mu] != 0:
    
                    index_kp_vec = np.arange (np.prod (k_size))
                    i_kp_vec = np.array (np.unravel_index (index_kp_vec, k_size, order='F'))
                    i_kpp_vec = i_k[:, np.newaxis] + (int (is_plus) * 2 - 1) * i_kp_vec[:, :]
                    index_kpp_vec = np.ravel_multi_index (i_kpp_vec, k_size, order='F', mode='wrap')

                    freq_product_np = (frequencies[index_kp_vec, :, np.newaxis] * \
                                       frequencies[index_kpp_vec, np.newaxis, :])
                    if sigma_in is None:
                        velocities = velocities.real
                        velocities[0, :3, :] = 0
    
                        sigma_tensor_np = calculate_broadening ( \
                            velocities[index_kp_vec, :, np.newaxis, :] - \
                            velocities[index_kpp_vec, np.newaxis, :, :], cell_inv, k_size)
                        sigma_small = sigma_tensor_np
                    else:
                        sigma_small = sigma_in
                        
                    if is_plus:
                        freq_diff_np = np.abs (
                            frequencies[index_k, mu] + frequencies[index_kp_vec, :, np.newaxis] -\
                            frequencies[index_kpp_vec, np.newaxis, :])
                        density_fact_np = density[index_kp_vec, :, np.newaxis] - density[index_kpp_vec, np.newaxis, :]

                    else:
                        freq_diff_np = np.abs (
                            frequencies[index_k, mu] - frequencies[index_kp_vec, :, np.newaxis] -\
                            frequencies[index_kpp_vec, np.newaxis, :])
                        density_fact_np = .5 * (
                                1 + density[index_kp_vec, :, np.newaxis] + density[index_kpp_vec, np.newaxis, :])

                    condition = (freq_diff_np < DELTA_THRESHOLD * sigma_small) & (
                            frequencies[index_kp_vec, :, np.newaxis] != 0) & (
                                            frequencies[index_kpp_vec, np.newaxis, :] != 0)
                    interactions = np.array (np.where (condition)).T
					# TODO: Benchmark something fast like
                    # interactions = np.array(np.unravel_index (np.flatnonzero (condition), condition.shape)).T
                    if interactions.size != 0:
                        # Logger ().info ('interactions: ' + str (interactions.size))
                        index_kp_vec = interactions[:, 0]
                        index_kpp_vec = index_kpp_vec[index_kp_vec]
                        mup_vec = interactions[:, 1]
                        mupp_vec = interactions[:, 2]
                        nup_vec = np.ravel_multi_index (np.array ([index_kp_vec, mup_vec]),
                                                        np.array ([nptk, n_modes]), order='C')
                        nupp_vec = np.ravel_multi_index (np.array ([index_kpp_vec, mupp_vec]),
                                                         np.array ([nptk, n_modes]), order='C')
                        
                        dirac_delta = density_fact_np[index_kp_vec, mup_vec, mupp_vec]
                        
                        dirac_delta /= freq_product_np[index_kp_vec, mup_vec, mupp_vec]
                        if sigma_in is None:
                            gaussian = gaussian_delta ([freq_diff_np[index_kp_vec, mup_vec, mupp_vec],
                                                        sigma_small[index_kp_vec, mup_vec, mupp_vec]])
                        
                        else:
                            gaussian = gaussian_delta (
                                [freq_diff_np[index_kp_vec, mup_vec, mupp_vec], sigma_in])
                        
                        dirac_delta *= gaussian
                        

						# TODO: find a better name
                        temp = np.einsum ('litj,al,at,aj,ai->a', first_projected_potential,
                                                         second_chi[index_kp_vec],
                                                         chi.conj()[index_kpp_vec],
                                                         third_eigenv_tf[nupp_vec],
                                                         second_eigenv_tf[nup_vec], optimize='greedy')
                        
                        
                        for i in range(nup_vec.shape[0]):
	                        nu_p = nup_vec[i]
	                        gamma_tensor[is_plus, index_k, mu, nu_p] += np.abs (temp[i]) ** 2 * dirac_delta[i]
                        gamma[is_plus, index_k, mu] = np.sum (gamma_tensor[is_plus, index_k, mu], axis=-1)
                        ps[is_plus, index_k, mu] += np.sum (dirac_delta)

                        gamma[is_plus, index_k, mu] /= frequencies[index_k, mu]
                        ps[is_plus, index_k, mu] /= frequencies[index_k, mu]
                    # Logger ().info ('q-point   = ' + str (index_k))
                    # Logger ().info ('mu-branch = ' + str (mu))
                
    for index_k, (associated_index, gp) in enumerate (zip (mapping, grid)):
        ps[:, index_k, :] = ps[:, associated_index, :]
        gamma[:, index_k, :] = gamma[:, associated_index, :]
        gamma_tensor[:, index_k, :] = gamma_tensor[:, associated_index, :]
    
    gamma = gamma * prefactor / nptk
    gamma_tensor = gamma_tensor * prefactor / nptk
    ps = ps / nptk / (2 * np.pi) ** 3
    gamma = np.sum (gamma, axis=0)
    return gamma, gamma_tensor.reshape((2, nptk, n_modes, nptk, n_modes))
