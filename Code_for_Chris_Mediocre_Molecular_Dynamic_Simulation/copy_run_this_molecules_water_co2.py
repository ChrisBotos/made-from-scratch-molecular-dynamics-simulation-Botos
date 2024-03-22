"""Import"""
import numpy as np
import time
import pygame
import cProfile
import pstats
import random


"""Parameters"""
profile = True

number_of_particles = np.array([80, 40, 0, 120])
number_of_types = 4
colors = np.array([(0, 255, 255), (255, 0, 0), (0, 0, 255), (0, 255, 0)])  # This is rgb.
masses = np.array([0.17 * 10 ** -26, 1.99 * 10 ** -26, 2.33 * 10 ** -26, 2.66 * 10 ** -26])  # Kilograms. This corresponds to the above color list. Hydrogen, Carbon, Nitrogen, Oxygen.
charges = np.array([0., 0., 0., 0.])  # Charges for free atoms not in molecules

x_border = 700  # For example 100 means that it goes from x 0 to x 100.
y_border = 700
z_border = 700

minimum_starting_x = 300
minimum_starting_y = 300
minimum_starting_z = 600

maximum_starting_x = 500
maximum_starting_y = 500
maximum_starting_z = 600

time_step = 1 * 10 ** -34
# Time_step defines how important the momentary force is.

min_sigma = 3 * 10 ** -10  # Sigma is the finite distance at which the inter-particle potential is zero.
max_sigma = 5 * 10 ** -10

min_epsilon = 1000  # Epsilon is the well-depth.
max_epsilon = 5000

pixel_size = 2 * 10 ** -11  # Meters

force_range = 1000 * pixel_size  # Meters.
force_cap = 10 ** +28 # Newtons.

newton_third_law = True

Leonard_Jones = True
attract = False
Dipole_Dipole = True

have_borders = True
have_energy_loss_when_border_collision = True
border_collision_energy_loss = 0.4  # 0.2 is 20% of its kinetic energy
border_collision_percentage_remaining_velocity = 1 - border_collision_energy_loss ** 0.5

have_molecules = True

array_of_molecule_types_atoms = [[0, 1, 0, 2],
                                 [2, 0, 0, 1]]  # The numbers correspond to the amount of atoms of that index inside the molecule.

# We put one of the atoms in the geometric shape as 0. Then we give the x, y and z distances of any other atom from it in lists.
# We put them in order from lightest to heavier like in the masses and colors lists.
array_of_molecule_types_shape = [[[0, -1.163 * 10 ** -10 / pixel_size, 1.163 * 10 ** -10 / pixel_size],
                                  [0, 0, 0],
                                  [0, 0, 0]],

                                  [[-0.757 * 10 ** -10 / pixel_size, 0.757 * 10 ** -10 / pixel_size , 0],
                                  [1.22 * 10 ** -10 / pixel_size, 1.22 * 10 ** -10 / pixel_size, 0],
                                  [0, 0, 0]]]

array_of_molecule_types_charge = [[0, 0, 0],
                                  [0.417, 0.417, -0.834]]  # The numbers correspond to the charge of atoms of that index inside the molecule.

# for i in range(10):
#     array_of_molecule_types_atoms = array_of_molecule_types_atoms + [array_of_molecule_types_atoms[1]]
#     array_of_molecule_types_shape = array_of_molecule_types_shape + [array_of_molecule_types_shape[1]]
#     array_of_molecule_types_charge = array_of_molecule_types_charge + [array_of_molecule_types_charge[1]]

z_multiplier = 0.01

scrolling_speed = 1

fps_cap = 10000000000000


"""Functions"""


'''Force Function'''
def force_function(distances, charges, atom_charge, time_step, A, B, pixel_size, force_cap, Leonard_Jones, attract):  # Negative means attraction and positive means repulsion.
    # I use the force calculate from the potential energy according to the Lennard-Jones potential.
    # The force between two atoms in a Lennard-Jones potential can be obtained by taking the negative gradient of the potential energy with respect to the separation distance.

    # This is the equation for the Leonard-Jones potential. We simplify it for computational speed:
    # F = 4 * epsilon / distances * (2 * (sigma / (distances)) ** 12 - (sigma / (distances)) ** 6)

    # A = 4 * epsilon
    # B = sigma ** 6

    F = np.zeros(len(distances))

    if Leonard_Jones:
        F += A / distances * (2 * B ** 2 / distances ** 12 - B / distances ** 6)

        force_mistake = np.abs(F) > force_cap

        F[force_mistake] = np.sign(F[force_mistake]) * force_cap

    if attract:
        F += -distances

    if Dipole_Dipole:
        F += 8987500000 * charges * atom_charge / distances ** 2  # This is Coulomb's Law.

    return F


'''Split Vectors to its x,y,z vectors Function'''
def F_to_Fx_Fy_Fz(F, sin_phi, cos_phi, sin_theta, cos_theta):  # These are all arrays
    Fy = sin_theta * F

    F_projection_to_x_z_plane = cos_theta * F

    Fx = cos_phi * F_projection_to_x_z_plane

    Fz = sin_phi * F_projection_to_x_z_plane

    return np.array([Fx, Fy, Fz])


'''Position after all border collisions Function'''
def collide_until_within_borders(positions, velocities, molecules, right_border, have_energy_loss_when_border_collision, border_collision_percentage_remaining_velocity, have_molecules):
    not_within_borders_mask = (positions > right_border) | (positions < 0)

    if not np.any(not_within_borders_mask):
        return positions, velocities

    initial_out_of_borders_positions = positions[not_within_borders_mask]
    initial_out_of_borders_velocities = velocities[not_within_borders_mask]
    initial_out_of_borders_molecules = molecules[not_within_borders_mask]


    '''Looking at only one atom per molecule'''
    if have_molecules:

        out_of_borders_positions = np.array([])
        out_of_borders_velocities = np.array([])
        out_of_borders_molecules = np.array([])

        molecules_covered = np.array([])

        single_appearance = np.array([]).astype(int)  # This is to battle the problem where we have the same number.

        counter = 0
        for molecule in initial_out_of_borders_molecules:

            if molecule == 0:
                free_atom = initial_out_of_borders_positions == initial_out_of_borders_positions[counter]

                out_of_borders_positions = np.append(out_of_borders_positions, initial_out_of_borders_positions[free_atom])
                out_of_borders_velocities = np.append(out_of_borders_velocities, initial_out_of_borders_velocities[free_atom])
                out_of_borders_molecules = np.append(out_of_borders_molecules, molecule)

                single_appearance = np.append(single_appearance, True)

                continue

            if np.isin(molecule, molecules_covered):
                single_appearance = np.append(single_appearance, False)
                continue

            in_the_same_molecule_mask = initial_out_of_borders_molecules == molecule

            particles = initial_out_of_borders_positions[in_the_same_molecule_mask]

            if np.any(particles < 0):

                most_left_atom_in_molecule = np.min(particles)

                out_of_borders_positions = np.append(out_of_borders_positions, most_left_atom_in_molecule)
                out_of_borders_velocities = np.append(out_of_borders_velocities, initial_out_of_borders_velocities[0])  # They all have the same velocity.
                out_of_borders_molecules = np.append(out_of_borders_molecules, molecule)

            else:

                most_right_atom_in_molecule = np.max(particles)

                out_of_borders_positions = np.append(out_of_borders_positions, most_right_atom_in_molecule)
                out_of_borders_velocities = np.append(out_of_borders_velocities, initial_out_of_borders_velocities[0])
                out_of_borders_molecules = np.append(out_of_borders_molecules, molecule)

            molecules_covered = np.append(molecules_covered, molecule)

            single_appearance = np.append(single_appearance, True)


        not_within_borders_mask[not_within_borders_mask] = single_appearance

        negative_positions = out_of_borders_positions < 0

        abs_out_of_borders_positions = np.abs(out_of_borders_positions)

        collisions = abs_out_of_borders_positions // right_border

        abs_out_of_borders_velocities = np.abs(out_of_borders_velocities)

    else:
        out_of_borders_molecules = initial_out_of_borders_molecules

        negative_positions = initial_out_of_borders_positions < 0

        abs_out_of_borders_positions = np.abs(initial_out_of_borders_positions)

        collisions = abs_out_of_borders_positions // right_border

        abs_out_of_borders_velocities = np.abs(initial_out_of_borders_velocities)

    # Here we try and simulate all the collision that the particle would have had in the time step in case it travelled a distance larger than the size of the allowed area.


    '''Reducing the velocity and position according to border collision energy loss.'''
    if have_energy_loss_when_border_collision:

        num_of_collisions = collisions.astype(np.int32)
        num_of_collisions[negative_positions] += 1

        for i in range(np.max(num_of_collisions)):
            collide = num_of_collisions > i

            # Getting the new reduced velocities. Ek' = Ek - Ek * loss => V' = V - V * loss ** 0.5
            reduced_out_of_borders_velocities = abs_out_of_borders_velocities.astype(np.float64)
            reduced_out_of_borders_velocities[collide] = abs_out_of_borders_velocities[collide] * border_collision_percentage_remaining_velocity  # V' = V * (1 - loss ** 0.5)

            abs_out_of_borders_positions[negative_positions & collide] = border_collision_percentage_remaining_velocity * abs_out_of_borders_positions[negative_positions & collide]  # time = V'/d' = V/d => d' = V'* d/V

            abs_out_of_borders_positions[~negative_positions & collide] = border_collision_percentage_remaining_velocity * (abs_out_of_borders_positions[~negative_positions & collide] - right_border) + right_border

            abs_out_of_borders_velocities = reduced_out_of_borders_velocities

        # We define the collisions again because with the reducing of the kinetic energy maybe not as many border collisions happened.
        collisions = abs_out_of_borders_positions // right_border


    '''Positions according to which border they hit last in the time step that passed'''
    hit_the_right_border_last = collisions % 2 != 0

    copy_not_within_borders_mask = not_within_borders_mask.copy()

    if have_molecules:
        '''Finding how much the positions of out of borders atoms changed and then making that change to all the atoms in the same molecule'''

        copy_not_within_borders_mask[not_within_borders_mask] = hit_the_right_border_last
        difference_right_last = positions[copy_not_within_borders_mask] - (right_border - (abs_out_of_borders_positions[hit_the_right_border_last] - right_border * collisions[hit_the_right_border_last]))

        copy_not_within_borders_mask[not_within_borders_mask] = ~hit_the_right_border_last
        difference_left_last = positions[copy_not_within_borders_mask] - (abs_out_of_borders_positions[~hit_the_right_border_last] - right_border * collisions[~hit_the_right_border_last])

        right_counter = 0
        left_counter = 0
        c = 0
        for i in out_of_borders_molecules:

            if i == 0:
                continue  # It will change below.

            same_molecule = molecules == i

            if hit_the_right_border_last[c]:
                positions[same_molecule] -= difference_right_last[right_counter]
                right_counter += 1

            else:
                positions[same_molecule] -= difference_left_last[left_counter]
                left_counter += 1

            c += 1

    else:
        copy_not_within_borders_mask[not_within_borders_mask] = hit_the_right_border_last
        positions[copy_not_within_borders_mask] = right_border - (abs_out_of_borders_positions[hit_the_right_border_last] - right_border * collisions[hit_the_right_border_last])

        copy_not_within_borders_mask[not_within_borders_mask] = ~hit_the_right_border_last
        positions[copy_not_within_borders_mask] = abs_out_of_borders_positions[~hit_the_right_border_last] - right_border * collisions[~hit_the_right_border_last]


    '''Changing the velocities'''
    velocity_directions = np.ones(len(abs_out_of_borders_positions))

    velocity_directions[hit_the_right_border_last] = -1  # Negative direction is towards the left.

    if have_molecules:

        c = 0
        for molecule in out_of_borders_molecules:

            if molecule == 0:
                continue  # It will change below.

            same_molecule = molecules == molecule
            velocities[same_molecule] = abs_out_of_borders_velocities[c] * velocity_directions[c]

            c += 1

        free_atoms = out_of_borders_molecules == 0

        if np.any(free_atoms):
            not_within_borders_mask[not_within_borders_mask] = free_atoms

            velocities[not_within_borders_mask] = abs_out_of_borders_velocities * velocity_directions

    else:
        velocities[not_within_borders_mask] = abs_out_of_borders_velocities * velocity_directions

    return positions, velocities


'''Sum forces calculating function'''
def get_forces(position, copy_of_particle_array, sigma_array, epsilon_array, time_step, force_range, masses, pixel_size, force_cap, Leonard_Jones, attract, have_molecules):

        x = position[0]
        y = position[1]
        z = position[2]
        color = position[6]
        molecule = position[7]
        atom_charge = position[8]


        '''Particles in range'''
        particles_in_range_mask = (np.abs(copy_of_particle_array[0] - x) <= force_range) & (np.abs(copy_of_particle_array[1] - y) <= force_range) & (np.abs(copy_of_particle_array[2] - z) <= force_range)

        if have_molecules and molecule != 0:
            not_in_the_same_molecule_mask = copy_of_particle_array[7] != molecule

            particles_in_range_mask = particles_in_range_mask & not_in_the_same_molecule_mask

        particles_in_range = copy_of_particle_array[:, particles_in_range_mask]


        '''Calculating their distances from our particle'''
        # This will also catch our particle too, but the distance will be 0 and so the F will be 0 anyway.
        # Distance is calculated by (dx^2 + dy^2 + dz^2) ^ 1/2.
        init_dx = x - particles_in_range[0]
        init_dy = y - particles_in_range[1]
        init_dz = z - particles_in_range[2]
        dx = np.abs(init_dx)
        dy = np.abs(init_dy)
        dz = np.abs(init_dz)
        distances = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5

        not_on_top_mask = distances != 0  # If they are exactly on top of each other what direction would one push the other to?

        distances = distances[not_on_top_mask]
        dx = dx[not_on_top_mask]
        dy = dy[not_on_top_mask]
        dz = dz[not_on_top_mask]
        init_dx = init_dx[not_on_top_mask]
        init_dy = init_dy[not_on_top_mask]
        init_dz = init_dz[not_on_top_mask]


        '''Calculating the charges of the particles in range'''
        charges = particles_in_range[8]
        charges = charges[not_on_top_mask]


        '''Calculating the forces acted on our particle'''
        # We need to use the sigma_array and get the sigmas for the appropriate color of each particle in range.
        sigmas_of_the_in_range_particles = sigma_array[int(color), np.asarray(particles_in_range[6], dtype=np.int32)][not_on_top_mask]  # The rows of the sigma array correspond to the force receivers and the columns to the force sources.

        # We need to use the epsilon_array and get the epsilons for the appropriate color of each particle in range.
        epsilons_of_the_in_range_particles = epsilon_array[int(color), np.asarray(particles_in_range[6], dtype=np.int32)][not_on_top_mask]  # The rows of the epsilon array correspond to the force receivers and the columns to the force sources.

        sum_forces = force_function(distances, charges, atom_charge, time_step, sigmas_of_the_in_range_particles, epsilons_of_the_in_range_particles, pixel_size, force_cap, Leonard_Jones, attract)


        not_zero_dx_mask = dx != 0
        not_zero_dy_mask = dy != 0
        not_zero_dz_mask = dz != 0

        number_of_in_range_particles = len(distances)

        d_projection_to_x_z_plane = (dx ** 2 + dz ** 2) ** 0.5

        sin_phi = np.zeros(number_of_in_range_particles)
        sin_phi[not_zero_dz_mask] = dz[not_zero_dz_mask] / d_projection_to_x_z_plane[not_zero_dz_mask]  # phi is the x to d_projection_to_x_z_plane angle.

        cos_phi = np.zeros(number_of_in_range_particles)
        cos_phi[not_zero_dx_mask] = dx[not_zero_dx_mask] / d_projection_to_x_z_plane[not_zero_dx_mask]

        sin_theta = np.zeros(number_of_in_range_particles)
        sin_theta[not_zero_dy_mask] = dy[not_zero_dy_mask] / distances[not_zero_dy_mask]  # theta is the d_projection_to_x_z_plane to distances angle.

        cos_theta = d_projection_to_x_z_plane / distances

        F_xyz = F_to_Fx_Fy_Fz(sum_forces, sin_phi, cos_phi, sin_theta, cos_theta)

        Fx = F_xyz[0]
        Fy = F_xyz[1]
        Fz = F_xyz[2]

        Fx = np.sum(Fx * np.sign(init_dx))
        Fy = np.sum(Fy * np.sign(init_dy))
        Fz = np.sum(Fz * np.sign(init_dz))

        return np.array([Fx, Fy, Fz])


"""Random Arrays"""


'''Random Positions Particle Array'''
def random_positions_particle_array(number_of_particles, number_of_types, minimum_starting_x, minimum_starting_y, minimum_starting_z, maximum_starting_x, maximum_starting_y, maximum_starting_z):
    x_velocity = np.zeros(sum(number_of_particles))  # We are starting with velocity of zero.
    y_velocity = np.zeros(sum(number_of_particles))
    z_velocity = np.zeros(sum(number_of_particles))


    '''Coloring'''
    colors_line = np.zeros(sum(number_of_particles))

    # Here we assign colors to the particles where 0 is the first color in the colors list, 1 is the second and so on...
    current_atom_index = 0
    for atom_type in range(number_of_types):

        for _ in range(number_of_particles[atom_type]):

            colors_line[current_atom_index] = atom_type

            current_atom_index += 1


    '''Free atom charges'''
    charges_line = np.zeros(sum(number_of_particles))

    # Here we assign charges to the particles where 0 is the first charge in the charges array, 1 is the second and so on...
    current_atom_index = 0
    for atom_type in range(number_of_types):

        for _ in range(number_of_particles[atom_type]):
            charges_line[current_atom_index] = charges[atom_type]

            current_atom_index += 1


    '''Separating the atoms to different molecules'''
    belongs_to_molecule = np.zeros(sum(number_of_particles))  # Zero means that it belongs to no molecules. Otherwise, the number corresponds to the molecule.

    if have_molecules:

        current_atom_index = 0

        for atom_type in range(number_of_types):

            molecule_number = 1
            molecule_type = 0

            counter = 0
            while counter < number_of_particles[atom_type]:

                if array_of_molecule_types_atoms[molecule_type][atom_type] != 0:

                    '''Numbering the atoms'''
                    for atom_of_this_molecule in range(array_of_molecule_types_atoms[molecule_type][atom_type]):
                        belongs_to_molecule[current_atom_index] = molecule_number

                        current_atom_index += 1

                        counter += 1

                molecule_number += 1

                molecule_type += 1

                if molecule_type >= len(array_of_molecule_types_atoms):
                    molecule_type = 0


    '''Sorting molecules together in the arrays and assigning their atoms positions'''
    if have_molecules:
        # Get the indices that would sort the array based on the molecules.
        sorted_idx = np.argsort(belongs_to_molecule + colors_line / 10)

        # Use the sorted indices to rearrange the array.
        belongs_to_molecule = belongs_to_molecule[sorted_idx]
        colors_line = colors_line[sorted_idx]
        charges_line = charges_line[sorted_idx]


        x_coordinates = np.zeros(sum(number_of_particles))
        y_coordinates = np.zeros(sum(number_of_particles))
        z_coordinates = np.zeros(sum(number_of_particles))

        molecule_center_x = random.uniform(minimum_starting_x, maximum_starting_x)
        molecule_center_y = random.uniform(minimum_starting_y, maximum_starting_y)
        molecule_center_z = random.uniform(minimum_starting_z, maximum_starting_z)

        molecule_type = -1  # A value that this will never be as initial.
        previous_molecule = -1  # A value that this will never be as initial.
        atom_in_molecule = 0
        for atom in range(sum(number_of_particles)):

            molecule = belongs_to_molecule[atom]

            if molecule == 0:

                x_coordinates[atom] = molecule_center_x
                y_coordinates[atom] = molecule_center_y
                z_coordinates[atom] = molecule_center_z

                previous_molecule = molecule

                molecule_center_x = random.uniform(minimum_starting_x, maximum_starting_x)
                molecule_center_y = random.uniform(minimum_starting_y, maximum_starting_y)
                molecule_center_z = random.uniform(minimum_starting_z, maximum_starting_z)

            else:

                if molecule != previous_molecule:
                    atom_in_molecule = 0

                    previous_molecule = molecule

                    molecule_center_x = random.uniform(minimum_starting_x, maximum_starting_x)
                    molecule_center_y = random.uniform(minimum_starting_y, maximum_starting_y)
                    molecule_center_z = random.uniform(minimum_starting_z, maximum_starting_z)

                    if molecule_type != -1:
                        molecule_type += 1

                    else:
                        molecule_type = 0  # Accounting for the first run.

                    if molecule_type >= len(array_of_molecule_types_atoms):
                        molecule_type = 0

                x_coordinates[atom] = molecule_center_x + array_of_molecule_types_shape[molecule_type][0][atom_in_molecule]
                y_coordinates[atom] = molecule_center_y + array_of_molecule_types_shape[molecule_type][1][atom_in_molecule]
                z_coordinates[atom] = molecule_center_z + array_of_molecule_types_shape[molecule_type][2][atom_in_molecule]

                atom_in_molecule += 1

    else:
        x_coordinates = np.random.uniform(minimum_starting_x, maximum_starting_x, sum(number_of_particles))
        y_coordinates = np.random.uniform(minimum_starting_y, maximum_starting_y, sum(number_of_particles))
        z_coordinates = np.random.uniform(minimum_starting_z, maximum_starting_z, sum(number_of_particles))

    '''Charges of Molecule's atoms'''
    molecule_type = 0
    counter = 0
    previous_molecule = 1
    atom_in_molecule = 0
    for molecule in belongs_to_molecule:

        if molecule != 0:

            if molecule == previous_molecule:
                charges_line[counter] = array_of_molecule_types_charge[molecule_type][atom_in_molecule]

                atom_in_molecule += 1

            else:
                previous_molecule = molecule

                atom_in_molecule = 0

                molecule_type += 1

                if molecule_type >= len(array_of_molecule_types_atoms):
                    molecule_type = 0

                charges_line[counter] = array_of_molecule_types_charge[molecule_type][atom_in_molecule]

                atom_in_molecule += 1

        counter += 1


    '''Resizing'''
    x_coordinates *= pixel_size
    y_coordinates *= pixel_size
    z_coordinates *= pixel_size


    particle_array = np.array([
        x_coordinates,
        y_coordinates,
        z_coordinates,
        x_velocity,
        y_velocity,
        z_velocity,
        colors_line,
        belongs_to_molecule,
        charges_line
    ])

    return particle_array


pa = random_positions_particle_array(number_of_particles, number_of_types, minimum_starting_x, minimum_starting_y, minimum_starting_z, maximum_starting_x, maximum_starting_y, maximum_starting_z)
print(pa, 'pa\n')


'''Random Sigma Array'''
def random_sigma_array(number_of_types, min_sigma, max_sigma, newton_third_law, Leonard_Jones):
    if number_of_types == 0:
        return np.array([[]])

    sigma_array = np.random.uniform(min_sigma, max_sigma, number_of_types)
    for _ in range(number_of_types - 1):
        sigma_array = np.vstack((sigma_array, np.random.uniform(min_sigma, max_sigma, number_of_types)))

    if newton_third_law:
        # Make the elements below the main diagonal the same as above.
        sigma_array[np.tril_indices(sigma_array.shape[0], k=-1)] = sigma_array.T[np.tril_indices(sigma_array.shape[0], k=-1)]

    '''For computational speed do the math on sigma outside the main loop.'''
    if Leonard_Jones:
        B = sigma_array ** 6
    else:
        B = sigma_array

    return sigma_array, B


sa = random_sigma_array(number_of_types, min_sigma, max_sigma, newton_third_law, Leonard_Jones)[1]
print(sa, 'sa\n')


'''Random Epsilon Array'''
def random_epsilon_array(number_of_types, min_epsilon, max_epsilon, newton_third_law, Leonard_Jones):
    if number_of_types == 0:
        return np.array([[]])

    epsilon_array = np.random.uniform(min_epsilon, max_epsilon, number_of_types)
    for _ in range(number_of_types - 1):
        epsilon_array = np.vstack((epsilon_array, np.random.uniform(min_epsilon, max_epsilon, number_of_types)))

    if newton_third_law:
        # Make the elements below the main diagonal the same as above.
        epsilon_array[np.tril_indices(epsilon_array.shape[0], k=-1)] = epsilon_array.T[np.tril_indices(epsilon_array.shape[0], k=-1)]

    '''For computational speed do the math on epsilon outside the main loop.'''
    if Leonard_Jones:
        A = epsilon_array * 4
    else:
        A = epsilon_array

    return epsilon_array, A


ea = random_epsilon_array(number_of_types, min_epsilon, max_epsilon, newton_third_law, Leonard_Jones)[1]
print(ea, 'ea\n')


"""Moving the particles"""
def move(particle_array, sigma_array, epsilon_array, time_step, force_range, have_borders, have_energy_loss_when_border_collision, border_collision_percentage_remaining_velocity, masses, pixel_size, force_cap, Leonard_Jones, attract):

    copy_of_particle_array = particle_array.copy()

    Fx = 0
    Fy = 0
    Fz = 0
    previous_molecule = 1
    c = 0
    molecule_length = 0
    molecule_mass = 0
    for particle in copy_of_particle_array.T:

        color = particle[6]
        molecule = particle[7]

        mass = masses[int(color)]

        F_xyz = get_forces(particle, copy_of_particle_array, sigma_array, epsilon_array, time_step, force_range, masses, pixel_size, force_cap, Leonard_Jones, attract, have_molecules)


        '''Changing Velocities'''
        # x increases towards the right.
        # y increases towards down.
        # z increases towards out.

        if have_molecules and molecule != 0:

            if molecule == previous_molecule:
                Fx += F_xyz[0]
                Fy += F_xyz[1]
                Fz += F_xyz[2]

                c += 1
                molecule_length += 1
                molecule_mass += mass

                continue

            else:
                previous_molecule = molecule

                ax = Fx / molecule_mass
                ay = Fy / molecule_mass
                az = Fz / molecule_mass

                for atom_in_molecule in range(molecule_length):
                    # For the cases where some of dx, dy, dz are 0 np.sign() returns 0.
                    # For acceleration negative means attraction and positive means repulsion just like the forces.

                    current_atom = c - molecule_length + atom_in_molecule

                    particle_array[3, current_atom] += ax * time_step
                    particle_array[4, current_atom] += ay * time_step
                    particle_array[5, current_atom] += az * time_step

                    '''Changing Positions'''
                    particle_array[0, current_atom] += copy_of_particle_array[3, current_atom] * time_step + 0.5 * particle_array[3, current_atom] * time_step
                    particle_array[1, current_atom] += copy_of_particle_array[4, current_atom] * time_step + 0.5 * particle_array[4, current_atom] * time_step
                    particle_array[2, current_atom] += copy_of_particle_array[5, current_atom] * time_step + 0.5 * particle_array[5, current_atom] * time_step


                '''Counting for the current atom which does not belong to the molecule we moved'''
                molecule_length = 1
                molecule_mass = mass

                Fx = F_xyz[0]
                Fy = F_xyz[1]
                Fz = F_xyz[2]


        else:
            Fx = F_xyz[0]
            Fy = F_xyz[1]
            Fz = F_xyz[2]

            ax = Fx / mass
            ay = Fy / mass
            az = Fz / mass

            # For the cases where some of dx, dy, dz are 0 np.sign() returns 0.
            # Negative means attraction and positive means repulsion just like the forces.

            particle_array[3, c] += ax * time_step
            particle_array[4, c] += ay * time_step
            particle_array[5, c] += az * time_step

            '''Changing Positions'''
            particle_array[0, c] += copy_of_particle_array[3, c] * time_step + 0.5 * particle_array[3, c] * time_step
            particle_array[1, c] += copy_of_particle_array[4, c] * time_step + 0.5 * particle_array[4, c] * time_step
            particle_array[2, c] += copy_of_particle_array[5, c] * time_step + 0.5 * particle_array[5, c] * time_step

            Fx = 0
            Fy = 0
            Fz = 0

            molecule_length = 0
            molecule_mass = 0

        c += 1


    '''Last molecule'''
    if have_molecules and molecule_length != 0:
        ax = Fx / molecule_mass
        ay = Fy / molecule_mass
        az = Fz / molecule_mass

        for atom_in_molecule in range(molecule_length):
            # For the cases where some of dx, dy, dz are 0 np.sign() returns 0.
            # For acceleration negative means attraction and positive means repulsion just like the forces.

            current_atom = c - molecule_length + atom_in_molecule

            particle_array[3, current_atom] += ax * time_step
            particle_array[4, current_atom] += ay * time_step
            particle_array[5, current_atom] += az * time_step

            '''Changing Positions'''
            particle_array[0, current_atom] += particle_array[3, current_atom] * time_step + 0.5 * particle_array[3, current_atom] * time_step
            particle_array[1, current_atom] += particle_array[4, current_atom] * time_step + 0.5 * particle_array[4, current_atom] * time_step
            particle_array[2, current_atom] += particle_array[5, current_atom] * time_step + 0.5 * particle_array[5, current_atom] * time_step


    '''Borders'''
    if have_borders:
        # In case the new position is two or more times as fat the x size of the box we need to calculate for multiple collisions with both borders.
        x_collisions = collide_until_within_borders(particle_array[0], particle_array[3], particle_array[7], x_border * pixel_size, have_energy_loss_when_border_collision, border_collision_percentage_remaining_velocity, have_molecules)

        particle_array[0] = x_collisions[0]
        particle_array[3] = x_collisions[1]

        y_collisions = collide_until_within_borders(particle_array[1], particle_array[4], particle_array[7], y_border * pixel_size, have_energy_loss_when_border_collision, border_collision_percentage_remaining_velocity, have_molecules)

        particle_array[1] = y_collisions[0]
        particle_array[4] = y_collisions[1]

        z_collisions = collide_until_within_borders(particle_array[2], particle_array[5], particle_array[7], z_border * pixel_size, have_energy_loss_when_border_collision, border_collision_percentage_remaining_velocity, have_molecules)

        particle_array[2] = z_collisions[0]
        particle_array[5] = z_collisions[1]

    return(particle_array)


"""Imaging"""


"""Main game loop"""
if __name__ == "__main__":
    # Initialize Pygame
    pygame.init()

    # Set up display
    width, height = 1000, 750
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Crappy MD Simulator by Chris")

    # Colors
    white = (255, 255, 255)
    black = (0, 0, 0)

    print('And so it begins...\n')

    if profile:
        # Create a profile object
        profiler = cProfile.Profile()

        # Start profiling
        profiler.enable()

    start_time = time.time()
    pause = False
    y_scroll = 0
    x_scroll = 0
    round = 1

    running = True
    while running:
        # Clear the screen
        screen.fill(black)

        if round % 100 == 0:
            print(round, 'round\n')
            print('Speed is', (time.time() - start_time) / 100, 'seconds per round.')

            Ukin = 0.5 * masses[pa[6].astype(int)] * (pa[3] ** 2 + pa[4] ** 2 + pa[5] ** 2)
            print('Total Kinetic energy is', np.sum(Ukin), 'Joules\n')
            start_time = time.time()

        if not pause:
            pa = move(pa, ea, sa, time_step, force_range, have_borders, have_energy_loss_when_border_collision, border_collision_percentage_remaining_velocity, masses, pixel_size, force_cap, Leonard_Jones, attract)

        # Get the indices that would sort the array based on the third row (z position)
        sorted_indices = np.argsort(pa[2])

        # Use the sorted indices to rearrange the array
        z_sorted_pa = pa[:, sorted_indices].copy()

        # Resize
        z_sorted_pa[:3] /= pixel_size

        # Draw points on the screen
        for x, y, z, color in zip(z_sorted_pa[0], z_sorted_pa[1], z_sorted_pa[2], z_sorted_pa[6]):
            rad = z * z_multiplier
            pygame.draw.circle(screen, colors[int(color)], (x + x_scroll, y + y_scroll), rad)

        # Draw borders
        pygame.draw.line(screen, white, (0 + x_scroll, 0 + y_scroll), (x_border + x_scroll, 0 + y_scroll))
        pygame.draw.line(screen, white, (x_border + x_scroll, 0 + y_scroll), (x_border + x_scroll, y_border + y_scroll))
        pygame.draw.line(screen, white, (x_border + x_scroll, y_border + y_scroll), (0 + x_scroll, y_border + y_scroll))
        pygame.draw.line(screen, white, (0 + x_scroll, y_border + y_scroll), (0 + x_scroll, 0 + y_scroll))

        # Update the display
        pygame.display.flip()

        # Cap the frame rate
        pygame.time.Clock().tick(fps_cap)

        if not pause:
            # print(pa[1],'\n', pa[3], 'pa\n')
            round += 1

        if round == 100 and profile:
            profiler.disable()
            stats = pstats.Stats(profiler)
            stats.sort_stats(pstats.SortKey.TIME)
            stats.print_stats()

        key = pygame.key.get_pressed()
        if key[pygame.K_UP]:
            y_scroll += scrolling_speed
        if key[pygame.K_DOWN]:
            y_scroll -= scrolling_speed
        if key[pygame.K_RIGHT]:
            x_scroll -= scrolling_speed
        if key[pygame.K_LEFT]:
            x_scroll += scrolling_speed

        if key[pygame.K_w]:
            pa[4] -= 9 * 10 ** 18
        if key[pygame.K_s]:
            pa[4] += 9 * 10 ** 18
        if key[pygame.K_a]:
            pa[3] -= 9 * 10 ** 18
        if key[pygame.K_d]:
            pa[3] += 9 * 10 ** 18
        if key[pygame.K_i]:
            pa[5] -= 9 * 10 ** 18
        if key[pygame.K_o]:
            pa[5] += 9 * 10 ** 18

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    pause = not pause
                if event.key == pygame.K_l:
                    Leonard_Jones = not Leonard_Jones

        pygame.display.update()

    pygame.quit()




