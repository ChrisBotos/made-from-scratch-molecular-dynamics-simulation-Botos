"""Import"""
import numpy as np
import time
import pygame
import cProfile
import pstats
import random


"""Parameters"""
profile = True

number_of_particles = np.array([0, 2, 0, 4])
number_of_types = 4
colors = np.array([(0, 255, 255), (255, 0, 0), (0, 0, 255), (0, 255, 0)])  # This is rgb.
masses = np.array([0.17 * 10 ** -26, 1.99 * 10 ** -26, 2.33 * 10 ** -26, 2.66 * 10 ** -26])  # Kilograms. This corresponds to the above color list. Hydrogen, Carbon, Nitrogen, Oxygen.
charges = np.array([0., -1., 0., 0.])  # Charges for free atoms not in molecules

x_border = 700  # For example 100 means that it goes from x 0 to x 100.
y_border = 700
z_border = 700

minimum_starting_x = 300
minimum_starting_y = 300
minimum_starting_z = 300

maximum_starting_x = 500
maximum_starting_y = 500
maximum_starting_z = 300

time_step = 10 ** -23  # Time_step defines how important the momentary force is.

min_sigma = 3 * 10 ** -10  # In Meters. Sigma is the finite distance at which the inter-particle potential is zero.
max_sigma = 5 * 10 ** -10

min_epsilon = 1 * 10 ** -19  # In Joules. Epsilon is the well-depth.
max_epsilon = 5 * 10 ** -19

force_range = 1000  # Pixels not meters.
force_cap = 10 ** -25  # Newtons.

newton_third_law = True

Van_der_Waals = True
attract = False

have_borders = True
have_energy_loss_when_border_collision = False
border_collision_energy_loss = 0.2  # 0.2 is 20% of its kinetic energy
border_collision_percentage_remaining_velocity = 1 - border_collision_energy_loss ** 0.5

have_molecules = True

array_of_molecule_types_atoms = np.array([[0, 1, 0, 2]])  # The numbers correspond to the amount of atoms of that index inside the molecule.

# We put one of the atoms in the geometric shape as 0. Then we give the x, y and z distances of any other atom from it in lists.
# We put them in order from lightest to heavier like in the masses and colors lists.
array_of_molecule_types_shape = np.array([[[0, -11.63 * 10 ** -10, 11.63 * 10 ** -10],
                                           [0, 0, 0],
                                           [0, 0, 0]]

                                          ])

array_of_molecule_types_charge = np.array([[0, 0, 0]])  # The numbers correspond to the charge of atoms of that index inside the molecule.

z_multiplier = 0.02

pixel_size = 5 * 10 ** -11  # Meters
array_of_molecule_types_shape /= pixel_size

scrolling_speed = 1

fps_cap = 10000000000000


"""Functions"""


'''Force Function'''
def force_function(distances, time_step, A, B, pixel_size, force_cap, Van_der_Waals, attract):  # Negative means attraction and positive means repulsion.
    # I use the force calculate from the potential energy according to the Lennard-Jones potential.
    # The force between two atoms in a Lennard-Jones potential can be obtained by taking the negative gradient of the potential energy with respect to the separation distance.
    distances_resized = distances * pixel_size

    # 24 / (6 * 10 ** 23) == 4e-23

    # This is the equation for the Leonard-Jones potential. We simplify it for computational speed:
    # F = 4e-23 * epsilon / distances_resized * (2 * (sigma / (distances_resized)) ** 12 - (sigma / (distances_resized)) ** 6)

    # A = 4e-23 * epsilon
    # B = sigma ** 6

    F = np.zeros(len(distances_resized))

    if Van_der_Waals:
        F += A / distances_resized * (2 * B ** 2 / distances_resized ** 12 - B / distances_resized ** 6)

    if attract:
        F += -distances_resized

    F *= time_step

    force_mistake = np.abs(F) > force_cap

    F[force_mistake] = np.sign(F[force_mistake]) * force_cap

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

    out_of_borders_positions = positions[not_within_borders_mask]
    out_of_borders_velocities = velocities[not_within_borders_mask]
    out_of_borders_molecules = molecules[not_within_borders_mask]

    negative_positions = out_of_borders_positions < 0

    out_of_borders_positions = np.abs(out_of_borders_positions)

    collisions = out_of_borders_positions // right_border

    abs_out_of_borders_velocities = np.abs(out_of_borders_velocities)

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

            out_of_borders_positions[negative_positions & collide] = border_collision_percentage_remaining_velocity * out_of_borders_positions[negative_positions & collide]  # time = V'/d' = V/d => d' = V'* d/V

            out_of_borders_positions[~negative_positions & collide] = border_collision_percentage_remaining_velocity * (out_of_borders_positions[~negative_positions & collide] - right_border) + right_border

            abs_out_of_borders_velocities = reduced_out_of_borders_velocities

        # We define the collisions again because with the reducing of the kinetic energy maybe not as many border collisions happened.
        collisions = out_of_borders_positions // right_border


    '''Positions according to which border they hit last in the time step that passed'''
    hit_the_right_border_last = collisions % 2 != 0

    copy_not_within_borders_mask = not_within_borders_mask.copy()

    if have_molecules:
        '''Finding how much the positions of out of borders atoms changed and then making that change to all the atoms in the same molecule'''
        copy_not_within_borders_mask[not_within_borders_mask] = hit_the_right_border_last
        difference_right_last = positions[copy_not_within_borders_mask] - right_border - (out_of_borders_positions[hit_the_right_border_last] - right_border * collisions[hit_the_right_border_last])

        copy_not_within_borders_mask[not_within_borders_mask] = ~hit_the_right_border_last
        difference_left_last = positions[copy_not_within_borders_mask] - out_of_borders_positions[~hit_the_right_border_last] - right_border * collisions[~hit_the_right_border_last]

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
        positions[copy_not_within_borders_mask] = right_border - (out_of_borders_positions[hit_the_right_border_last] - right_border * collisions[hit_the_right_border_last])

        copy_not_within_borders_mask[not_within_borders_mask] = ~hit_the_right_border_last
        positions[copy_not_within_borders_mask] = out_of_borders_positions[~hit_the_right_border_last] - right_border * collisions[~hit_the_right_border_last]


    '''Changing the velocities'''
    velocity_directions = np.ones(len(out_of_borders_positions))

    velocity_directions[hit_the_right_border_last] = -1  # Negative direction is towards the left.

    if have_molecules:

        c = 0
        for i in out_of_borders_molecules:

            if i == 0:
                continue  # It will change below.

            same_molecule = molecules == i
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
def get_forces(position, copy_of_particle_array, sigma_array, epsilon_array, time_step, force_range, masses, pixel_size, force_cap, Van_der_Waals, attract, have_molecules):

        x = position[0]
        y = position[1]
        z = position[2]
        color = position[6]
        molecule = position[7]

        mass = masses[int(color)]


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


        '''Calculating the forces acted on our particle'''
        # We need to use the sigma_array and get the sigmas for the appropriate color of each particle in range.
        sigmas_of_the_in_range_particles = sigma_array[int(color), np.asarray(particles_in_range[6], dtype=np.int32)][not_on_top_mask]  # The rows of the sigma array correspond to the force receivers and the columns to the force sources.

        # We need to use the epsilon_array and get the epsilons for the appropriate color of each particle in range.
        epsilons_of_the_in_range_particles = epsilon_array[int(color), np.asarray(particles_in_range[6], dtype=np.int32)][not_on_top_mask]  # The rows of the epsilon array correspond to the force receivers and the columns to the force sources.

        sum_forces = force_function(distances, time_step, sigmas_of_the_in_range_particles, epsilons_of_the_in_range_particles, pixel_size, force_cap, Van_der_Waals, attract)


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
            print(atom_type, 'type\n')

            molecule_number = 1
            molecule_type = 0

            counter = 0
            while counter < number_of_particles[atom_type]:

                if array_of_molecule_types_atoms[molecule_type][atom_type] == 0:
                    break

                '''Numbering the atoms'''
                for atom_of_this_molecule in range(array_of_molecule_types_atoms[molecule_type][atom_type]):

                    belongs_to_molecule[current_atom_index] = molecule_number
                    print(current_atom_index, 'idx\n')

                    current_atom_index += 1

                    counter += 1

                molecule_number += 1

                molecule_type += 1

                if molecule_type >= len(array_of_molecule_types_atoms):
                    molecule_type = 0

    # if have_molecules:
    #
    #     counter = 0
    #     molecule_number = 1
    #     not_first_molecule_type = False
    #
    #     storage_array = np.full(np.max(number_of_particles), number_of_types, -1)
    #
    #     arr =
    #
    #     for molecule_type in array_of_molecule_types_atoms:
    #         number_of_molecules_of_this_type = array_of_molecule_types_percentages[counter] * sum(number_of_particles)
    #
    #         atom_type_idx = array_of_molecule_types_percentages - np.cumsum(array_of_molecule_types_percentages[counter])
    #         for atom_type in molecule_type:
    #
    #             row_number = 0
    #             for _ in number_of_molecules_of_this_type:
    #
    #                 for atom in range(atom_type):
    #
    #                     storage_array[row_number, atom_type_idx] = molecule_number
    #
    #                     molecule_number += 1
    #
    #         counter += 1



            # atoms_in_moleculesnp.sum(array_of_molecule_types_atoms.T[atom_type])




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
                    print(molecule_center_x, 'mc\n')

                    if molecule_type != -1:
                        molecule_type += 1

                    else:
                        molecule_type = 0  # Accounting for the first run.

                    if molecule_type >= len(array_of_molecule_types_atoms):
                        molecule_type = 0

                print(molecule_center_x + array_of_molecule_types_shape[molecule_type][0][atom_in_molecule], 'mt\n')
                x_coordinates[atom] = molecule_center_x + array_of_molecule_types_shape[molecule_type][0][atom_in_molecule]
                y_coordinates[atom] = molecule_center_y + array_of_molecule_types_shape[molecule_type][1][atom_in_molecule]
                z_coordinates[atom] = molecule_center_z + array_of_molecule_types_shape[molecule_type][2][atom_in_molecule]
                print(x_coordinates, y_coordinates, 'xy\n')

                atom_in_molecule += 1

    else:
        x_coordinates = np.random.uniform(minimum_starting_x, maximum_starting_x, sum(number_of_particles))
        y_coordinates = np.random.uniform(minimum_starting_y, maximum_starting_y, sum(number_of_particles))
        z_coordinates = np.random.uniform(minimum_starting_z, maximum_starting_z, sum(number_of_particles))

    # '''Charges of Molecule's atoms'''
    # molecule_type = 0
    # counter = 0
    # for atom_in_molecule in belongs_to_molecule:
    #
    #     if atom_in_molecule != 0:
    #
    #         charges_line[counter] = array_of_molecule_types_charge[molecule_type][colors_line[counter]]




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
def random_sigma_array(number_of_types, min_sigma, max_sigma, newton_third_law, Van_der_Waals):
    if number_of_types == 0:
        return np.array([[]])

    sigma_array = np.random.uniform(min_sigma, max_sigma, number_of_types)
    for _ in range(number_of_types - 1):
        sigma_array = np.vstack((sigma_array, np.random.uniform(min_sigma, max_sigma, number_of_types)))

    if newton_third_law:
        # Make the elements below the main diagonal the same as above.
        sigma_array[np.tril_indices(sigma_array.shape[0], k=-1)] = sigma_array.T[np.tril_indices(sigma_array.shape[0], k=-1)]

    '''For computational speed do the math on sigma outside the main loop.'''
    if Van_der_Waals:
        B = sigma_array ** 6
    else:
        B = sigma_array

    return sigma_array, B


sa = random_sigma_array(number_of_types, min_sigma, max_sigma, newton_third_law, Van_der_Waals)[1]
print(sa, 'sa\n')


'''Random Epsilon Array'''
def random_epsilon_array(number_of_types, min_epsilon, max_epsilon, newton_third_law, Van_der_Waals):
    if number_of_types == 0:
        return np.array([[]])

    epsilon_array = np.random.uniform(min_epsilon, max_epsilon, number_of_types)
    for _ in range(number_of_types - 1):
        epsilon_array = np.vstack((epsilon_array, np.random.uniform(min_epsilon, max_epsilon, number_of_types)))

    if newton_third_law:
        # Make the elements below the main diagonal the same as above.
        epsilon_array[np.tril_indices(epsilon_array.shape[0], k=-1)] = epsilon_array.T[np.tril_indices(epsilon_array.shape[0], k=-1)]

    '''For computational speed do the math on epsilon outside the main loop.'''
    if Van_der_Waals:
        A = epsilon_array * 4 * 10 ** -23
    else:
        A = epsilon_array

    return epsilon_array, A


ea = random_epsilon_array(number_of_types, min_epsilon, max_epsilon, newton_third_law, Van_der_Waals)[1]
print(ea, 'ea\n')


"""Moving the particles"""
def move(particle_array, sigma_array, epsilon_array, time_step, force_range, have_borders, have_energy_loss_when_border_collision, border_collision_percentage_remaining_velocity, masses, pixel_size, force_cap, Van_der_Waals, attract):

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

        F_xyz = get_forces(particle, copy_of_particle_array, sigma_array, epsilon_array, time_step, force_range, masses, pixel_size, force_cap, Van_der_Waals, attract, have_molecules)


        '''Changing Velocities'''
        # x increases towards the right.
        # y increases towards down.
        # z increases towards out.

        if have_molecules and molecule != 0:
            Fx += F_xyz[0]
            Fy += F_xyz[1]
            Fz += F_xyz[2]

            if molecule == previous_molecule:
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

                    particle_array[3, current_atom] += ax
                    particle_array[4, current_atom] += ay
                    particle_array[5, current_atom] += az

                    '''Changing Positions'''
                    particle_array[0, current_atom] += particle_array[3, current_atom]
                    particle_array[1, current_atom] += particle_array[4, current_atom]
                    particle_array[2, current_atom] += particle_array[5, current_atom]


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

            particle_array[3, c] += ax
            particle_array[4, c] += ay
            particle_array[5, c] += az

            '''Changing Positions'''
            particle_array[0, c] += particle_array[3, c]
            particle_array[1, c] += particle_array[4, c]
            particle_array[2, c] += particle_array[5, c]

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

            particle_array[3, current_atom] += ax
            particle_array[4, current_atom] += ay
            particle_array[5, current_atom] += az

            '''Changing Positions'''
            particle_array[0, current_atom] += particle_array[3, current_atom]
            particle_array[1, current_atom] += particle_array[4, current_atom]
            particle_array[2, current_atom] += particle_array[5, current_atom]


    '''Borders'''
    if have_borders:
        # molecules_not_within_borders = np.array(list(set(particle_array[7, x_not_within_borders])))
        #  = np.isin(particle_array[7], molecules_not_within_borders)  # TODO make it so the whole molecule bounces back
        # In case the new position is two or more times as fat the x size of the box we need to calculate for multiple collisions with both borders.
        x_collisions = collide_until_within_borders(particle_array[0], particle_array[3], particle_array[7], x_border, have_energy_loss_when_border_collision, border_collision_percentage_remaining_velocity, have_molecules)

        particle_array[0] = x_collisions[0]
        particle_array[3] = x_collisions[1]

        y_collisions = collide_until_within_borders(particle_array[1], particle_array[4], particle_array[7], y_border, have_energy_loss_when_border_collision, border_collision_percentage_remaining_velocity, have_molecules)

        particle_array[1] = y_collisions[0]
        particle_array[4] = y_collisions[1]

        z_collisions = collide_until_within_borders(particle_array[2], particle_array[5], particle_array[7], z_border, have_energy_loss_when_border_collision, border_collision_percentage_remaining_velocity, have_molecules)

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
            pa = move(pa, ea, sa, time_step, force_range, have_borders, have_energy_loss_when_border_collision, border_collision_percentage_remaining_velocity, masses, pixel_size, force_cap, Van_der_Waals, attract)

        # Get the indices that would sort the array based on the third row (z position)
        sorted_indices = np.argsort(pa[2])

        # Use the sorted indices to rearrange the array
        z_sorted_pa = pa[:, sorted_indices].copy()

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
            # print(pa[:6], 'pa\n')
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
            pa[4] -= 1 * pixel_size / pixel_size
        if key[pygame.K_s]:
            pa[4] += 1 * pixel_size / pixel_size
        if key[pygame.K_a]:
            pa[3] -= 1 * pixel_size / pixel_size
        if key[pygame.K_d]:
            pa[3] += 1 * pixel_size / pixel_size
        if key[pygame.K_i]:
            pa[5] -= 1 * pixel_size / pixel_size
        if key[pygame.K_o]:
            pa[5] += 1 * pixel_size / pixel_size

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    pause = not pause
                if event.key == pygame.K_v:
                    Van_der_Waals = not Van_der_Waals

        pygame.display.update()

    pygame.quit()




