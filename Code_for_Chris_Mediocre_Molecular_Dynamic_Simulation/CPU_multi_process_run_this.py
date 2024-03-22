"""Import"""
import numpy as np
import time
import pygame
import cProfile
import pstats
from numba import njit, prange


"""Parameters"""
profile = True

number_of_particles = 4
number_of_types = 4
colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (0, 0, 255)]  # This is rgb.
masses = np.array([2.66 * 10 ** -26, 1.99 * 10 ** -26, 2.32 * 10 ** -26, 0.167 * 10 ** -26])  # Kilograms. This corresponds to the above color list. Oxygen, Carbon, Nitrogen, Hydrogen.

x_border = 700  # For example 100 means that it goes from x 0 to x 100.
y_border = 700
z_border = 700

minimum_starting_x = 300
minimum_starting_y = 300
minimum_starting_z = 300

maximum_starting_x = 500
maximum_starting_y = 500
maximum_starting_z = 300

time_step = 10 ** -18  # Time_step defines how important the momentary force is.

min_sigma = 3 * 10 ** -10  # Sigma is the finite distance at which the inter-particle potential is zero.
max_sigma = 5 * 10 ** -10

min_epsilon = 1000  # Epsilon is the well-depth.
max_epsilon = 5000

force_range = 1000  # Pixels not meters.
force_cap = 10 ** -25

newton_third_law = True

Van_der_Waals = False
attract = True

have_borders = True
have_energy_loss_when_border_collision = True
border_collision_energy_loss = 0.2  # 0.2 is 20% of its kinetic energy
border_collision_percentage_remaining_velocity = 1 - border_collision_energy_loss ** 0.5

z_multiplier = 0.02

pixel_size = 10 ** -11  # Meters

scrolling_speed = 1

fps_cap = 10000000000

"""Functions"""


'''Force Function'''
@njit
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

    force_mistake = F > force_cap

    F[force_mistake] = force_cap

    return F


'''Split Vectors to its x,y,z vectors Function'''
@njit
def a_to_ax_ay_az(a, sin_phi, cos_phi, sin_theta, cos_theta):  # These are all arrays
    ay = sin_theta * a

    a_projection_to_x_z_plane = cos_theta * a

    ax = cos_phi * a_projection_to_x_z_plane

    az = sin_phi * a_projection_to_x_z_plane

    return ax, ay, az


'''Position after all border collisions Function'''
@njit
def collide_until_within_borders(positions, velocities, right_border, have_energy_loss_when_border_collision, border_collision_percentage_remaining_velocity):
    if positions.size == 0:
        return positions, velocities

    negative_positions = positions < 0

    positions = np.abs(positions)

    collisions = positions // right_border

    abs_velocities = np.abs(velocities)

    # Here we try and simulate all the collision that the particle would have had in the time step in case it travelled a distance larger than the size of the allowed area.


    '''Reducing the velocity and position according to border collision energy loss.'''
    if have_energy_loss_when_border_collision:

        num_of_collisions = collisions.astype(np.int32)
        num_of_collisions[negative_positions] += 1

        for i in range(np.max(num_of_collisions)):
            collide = num_of_collisions > i

            # Getting the new reduced velocities. Ek' = Ek - Ek * loss => V' = V - V * loss ** 0.5
            reduced_velocities = abs_velocities.astype(np.float64)
            reduced_velocities[collide] = abs_velocities[collide] * border_collision_percentage_remaining_velocity  # V' = V * (1 - loss ** 0.5)

            positions[negative_positions & collide] = border_collision_percentage_remaining_velocity * positions[negative_positions & collide]  # time = V'/d' = V/d => d' = V'* d/V

            positions[~negative_positions & collide] = border_collision_percentage_remaining_velocity * (positions[~negative_positions & collide] - right_border) + right_border

            abs_velocities = reduced_velocities

        # We define the collisions again because with the reducing of the kinetic energy maybe not as many border collisions happened.
        collisions = positions // right_border


    '''Positions according to which border they hit last in the time step that passed'''
    hit_the_right_border_last = collisions % 2 != 0

    positions[hit_the_right_border_last] = right_border - (positions[hit_the_right_border_last] - right_border * collisions[hit_the_right_border_last])

    positions[~hit_the_right_border_last] = positions[~hit_the_right_border_last] - right_border * collisions[~hit_the_right_border_last]


    '''Changing the velocities'''
    velocity_directions = np.ones(len(positions))

    velocity_directions[hit_the_right_border_last] = -1  # Negative direction is towards the left.

    velocities = abs_velocities * velocity_directions

    return positions, velocities


"""Random Arrays"""


'''Random Positions Particle Array'''
def random_positions_particle_array(number_of_particles, number_of_types, x_border, y_border, z_border, minimum_starting_x, minimum_starting_y, minimum_starting_z, maximum_starting_x, maximum_starting_y, maximum_starting_z):
    x_coordinates = np.random.uniform(minimum_starting_x, maximum_starting_x, number_of_particles)
    y_coordinates = np.random.uniform(minimum_starting_y, maximum_starting_y, number_of_particles)
    z_coordinates = np.random.uniform(minimum_starting_z, maximum_starting_z, number_of_particles)

    x_velocity = np.zeros(number_of_particles)  # We are starting with velocity of zero.
    y_velocity = np.zeros(number_of_particles)
    z_velocity = np.zeros(number_of_particles)

    colors_line = np.zeros(number_of_particles)

    # Here we assign colors to the particles where 0 is the first color in the colors list, 1 is the second and so on...
    slice_length = number_of_particles // number_of_types
    for i in range(1, number_of_types):
        colors_line[i * slice_length: (i + 1) * slice_length] = i

    particle_array = np.array([
        x_coordinates,
        y_coordinates,
        z_coordinates,
        x_velocity,
        y_velocity,
        z_velocity,
        colors_line
    ])

    return particle_array


pa = random_positions_particle_array(number_of_particles, number_of_types, x_border, y_border, z_border, minimum_starting_x, minimum_starting_y, minimum_starting_z, maximum_starting_x, maximum_starting_y, maximum_starting_z)
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
@njit(parallel=True)
def move(particle_array, sigma_array, epsilon_array, time_step, force_range, have_borders, have_energy_loss_when_border_collision, border_collision_percentage_remaining_velocity, masses, pixel_size, force_cap, Van_der_Waals, attract):

    copy_of_particle_array = particle_array.copy()

    c = 0
    for i in prange(len(copy_of_particle_array[0])):
        position = copy_of_particle_array[:, i]
        x = position[0]
        y = position[1]
        z = position[2]
        color = position[6]

        mass = masses[int(color)]


        '''Particles in range'''
        particles_in_range_mask = (np.abs(copy_of_particle_array[0] - x) <= force_range) & (np.abs(copy_of_particle_array[1] - y) <= force_range) & (np.abs(copy_of_particle_array[2] - z) <= force_range)

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


        '''Calculating ax, ay, az from the sum_forces F we just calculated'''
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

        acceleration = sum_forces / mass

        a_xyz = a_to_ax_ay_az(acceleration, sin_phi, cos_phi, sin_theta, cos_theta)


        '''Changing Velocities'''
        # x increases towards the right.
        # y increases towards up.
        # z increases towards in.

        ax = np.sign(init_dx) * a_xyz[0]
        ay = np.sign(init_dy) * a_xyz[1]
        az = np.sign(init_dz) * a_xyz[2]
        # For the cases where some of dx, dy, dz are 0 np.sign() returns 0.
        # Negative means attraction and positive means repulsion just like the forces.


        position[3] += np.sum(ax)
        position[4] += np.sum(ay)
        position[5] += np.sum(az)


        '''Changing Positions'''
        particle_array[0, c] += position[3]
        particle_array[1, c] += position[4]
        particle_array[2, c] += position[5]

        c += 1

    '''Borders'''
    if have_borders:
        x_not_within_borders = (particle_array[0] > x_border) | (particle_array[0] < 0)
        # In case the new position is two or more times as fat the x size of the box we need to calculate for multiple collisions with both borders.
        x_collisions = collide_until_within_borders(particle_array[0, x_not_within_borders], particle_array[3, x_not_within_borders], x_border, have_energy_loss_when_border_collision, border_collision_percentage_remaining_velocity)

        particle_array[0, x_not_within_borders] = x_collisions[0]
        particle_array[3, x_not_within_borders] = x_collisions[1]

        y_not_within_borders = (particle_array[1] > y_border) | (particle_array[1] < 0)
        y_collisions = collide_until_within_borders(particle_array[1, y_not_within_borders], particle_array[4, y_not_within_borders], y_border, have_energy_loss_when_border_collision, border_collision_percentage_remaining_velocity)

        particle_array[1, y_not_within_borders] = y_collisions[0]
        particle_array[4, y_not_within_borders] = y_collisions[1]

        z_not_within_borders = (particle_array[2] > z_border) | (particle_array[2] < 0)
        z_collisions = collide_until_within_borders(particle_array[2, z_not_within_borders], particle_array[5, z_not_within_borders], z_border, have_energy_loss_when_border_collision, border_collision_percentage_remaining_velocity)

        particle_array[2, z_not_within_borders] = z_collisions[0]
        particle_array[5, z_not_within_borders] = z_collisions[1]

    return(particle_array)


"""Imaging"""

# Initialize Pygame
pygame.init()

# Set up display
width, height = 1000, 750
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Crappy MD Simulator by Chris")

# Colors
white = (255, 255, 255)
black = (0, 0, 0)

"""Main game loop"""
if __name__ == "__main__":
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
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    pause = not pause

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
            pa[4] -= 1 * pixel_size/pixel_size
        if key[pygame.K_s]:
            pa[4] += 1 * pixel_size/pixel_size
        if key[pygame.K_a]:
            pa[3] -= 1 * pixel_size/pixel_size
        if key[pygame.K_d]:
            pa[3] += 1 * pixel_size/pixel_size
        if key[pygame.K_i]:
            pa[5] -= 1 * pixel_size/pixel_size
        if key[pygame.K_o]:
            pa[5] += 1 * pixel_size/pixel_size

        if round % 100 == 0:
            print(round, 'round\n')
            print('Speed is', (time.time() - start_time) / 100, 'seconds per round.')

            Ukin = 0.5 * masses[pa[6].astype(int)] * (pa[3] ** 2 + pa[4] ** 2 + pa[5] ** 2)
            print('Total Kinetic energy is', np.sum(Ukin), 'Joules\n')
            start_time = time.time()

        if not pause:
            pa = move(pa, ea, sa, time_step, force_range, have_borders, have_energy_loss_when_border_collision, border_collision_percentage_remaining_velocity, masses, pixel_size, force_cap, Van_der_Waals, attract)

        # Clear the screen
        screen.fill(black)

        # Get the indices that would sort the array based on the third row (z position)
        sorted_indices = np.argsort(pa[2])

        # Use the sorted indices to rearrange the array
        pa = pa[:, sorted_indices]

        # Draw points on the screen
        print(pa[:6], 'pa\n')
        for x, y, z, color in zip(pa[0], pa[1], pa[2], pa[6]):
            pygame.draw.circle(screen, colors[int(color)], (x + x_scroll, y + y_scroll), z * z_multiplier)

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
            round += 1

        if round == 100 and profile:
            profiler.disable()
            stats = pstats.Stats(profiler)
            stats.sort_stats(pstats.SortKey.TIME)
            stats.print_stats()



