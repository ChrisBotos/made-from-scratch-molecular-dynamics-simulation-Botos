"""Import"""
import time
import pygame
import torch
import cProfile
import pstats

from PyTorch_function_calculate_forces_and_change_positions import calculate_forces_and_change_positions
from GPU_functions import collide_until_within_borders


"""Parameters"""
profile = True

device = "cpu"

number_of_particles = 4
number_of_types = 4
colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (0, 0, 255)]  # This is rgb.
masses = torch.tensor([2.66 * 10 ** -26, 1.99 * 10 ** -26, 2.32 * 10 ** -26, 0.167 * 10 ** -26]).to(device)  # Kilograms. This corresponds to the above color list. Oxygen, Carbon, Nitrogen, Hydrogen.

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

min_sigma = 3 * 10 ** -10  # In Meters. Sigma is the finite distance at which the inter-particle potential is zero.
max_sigma = 5 * 10 ** -10

min_epsilon = 1 * 10 ** -19  # In Joules. Epsilon is the well-depth.
max_epsilon = 5 * 10 ** -19

force_range = 1000  # Pixels not meters.
force_cap = 10 ** -25

newton_third_law = True

Van_der_Waals = False
attract = True

have_borders = True
have_energy_loss_when_border_collision = True
border_collision_energy_loss = 0.5  # 0.2 is 20% of its kinetic energy
border_collision_percentage_remaining_velocity = 1 - border_collision_energy_loss ** 0.5

z_multiplier = 0.02

pixel_size = 10 ** -11  # Meters

scrolling_speed = 1

fps_cap = 10000000000




"""Random Arrays"""


'''Random Positions Particle Array'''
def random_positions_particle_array(number_of_particles, number_of_types, x_border, y_border, z_border, minimum_starting_x, minimum_starting_y, minimum_starting_z, maximum_starting_x, maximum_starting_y, maximum_starting_z):
    x_coordinates = (maximum_starting_x - minimum_starting_x) * torch.rand(number_of_particles).to(device) + minimum_starting_x
    y_coordinates = (maximum_starting_y - minimum_starting_y) * torch.rand(number_of_particles).to(device) + minimum_starting_y
    z_coordinates = (maximum_starting_z - minimum_starting_z) * torch.rand(number_of_particles).to(device) + minimum_starting_z

    x_velocity = torch.zeros(number_of_particles).to(device)  # We are starting with velocity of zero.
    y_velocity = torch.zeros(number_of_particles).to(device)
    z_velocity = torch.zeros(number_of_particles).to(device)

    colors_line = torch.zeros(number_of_particles).to(device)

    # Here we assign colors to the particles where 0 is the first color in the colors list, 1 is the second and so on...
    slice_length = number_of_particles // number_of_types
    for i in range(1, number_of_types):
        colors_line[i * slice_length: (i + 1) * slice_length] = i

    particle_array = torch.stack([
        x_coordinates,
        y_coordinates,
        z_coordinates,
        x_velocity,
        y_velocity,
        z_velocity,
        colors_line
    ], dim=0).to(device)

    return particle_array


pa = random_positions_particle_array(number_of_particles, number_of_types, x_border, y_border, z_border, minimum_starting_x, minimum_starting_y, minimum_starting_z, maximum_starting_x, maximum_starting_y, maximum_starting_z)
print(pa, 'pa\n')


'''Random Sigma Array'''
def random_sigma_array(number_of_types, min_sigma, max_sigma, newton_third_law, Van_der_Waals):
    if number_of_types == 0:
        return torch.tensor([[]]).to(device)

    sigma_array = (max_sigma - min_sigma) * torch.rand(number_of_types).to(device) + min_sigma
    for _ in range(number_of_types - 1):
        sigma_array = torch.vstack((sigma_array, (max_sigma - min_sigma) * torch.rand(number_of_types).to(device) + min_sigma))

    if newton_third_law:
        # Make the elements below the main diagonal the same as above.
        indices = torch.tril_indices(sigma_array.shape[0], sigma_array.shape[1], offset=-1)
        sigma_array[indices[0], indices[1]] = sigma_array.T[indices[0], indices[1]]

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
        return torch.tensor([[]]).to(device)

    epsilon_array = (max_epsilon - min_epsilon) * torch.rand(number_of_types).to(device) + min_epsilon
    for _ in range(number_of_types - 1):
        epsilon_array = torch.vstack((epsilon_array, (max_epsilon - min_epsilon) * torch.rand(number_of_types).to(device) + min_epsilon))

    if newton_third_law:
        # Make the elements below the main diagonal the same as above.
        indices = torch.tril_indices(epsilon_array.shape[0], epsilon_array.shape[1], offset=-1)
        epsilon_array[indices[0], indices[1]] = epsilon_array.T[indices[0], indices[1]]

    '''For computational speed do the math on epsilon outside the main loop.'''
    if Van_der_Waals:
        A = epsilon_array * 4 * 10 ** -23
    else:
        A = epsilon_array

    return epsilon_array, A


ea = random_epsilon_array(number_of_types, min_epsilon, max_epsilon, newton_third_law, Van_der_Waals)[1]
print(ea, 'ea\n')


"""Moving the particles"""
def move(particle_array, sigma_array, epsilon_array, time_step, force_range, have_borders, have_energy_loss_when_border_collision, border_collision_percentage_remaining_velocity, masses, pixel_size, force_cap, device):

    particle_array = calculate_forces_and_change_positions(particle_array, sigma_array, epsilon_array, time_step, force_range, masses, pixel_size, force_cap, Van_der_Waals, attract, device)

    '''Borders'''
    if have_borders:
        x_not_within_borders = (particle_array[0] > x_border) | (particle_array[0] < 0)
        # In case the new position is two or more times as far as the x size of the box ,we need to calculate for multiple collisions with both borders.
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

    if profile:
        profiler = cProfile.Profile()

        profiler.enable()

    start_time = time.time()
    pause = False
    y_scroll = 0
    x_scroll = 0
    round = 1
    while True:

        if round % 100 == 0:
            print(round, 'round\n')
            print('Speed is', (time.time() - start_time) / 100, 'seconds per round.')

            Ukin = 0.5 * masses[pa[6].to(torch.int)] * (pa[3] ** 2 + pa[4] ** 2 + pa[5] ** 2)
            print('Total Kinetic energy is', torch.sum(Ukin), 'Joules\n')
            start_time = time.time()

        if not pause:
            pa = move(pa, ea, sa, time_step, force_range, have_borders, have_energy_loss_when_border_collision, border_collision_percentage_remaining_velocity, masses, pixel_size, force_cap, device).to("cpu")

        # Clear the screen
        screen.fill(black)

        # Get the indices that would sort the array based on the third row (z position) so that closer to camera is shown above.
        sorted_indices = torch.argsort(pa[2])

        # Use the sorted indices to rearrange the array
        pa = pa[:, sorted_indices]

        # Draw points on the screen
        for x, y, z, color in zip(pa[0], pa[1], pa[2], pa[6]):
            pygame.draw.circle(screen, colors[int(color)], (x.item() + x_scroll, y.item() + y_scroll), z.item() * z_multiplier)

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
            pa = pa.to(device)

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

        if round == 100 and profile:
            profiler.disable()
            stats = pstats.Stats(profiler)
            stats.sort_stats(pstats.SortKey.TIME)
            stats.print_stats()




