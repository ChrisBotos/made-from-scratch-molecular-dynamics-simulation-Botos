import torch
from GPU_functions import force_function, a_to_ax_ay_az


def calculate_forces_and_change_positions(particle_array, sigma_array, epsilon_array, time_step, force_range, masses, pixel_size, force_cap, Van_der_Waals, attract, device):

    expanded_particle_array = particle_array.unsqueeze(0).unsqueeze(1).expand(len(particle_array[0]), len(particle_array[0]), -1).clone()  # Cloning them just to be safe.
    expanded_sigma_array = sigma_array.unsqueeze(0).unsqueeze(1).expand(len(particle_array[0]), len(particle_array[0]), -1).clone()
    expanded_epsilon_array = epsilon_array.unsqueeze(0).unsqueeze(1).expand(len(particle_array[0]), len(particle_array[0]), -1).clone()

    x = particle_array[0]
    y = particle_array[1]
    z = particle_array[2]
    color = particle_array[6].to(torch.int)

    all_masses = color.clone()
    c = 0
    for i in color:
        all_masses[c] = masses[i.item()]
        c += 1


    '''Particles in range'''
    particles_in_range_mask = (torch.abs(expanded_particle_array[:, 0] - x) <= force_range) & (torch.abs(expanded_particle_array[:, 1] - y) <= force_range) & (torch.abs(expanded_particle_array[:, 2] - z) <= force_range)

    particles_in_range = expanded_particle_array[:, :, particles_in_range_mask]


    '''Calculating their distances from our particle'''
    # This will also catch our particle too, but the distance will be 0 and so the F will be 0 anyway.
    # Distance is calculated by (dx^2 + dy^2 + dz^2) ^ 1/2.
    init_dx = x - particles_in_range[:, 0]
    init_dy = y - particles_in_range[:, 1]
    init_dz = z - particles_in_range[:, 2]
    dx = torch.abs(init_dx)
    dy = torch.abs(init_dy)
    dz = torch.abs(init_dz)
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
    # We need to use the epsilon_array and get the epsilons for the appropriate color of each particle in range.
    epsilons_of_the_in_range_particles = expanded_epsilon_array[:, color, particles_in_range[6].to(torch.int)][not_on_top_mask]  # The rows of the epsilon array correspond to the force receivers and the columns to the force sources.

    # We need to use the sigma_array and get the sigmas for the appropriate color of each particle in range.
    sigmas_of_the_in_range_particles = expanded_sigma_array[:, color, particles_in_range[6].to(torch.int)][not_on_top_mask]  # The rows of the sigma array correspond to the force receivers and the columns to the force sources.

    sum_forces = force_function(distances, time_step, sigmas_of_the_in_range_particles, epsilons_of_the_in_range_particles, pixel_size, force_cap, Van_der_Waals, attract, device)


    '''Calculating ax, ay, az from the sum_forces F we just calculated'''
    not_zero_dx_mask = dx != 0
    not_zero_dy_mask = dy != 0
    not_zero_dz_mask = dz != 0

    number_of_in_range_particles = len(distances[0])

    d_projection_to_x_z_plane = (dx ** 2 + dz ** 2) ** 0.5

    sin_phi = torch.zeros(number_of_in_range_particles, number_of_in_range_particles).to(device)
    sin_phi[not_zero_dz_mask] = dz[not_zero_dz_mask] / d_projection_to_x_z_plane[not_zero_dz_mask]  # phi is the x to d_projection_to_x_z_plane angle.

    cos_phi = torch.zeros(number_of_in_range_particles, number_of_in_range_particles).to(device)
    cos_phi[not_zero_dx_mask] = dx[not_zero_dx_mask] / d_projection_to_x_z_plane[not_zero_dx_mask]

    sin_theta = torch.zeros(number_of_in_range_particles, number_of_in_range_particles).to(device)
    sin_theta[not_zero_dy_mask] = dy[not_zero_dy_mask] / distances[not_zero_dy_mask]  # theta is the d_projection_to_x_z_plane to distances angle.

    cos_theta = d_projection_to_x_z_plane / distances

    print(sum_forces, 'sum\n', all_masses, 'massaz\n')
    acceleration = sum_forces / all_masses

    a_xyz = a_to_ax_ay_az(acceleration, sin_phi, cos_phi, sin_theta, cos_theta)


    '''Changing Velocities'''
    # x increases towards the right.
    # y increases towards up.
    # z increases towards in.

    ax = torch.sign(init_dx) * a_xyz[0]
    ay = torch.sign(init_dy) * a_xyz[1]
    az = torch.sign(init_dz) * a_xyz[2]
    # For the cases where some of dx, dy, dz are 0 np.sign() returns 0.
    # Negative means attraction and positive means repulsion just like the forces.

#     position[3].add_(torch.sum(ax))
#     position[4].add_(torch.sum(ay))
#     position[5].add_(torch.sum(az))
#
#
#     '''Changing Positions'''
#     particle_array[0, c].add_(position[3])
#     particle_array[1, c].add_(position[4])
#     particle_array[2, c].add_(position[5])
#
#     c += 1
#
# return particle_array
