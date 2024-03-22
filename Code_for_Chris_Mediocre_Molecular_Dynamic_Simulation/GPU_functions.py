import torch


"""Functions"""


'''Force Function'''
def force_function(distances, time_step, A, B, pixel_size, force_cap, Van_der_Waals, attract, device):  # Negative means attraction and positive means repulsion.
    # I use the force calculate from the potential energy according to the Lennard-Jones potential.
    # The force between two atoms in a Lennard-Jones potential can be obtained by taking the negative gradient of the potential energy with respect to the separation distance.
    distances_resized = distances * pixel_size

    # 24 / (6 * 10 ** 23) == 4e-23

    # This is the equation for the Leonard-Jones potential. We simplify it for computational speed:
    # F = 4e-23 * epsilon / distances_resized * (2 * (sigma / (distances_resized)) ** 12 - (sigma / (distances_resized)) ** 6)

    # A = 4e-23 * epsilon
    # B = sigma ** 6

    F = torch.zeros(len(distances_resized[:, 0]), len(distances_resized[0])).to(device)

    if Van_der_Waals:
        F.add_(A / distances_resized * (2 * B ** 2 / distances_resized ** 12 - B / distances_resized ** 6))

    if attract:
        F.add_(-distances_resized)

    F.mul_(time_step)

    force_mistake = F > force_cap

    F[force_mistake] = force_cap

    return F


'''Split Vectors to its x,y,z vectors Function'''
def a_to_ax_ay_az(a, sin_phi, cos_phi, sin_theta, cos_theta):  # These are all arrays
    ay = sin_theta * a

    a_projection_to_x_z_plane = cos_theta * a

    ax = cos_phi * a_projection_to_x_z_plane

    az = sin_phi * a_projection_to_x_z_plane

    return ax, ay, az


'''Collide until within borders Function'''
def collide_until_within_borders(positions, velocities, right_border, have_energy_loss_when_border_collision, border_collision_percentage_remaining_velocity):
    if positions.numel() == 0:
        return positions, velocities

    negative_positions = positions < 0

    positions = torch.abs(positions)

    collisions = positions // right_border

    abs_velocities = torch.abs(velocities)

    # Here we try and simulate all the collision that the particle would have had in the time step in case it travelled a distance larger than the size of the allowed area.


    '''Reducing the velocity and position according to border collision energy loss.'''
    if have_energy_loss_when_border_collision:

        num_of_collisions = collisions.to(torch.int)
        num_of_collisions[negative_positions].add_(1)

        for i in range(torch.max(num_of_collisions)):
            collide = num_of_collisions > i

            # Getting the new reduced velocities. Ek' = Ek - Ek * loss => V' = V - V * loss ** 0.5
            reduced_velocities = abs_velocities
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
    velocity_directions = torch.ones(len(positions))

    velocity_directions[hit_the_right_border_last] = -1  # Negative direction is towards the left.

    velocities = abs_velocities * velocity_directions

    return positions, velocities
