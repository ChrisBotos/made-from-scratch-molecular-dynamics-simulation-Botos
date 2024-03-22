import OpenGL.GL.shaders
from random import random as rnd
from pyrr import Vector3, Matrix44
import pygame
from OpenGL_assistance_code import *

"""Parameters"""
camera_speed = 0.1

vertex_src3 = """
#version 330 core
in vec3 aPos; // the positions of the model only

in vec4 aPosIns; // the center positions of all models

in vec3 aScaleIns; // the scale of all models
in vec4 aColorIns;
out vec4 aColorInsV;

uniform vec3 center;

uniform mat4 proj;
uniform mat4 view;

void main()
{
    vec4 scaledModel = vec4(aPos.x * aScaleIns.x, aPos.y * aScaleIns.y, aPos.z * aScaleIns.z, 1.);
    vec4 finalPos = scaledModel + aPosIns + vec4(center, 1.);

    gl_Position = proj * view * finalPos;

    aColorInsV = aColorIns;
}
"""

fragment_src3 = """
#version 330 core

in vec4 aColorInsV; // the color values for all models

out vec4 FragColor;

void main()
{
    FragColor = aColorInsV;
}
"""


def main():
    pygame.init()

    screen_height = 500
    screen_width = 500

    pygame.display.set_mode((screen_height, screen_width), pygame.OPENGL | pygame.DOUBLEBUF)

    GL.glEnable(GL.GL_DEPTH_TEST)

    # shaders compilation start
    v_shader_3 = GL.shaders.compileShader(vertex_src3, GL.GL_VERTEX_SHADER)
    f_shader_3 = GL.shaders.compileShader(fragment_src3, GL.GL_FRAGMENT_SHADER)
    shader3_program = GL.shaders.compileProgram(v_shader_3, f_shader_3)

    objects = VAO()

    plane_buffer = np.ones(16, dtype=np.float32)  # model for the object

    side_size = np.float32(1.0)

    plane_buffer[0] = np.float32(side_size)
    plane_buffer[1] = np.float32(side_size)
    plane_buffer[2] = np.float32(0.0)

    plane_buffer[4] = np.float32(0.0)
    plane_buffer[5] = np.float32(side_size)
    plane_buffer[6] = np.float32(0.0)

    plane_buffer[8] = np.float32(0.0)
    plane_buffer[9] = np.float32(0.0)
    plane_buffer[10] = np.float32(0.0)

    plane_buffer[12] = np.float32(side_size)
    plane_buffer[13] = np.float32(0.0)
    plane_buffer[14] = np.float32(0.0)

    instances_size = 100

    instance_buffer = np.ones(4 * instances_size, dtype=np.float32)
    # it should load the file here for the starting center positions for all objects in this simulation.
    for v in range(0, 4 * instances_size, 4):
        instance_buffer[v] = np.float32(rnd() * 50 - 20 / 2 + 1)
        instance_buffer[v + 1] = np.float32(rnd() * 50 - 20 / 2 + 1)
        instance_buffer[v + 2] = np.float32(rnd() * 50 - 2 * 20 - 1)

    instance_buffer_scale = np.ones(4 * instances_size, dtype=np.float32)
    # it should load the file here for the starting scales for all objects in this simulation.
    for v in range(0, 4 * instances_size, 4):
        instance_buffer_scale[v] = np.float32(1.)
        instance_buffer_scale[v + 1] = np.float32(1.)
        instance_buffer_scale[v + 2] = np.float32(1.)

    # instance_buffer_scale[0] = np.float32(2.)  # TODO change this

    instance_buffer_color = np.ones(4 * instances_size, dtype=np.float32)
    # it should load the file here for the colors for all objects in this simulation.
    for v in range(0, 4 * instances_size, 4):
        instance_buffer_color[v] = np.float32(0.)
        instance_buffer_color[v + 1] = np.float32(0.)
        instance_buffer_color[v + 2] = np.float32(1.)

    instance_buffer_color[0] = np.float32(0.)
    instance_buffer_color[1] = np.float32(1.)
    instance_buffer_color[2] = np.float32(0.)

    # load model in gpu statically
    objects.new_array_buffer(shader3_program, "aPos", 4, plane_buffer, GL.GL_STATIC_DRAW)
    # load center pos for all models statically
    objects.new_instance_buffer(shader3_program, "aPosIns", instances_size, instance_buffer, GL.GL_DYNAMIC_DRAW)
    # load scales for all models statically
    objects.new_instance_buffer(shader3_program, "aScaleIns", instances_size, instance_buffer_scale, GL.GL_STATIC_DRAW)
    # load color for all models statically
    objects.new_instance_buffer(shader3_program, "aColorIns", instances_size, instance_buffer_color, GL.GL_STATIC_DRAW)

    # VAOS constraction code end

    clock = pygame.time.Clock()

    camera_pos = Vector3([0.0, 0.0, 20.])  # Declare camera_pos outside the loop

    round_counter = 1
    while True:
        clock.tick()

        # use background method before any VAO display method
        background(0., 0., 0., 0.)

        XY = pygame.mouse.get_pos()
        # XY[0] = x
        # XY[1] = y
        translate_3f([shader3_program], "center", camera_pos.x, camera_pos.y, camera_pos.z)

        # example camera code is below start
        camera_front = Vector3([np.sin(2 * np.pi * XY[0] / screen_width), 0.0, np.cos(2 * np.pi * XY[0] / screen_width)])  # ([0.0,0.0,-1.0])

        set_camera_2_Matrix4fv([shader3_program],
                               "proj",
                               "view",
                               screen_height,
                               screen_width,
                               camera_pos,
                               camera_front,
                               Vector3([0.0, 1.0, 0.0]))

        display_instances(shader3_program,
                          objects.ID,
                          objects.verticies_sizes[0],
                          objects.verticies_sizes[1],
                          GL.GL_TRIANGLE_FAN)

        keys = pygame.key.get_pressed()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                destructor([], [objects], [v_shader_3, f_shader_3], [shader3_program])
                return
            if event.type == pygame.KEYUP and event.key == pygame.K_ESCAPE:
                destructor([], [objects], [v_shader_3, f_shader_3], [shader3_program])
                return

        # Check for arrow key events continuously
        if keys[pygame.K_LEFT]:
            camera_pos.x -= camera_speed
        if keys[pygame.K_RIGHT]:
            camera_pos.x += camera_speed
        if keys[pygame.K_UP]:
            camera_pos.y += camera_speed
        if keys[pygame.K_DOWN]:
            camera_pos.y -= camera_speed
        if keys[pygame.K_w]:
            camera_pos.z += camera_speed
        if keys[pygame.K_s]:
            camera_pos.z -= camera_speed

        # VAOS display code end
        pygame.display.flip()

        if round_counter % 100 == 0:
            print(round(clock.get_fps(), 2))

        round_counter += 1


if __name__ == '__main__':
    try:
        main()
    finally:
        pygame.quit()
