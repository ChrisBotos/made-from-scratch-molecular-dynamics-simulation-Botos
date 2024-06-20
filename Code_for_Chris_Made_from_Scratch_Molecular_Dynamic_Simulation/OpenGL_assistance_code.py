import OpenGL.GL as GL
import ctypes
import numpy as np
import pygame
import pyrr
from pyrr import matrix44


class VAO:
    def __init__(self):
        self.ID = GL.glGenVertexArrays(1)
        # GL.glBindVertexArray(self.ID)
        self.bind_ids = []
        self.shader_ids = []
        self.verticies_sizes = []

    def new_array_buffer(self, shader_name, shader_var_name, verticies_size, input_buffer, DRAW_TYPE):
        GL.glBindVertexArray(self.ID)
        self.bind_ids.append(GL.glGenBuffers(1))
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.bind_ids[-1])
        self.shader_ids.append(GL.glGetAttribLocation(shader_name, shader_var_name))
        GL.glEnableVertexAttribArray(self.shader_ids[-1])
        GL.glVertexAttribPointer(self.shader_ids[-1], 4, GL.GL_FLOAT, False, 0, ctypes.c_void_p(0))
        GL.glBufferData(GL.GL_ARRAY_BUFFER, verticies_size * 16, input_buffer, DRAW_TYPE)
        self.verticies_sizes.append(verticies_size)
        # not tested yet

    def new_element_buffer(self, verticies_size, input_buffer, DRAW_TYPE):
        GL.glBindVertexArray(self.ID)
        self.bind_ids.append(GL.glGenBuffers(1))
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.bind_ids[-1])
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, verticies_size * 4, input_buffer, DRAW_TYPE)
        self.verticies_sizes.append(verticies_size)

    def new_instance_buffer(self, shader_name, shader_var_name, verticies_size, input_buffer, DRAW_TYPE):
        GL.glBindVertexArray(self.ID)
        self.bind_ids.append(GL.glGenBuffers(1))
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.bind_ids[-1])
        self.shader_ids.append(GL.glGetAttribLocation(shader_name, shader_var_name))
        GL.glEnableVertexAttribArray(self.shader_ids[-1])
        GL.glVertexAttribPointer(self.shader_ids[-1], 4, GL.GL_FLOAT, False, 0, ctypes.c_void_p(0))
        GL.glBufferData(GL.GL_ARRAY_BUFFER, verticies_size * 16, input_buffer, DRAW_TYPE)
        GL.glVertexAttribDivisor(self.shader_ids[-1], 1)
        self.verticies_sizes.append(verticies_size)

    def enable_buffer(self, index):
        GL.glBindVertexArray(self.ID)
        GL.glEnableVertexAttribArray(self.shader_ids[index])

    def disable_buffer(self, index):
        GL.glBindVertexArray(self.ID)
        GL.glDisableVertexAttribArray(self.shader_ids[index])

    def replace_buffer(self, new_buffer, index):
        GL.glBindVertexArray(self.ID)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.bind_ids[index])
        GL.glBufferSubData(GL.GL_ARRAY_BUFFER, 0, self.verticies_sizes[index] * 16, new_buffer)

    def delete(self):
        if len(self.bind_ids) > 0:
            GL.glBindVertexArray(self.ID)
            bind_ids_buf = np.array(self.bind_ids, dtype=np.uint32)
            GL.glDeleteBuffers(len(self.bind_ids), bind_ids_buf)
            self.bind_ids.clear()
            self.shader_ids.clear()
            self.verticies_sizes.clear()
        VAO_buf = np.zeros(1, dtype=np.uint32)
        VAO_buf[0] = np.uint32(self.ID)
        GL.glDeleteVertexArrays(1, VAO_buf)


class Texture:
    def __init__(self, filepath, vertical_flip, horizontal_flip, DRAW_WRAPPER, DRAW_FILTER):
        self.texture = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, DRAW_WRAPPER)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, DRAW_WRAPPER)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, DRAW_FILTER)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, DRAW_FILTER)
        image = pygame.image.load(filepath).convert_alpha()
        image = pygame.transform.flip(image, vertical_flip, horizontal_flip)
        image_width, image_height = image.get_rect().size
        img_data = pygame.image.tostring(image, 'RGBA')
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, image_width, image_height, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE,
                        img_data)
        GL.glGenerateMipmap(GL.GL_TEXTURE_2D)

    def use(self):
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture)

    def delete(self):
        texture_buf = np.zeros(1, dtype=np.uint32)
        texture_buf[0] = np.uint32(self.texture)
        GL.glDeleteTextures(1, texture_buf)


def background(r, g, b, a):
    GL.glClearColor(r, g, b, a)
    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)


def display_arrays(shader, vertex_array_object, size, DRAW_TYPE):
    GL.glUseProgram(shader)
    GL.glBindVertexArray(vertex_array_object)
    GL.glDrawArrays(DRAW_TYPE, 0, size)
    # not tested yet


def display_elements(shader, vertex_element_array_object, size, DRAW_TYPE):
    GL.glUseProgram(shader)
    GL.glBindVertexArray(vertex_element_array_object)
    GL.glDrawElements(DRAW_TYPE, size, GL.GL_UNSIGNED_INT, None)


def display_instances(shader, vertex_array_object, size, instance_size, DRAW_TYPE):
    GL.glUseProgram(shader)
    GL.glBindVertexArray(vertex_array_object)
    GL.glDrawArraysInstanced(DRAW_TYPE, 0, size, instance_size)
    # not tested yet


def display_elements_instanced(shader, vertex_element_array_object, size, instance_size, DRAW_TYPE):
    GL.glUseProgram(shader)
    GL.glBindVertexArray(vertex_element_array_object)
    GL.glDrawElementsInstanced(DRAW_TYPE, size, GL.GL_UNSIGNED_INT, None, instance_size)


def scale_1f(shader_names, shader_var_name, float32_value):
    for shader_name in shader_names:
        GL.glUseProgram(shader_name)
        GL.glUniform1f(GL.glGetUniformLocation(shader_name, shader_var_name), float32_value)


def scale_3f(shader_names, shader_var_name, float32_valuex, float32_valuey, float32_valuez):
    for shader_name in shader_names:
        GL.glUseProgram(shader_name)
        GL.glUniform3f(GL.glGetUniformLocation(shader_name, shader_var_name), float32_valuex, float32_valuey,
                       float32_valuez)


def translate_3f(shader_names, shader_var_name, float32_valuex, float32_valuey, float32_valuez):
    for shader_name in shader_names:
        GL.glUseProgram(shader_name)
        GL.glUniform3f(GL.glGetUniformLocation(shader_name, shader_var_name), float32_valuex, float32_valuey,
                       float32_valuez)
    # not tested yet


def rotate_Matrix4fv(shader_names, shader_rot_var_name, pos, front, up):
    for shader_name in shader_names:
        GL.glUseProgram(shader_name)
        rot = matrix44.create_look_at(pos, pos + front, up)
        rot_loc = GL.glGetUniformLocation(shader_name, shader_rot_var_name)
        GL.glUniformMatrix4fv(rot_loc, 1, GL.GL_FALSE, rot)


def set_camera_2_Matrix4fv(shader_names, shader_proj_var_name, shader_view_var_name, texture_dimx, texture_dimy, pos, front, up):
    for shader_name in shader_names:
        GL.glUseProgram(shader_name)
        projection = pyrr.matrix44.create_perspective_projection_matrix(
            -(0.0001 + (texture_dimx + texture_dimy) / 2.0) / 2.0, texture_dimx / texture_dimy, 0.0001,
            (texture_dimx + texture_dimy) / 2.0)
        view = matrix44.create_look_at(pos, pos + front, up)
        proj_loc = GL.glGetUniformLocation(shader_name, shader_proj_var_name)
        view_loc = GL.glGetUniformLocation(shader_name, shader_view_var_name)
        GL.glUniformMatrix4fv(proj_loc, 1, GL.GL_FALSE, projection)
        GL.glUniformMatrix4fv(view_loc, 1, GL.GL_FALSE, view)
    # not tested yet


def background_color_3f(shader_names, shader_var_name, float32_valuex, float32_valuey, float32_valuez):
    for shader_name in shader_names:
        GL.glUseProgram(shader_name)
        GL.glUniform3f(GL.glGetUniformLocation(shader_name, shader_var_name), float32_valuex, float32_valuey,
                       float32_valuez)
    # not tested yet


def in_color_3f(shader_names, shader_var_name, float32_valuex, float32_valuey, float32_valuez, float32_valuew):
    for shader_name in shader_names:
        GL.glUseProgram(shader_name)
        GL.glUniform4f(GL.glGetUniformLocation(shader_name, shader_var_name), float32_valuex, float32_valuey,
                       float32_valuez, float32_valuew)


def destructor(textures, VAOS, shaders, programs):
    if textures != None:
        for texture in textures:
            texture.delete()
    if VAOS != None:
        for VAO in VAOS:
            VAO.delete()
    if shaders != None:
        for shader in shaders:
            GL.glDeleteShader(shader)
    if programs != None:
        for program in programs:
            GL.glDeleteProgram(program)