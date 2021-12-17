# -*- coding: utf-8 -*-

import math
import collections

import numpy as np

import pyglet


class FirstPersonCamera(object):
    """First person camera implementation

    Usage:
        import pyglet
        from pyglet.gl import *
        from camera import FirstPersonCamera


        window = pyglet.window.Window(fullscreen=True)
        window.set_exclusive_mouse(True)
        camera = FirstPersonCamera(window)

        @window.event
        def on_draw():
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glLoadIdentity()

            camera.draw()

            # Your draw code here

            return pyglet.event.EVENT_HANDLED

        def on_update(delta_time):
            camera.update(delta_time)

            # Your update code here

        if __name__ == '__main__':
            pyglet.clock.schedule(on_update)
            pyglet.app.run()
    """

    DEFAULT_MOVEMENT_SPEED = 2.0

    DEFAULT_MOUSE_SENSITIVITY = 0.4

    DEFAULT_KEY_MAP = None

    class InputHandler(object):
        def __init__(self):
            self.pressed = collections.defaultdict(bool)
            self.dx = 0
            self.dy = 0

        def on_key_press(self, symbol, modifiers):
            self.pressed[symbol] = True

        def on_key_release(self, symbol, modifiers):
            self.pressed[symbol] = False

        def on_mouse_motion(self, x, y, dx, dy):
            self.dx = dx
            self.dy = dy

    def __init__(
        self,
        window,
        position=(-1, 0, 0.8),
        key_map=None,
        movement_speed=DEFAULT_MOVEMENT_SPEED,
        mouse_sensitivity=DEFAULT_MOUSE_SENSITIVITY,
        y_inv=True,
    ):
        """Create camera object

        Arguments:
            window -- pyglet window which camera attach
            position -- position of camera
            key_map -- dict like FirstPersonCamera.DEFAULT_KEY_MAP
            movement_speed -- speed of camera move (scalar)
            mouse_sensitivity -- sensitivity of mouse (scalar)
            y_inv -- inversion turn above y-axis
        """

        self.position = list(position)

        self.direction = np.array([0.0, 1.0, 0.0])
        self.position = np.array([0.0, 0.0, -1.0])

        self.view_matrix = np.eye(3)

        self.__input_handler = FirstPersonCamera.InputHandler()

        window.push_handlers(self.__input_handler)

        DEFAULT_KEY_MAP = {
            "forward": pyglet.window.key.W,
            "backward": pyglet.window.key.S,
            "left": pyglet.window.key.A,
            "right": pyglet.window.key.D,
            "up": pyglet.window.key.Z,
            "down": pyglet.window.key.X,
        }

        self.key_map = DEFAULT_KEY_MAP if key_map is None else key_map
        self.movement_speed = movement_speed
        self.mouse_sensitivity = mouse_sensitivity

    def update(self, delta_time):
        """Update camera state"""
        angle = np.array(
            [
                math.atan2(self.direction[2], self.direction[0]),
                math.atan2(
                    math.sqrt(
                        self.direction[0] * self.direction[0]
                        + self.direction[2] * self.direction[2]
                    ),
                    self.direction[1],
                ),
            ]
        )
        angle = (
            angle
            + np.array([self.__input_handler.dx, self.__input_handler.dy])
            * self.mouse_sensitivity
            * delta_time
        )
        angle = np.array([angle[0], min(math.pi - 0.1, max(angle[1], 0.1))])

        self.direction = np.array(
            [
                np.sin(angle[1]) * np.cos(angle[0]),
                np.cos(angle[1]),
                np.sin(angle[1]) * np.sin(angle[0]),
            ]
        )
        print("direction is ", self.direction)

        # reset camera position
        self.__input_handler.dx = 0
        self.__input_handler.dy = 0

        f = self.direction / np.linalg.norm(self.direction)

        up = np.array([0.0, 1.0, 0.0])

        s = np.array(
            [
                f[1] * up[2] - f[2] * up[1],
                f[2] * up[0] - f[0] * up[2],
                f[0] * up[1] - f[1] * up[0],
            ]
        )
        s /= np.linalg.norm(s)

        u = np.array(
            [
                s[1] * f[2] - s[2] * f[1],
                s[2] * f[0] - s[0] * f[2],
                s[0] * f[1] - s[1] * f[0],
            ]
        )

        dp = delta_time * self.movement_speed

        if self.__input_handler.pressed[self.key_map["forward"]]:
            self.position += s * dp

        if self.__input_handler.pressed[self.key_map["backward"]]:
            self.position -= s * dp

        if self.__input_handler.pressed[self.key_map["left"]]:
            self.position -= u * dp

        if self.__input_handler.pressed[self.key_map["right"]]:
            self.position += u * dp

        if self.__input_handler.pressed[self.key_map["up"]]:
            self.position += f * dp

        if self.__input_handler.pressed[self.key_map["down"]]:
            self.position -= f * dp

        p = np.array(
            [
                -self.position[0] * s[0]
                - self.position[1] * s[1]
                - self.position[2] * s[2],
                -self.position[0] * u[0]
                - self.position[1] * u[1]
                - self.position[2] * u[2],
                -self.position[0] * f[0]
                - self.position[1] * f[1]
                - self.position[2] * f[2],
            ]
        )

        self.view_matrix = np.array(
            [
                [s[0], u[0], f[0]],
                [s[1], u[1], f[1]],
                [s[2], u[2], f[2]],
                [p[0], p[1], p[2]],
            ]
        ).transpose()
