import os
import math
import time
import numpy as np


# Function to clear the screen
def clear():
    os.system("cls" if os.name == "nt" else "clear")


# Helper functions for vector operations
def dot_product(v1, v2):
    return np.dot(v1, v2)


def vector_subtract(v1, v2):
    return v1 - v2


def vector_scalar_multiply(v, scalar):
    return v * scalar


def vector_norm_squared(v):
    return np.dot(v, v)


def update_velocities(v1, v2, x1, x2, m1, m2):
    # Perfectly elastic collision
    delta_v = vector_subtract(v1, v2)
    delta_x = vector_subtract(x1, x2)
    dot_product_value = dot_product(delta_v, delta_x)
    norm_squared = vector_norm_squared(delta_x)

    if norm_squared == 0:
        return v1, v2

    coefficient1 = (2 * m2) / (m1 + m2)
    coefficient2 = (2 * m1) / (m1 + m2)

    v1_new = vector_subtract(
        v1,
        vector_scalar_multiply(
            delta_x, coefficient1 * (dot_product_value / norm_squared)
        ),
    )
    v2_new = vector_subtract(
        v2,
        vector_scalar_multiply(
            delta_x, -coefficient2 * (dot_product_value / norm_squared)
        ),
    )

    return v1_new, v2_new


class Matrix:
    """Base class for matrix (Output screen)"""

    def __init__(self, width, height, fill=0):
        self.width = width
        self.height = height
        self.matrix = np.full((height, width), fill, dtype=str)

    def output(self):
        for row in self.matrix:
            print(" ".join(row))

    def draw_circle(self, circle):
        y, x = np.ogrid[: self.height, : self.width]
        mask = (x - circle.center_x) ** 2 + (
            y - circle.center_y
        ) ** 2 <= circle.radius**2
        self.matrix[mask] = str(circle.fill)

    def draw_trace(self, trace_buffer):
        for i, trace in enumerate(trace_buffer):
            for point in trace:
                x, y = point
                if 0 <= x < self.width and 0 <= y < self.height:
                    self.matrix[y, x] = "."


class Object:
    """Base class for physical objects"""

    def __init__(
        self, center_x, center_y, fill=1, mass=1, vx=0, vy=0, movable=True, gravity=True
    ):
        self.center_x, self.center_y = center_x, center_y
        self.fill = fill
        self.mass = mass
        self.vx, self.vy = vx, vy
        self.movable = movable
        self.gravity = gravity

    def move(self, dt):
        if self.movable:
            self.center_x += self.vx * dt
            self.center_y += self.vy * dt


class Circle(Object):
    """Circle with a radius"""

    def __init__(
        self,
        radius,
        center_x,
        center_y,
        fill=1,
        mass=1,
        vx=0,
        vy=0,
        collision=True,
        movable=True,
        gravity=True,
    ):
        super().__init__(center_x, center_y, fill, mass, vx, vy, movable, gravity)
        self.radius = radius
        self.collision = collision

    def resize(self, radius):
        self.radius = radius


G = 1  # Gravitational constant
MIN_DISTANCE_SQ = 1e-4  # Minimum distance squared for stability


# Universal law of gravitation
def calc_accel(obj_1, obj_2):
    """
    Calculate ojbects velocities relative to each other
    """
    if obj_2.gravity:
        dx = obj_2.center_x - obj_1.center_x
        dy = obj_2.center_y - obj_1.center_y
        distance_sq = dx**2 + dy**2
        distance_sq = max(distance_sq, MIN_DISTANCE_SQ)  # Limit minimum distance
        distance = math.sqrt(distance_sq)

        force = G * obj_1.mass * obj_2.mass / distance_sq
        a_x = force * dx / distance
        a_y = force * dy / distance
        return a_x, a_y
    else:
        return 0, 0


def handle_collision(circle1, circle2):
    """
    Collision handler between two circles(https://en.wikipedia.org/wiki/Elastic_collision)
    """
    if circle1.collision:
        dx = circle2.center_x - circle1.center_x
        dy = circle2.center_y - circle1.center_y
        distance_sq = dx**2 + dy**2
        min_distance = circle1.radius + circle2.radius

        if distance_sq <= min_distance**2:
            distance = math.sqrt(
                max(distance_sq, MIN_DISTANCE_SQ)
            )  # Prevent division by zero
            overlap = min_distance - distance
            if distance < MIN_DISTANCE_SQ:
                dx, dy = 1, 0  # Horizontal separation if centers coincide

            # Position correction
            correction = (overlap / (2 * distance)) if distance != 0 else 0
            circle1.center_x -= dx * correction
            circle1.center_y -= dy * correction
            circle2.center_x += dx * correction
            circle2.center_y += dy * correction

            # Update velocities
            new_v1, new_v2 = update_velocities(
                np.array([circle1.vx, circle1.vy]),
                np.array([circle2.vx, circle2.vy]),
                np.array([circle1.center_x, circle1.center_y]),
                np.array([circle2.center_x, circle2.center_y]),
                circle1.mass,
                circle2.mass,
            )
            circle1.vx, circle1.vy = new_v1[0], new_v1[1]
            circle2.vx, circle2.vy = new_v2[0], new_v2[1]


def handle_wall_collision(circle, matrix):
    """
    Collision handler with matrix boundaries
    """
    if circle.collision:
        # Adjust position before changing velocity
        if circle.center_x - circle.radius < 0:
            circle.center_x = circle.radius
            circle.vx = abs(circle.vx)  # Bounce right
        elif circle.center_x + circle.radius > matrix.width:
            circle.center_x = matrix.width - circle.radius
            circle.vx = -abs(circle.vx)  # Bounce left

        if circle.center_y - circle.radius < 0:
            circle.center_y = circle.radius
            circle.vy = abs(circle.vy)  # Bounce down
        elif circle.center_y + circle.radius > matrix.height:
            circle.center_y = matrix.height - circle.radius
            circle.vy = -abs(circle.vy)  # Bounce up


def create_trace_buffer(circles, length=5):
    """
    Create a buffer of past coordinates
    """
    buffer = [[] for _ in range(len(circles))]
    while True:
        for i, circle in enumerate(circles):
            buffer[i].append([int(circle.center_x), int(circle.center_y)])
            if len(buffer[i]) > length:
                buffer[i].pop(0)
        yield buffer


def main():
    # Create circles
    HEIGHT = 50
    WIDTH = 50
    GRAVITY = 50000
    circle1 = Circle(7, 10, 25, fill=8, vx=7, vy=0, mass=8)
    circle2 = Circle(2, 40, 30, fill=3, mass=3)
    circle3 = Circle(4, 20, 20, fill=5, mass=5)
    gravity = Circle(1, WIDTH // 2, HEIGHT + 40, 0, GRAVITY, 0, 0, False, False)
    circles = [gravity, circle1, circle2, circle3]
    dt = 0.01  # Time step

    trace_buffer_generator = create_trace_buffer(circles, length=30)

    # Main loop
    while True:
        clear()
        matrix = Matrix(WIDTH, HEIGHT, "_")

        # Update gravity
        for i in range(len(circles)):
            for j in range(len(circles)):
                if i != j:
                    a_x, a_y = calc_accel(circles[i], circles[j])
                    circles[i].vx += a_x * dt
                    circles[i].vy += a_y * dt

        # Update positions
        for circle in circles:
            circle.move(dt)

        # Handle collisions
        for i in range(len(circles)):
            handle_wall_collision(circles[i], matrix)
            for j in range(i + 1, len(circles)):
                handle_collision(circles[i], circles[j])

        # Drawing
        for circle in circles:
            matrix.draw_circle(circle)

        trace_buffer = next(trace_buffer_generator)
        matrix.draw_trace(trace_buffer)

        matrix.output()
        time.sleep(0.05)


if __name__ == "__main__":
    main()
