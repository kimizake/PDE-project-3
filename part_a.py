import time

import matplotlib.pyplot as plt
import numpy as np


def first_pass(**kwargs):
    """
    Defines our initial U  values
    """
    dx = kwargs['dx']
    delta = kwargs['delta']
    h = int(1 + 4 / dx)

    def calc_value(n):
        x = n * dx - 2
        if -delta <= x <= delta:
            return np.cos(np.pi * x / 2 / delta)
        else:
            return 0

    return np.vectorize(calc_value, otypes=[float])(np.arange(h)).reshape(h, 1)


def run(u, **kwargs):
    """
    Defines remaining U values
    """
    dt = kwargs['dt']
    dx = kwargs['dx']
    h, t = u.shape
    j = t - 1
    q = dt / dx

    def calc_value(n):
        def is_at_boundary(index):
            return index == 0 or index == h - 1

        _u = u[n + 1, j] if n == 0 else u[n - 1, j] if n == h - 1 else None     # Define the ghost point as required

        """
        When we want to define u at t = 1, we need u at t = 0 and at t = -1, however the latter point
        obviously doesn't exist. We instead use the initial conditions and the ghost point to generate these values.
        """
        if t == 1 and is_at_boundary(n):
            return u[n, j] + q ** 2 * (_u - u[n, j])
        elif t == 1 and not is_at_boundary(n):
            return u[n, j] + q ** 2 * (u[n + 1, j] + u[n - 1, j] - 2 * u[n, j]) / 2

        """
        This is for the general case t > 1. When we are at a boundary we want to apply the boundary condition,
        otherwise we just apply the normal leapfrog scheme.
        """
        if is_at_boundary(n):
            bc = kwargs['boundary_condition']       # bc is a pointer to a function
            return bc(q, u[n, j - 1], u[n, j], _u)
        else:
            return -u[n, j - 1] + 2 * u[n, j] + q ** 2 * (u[n + 1, j] - 2 * u[n, j] + u[n - 1, j])

    """
    We apply the algorithm to the current layer and 
    """
    new_u = np.vectorize(calc_value)(np.arange(h)).reshape((h, 1))

    return np.hstack((u, new_u))


def non_reflective(q, prev, curr, _u):
    """
    Applies the non-non reflective boundary condition u_x = +- u_t
    :param q: The Courant Number
    :param prev: U at time step j - 1
    :param curr: U at time step j
    :param _u: value of the ghost point beyond boundary
    :return: U at time step j + 1
    """
    a = 1 - q
    b = 1 + q
    c = a / b

    return -c * prev + 2 * a * curr + 2 * q ** 2 * _u / b


def reflective(q, prev, curr, _u):
    """
        Applies the Neumann non reflective boundary condition u_x = 0
        :param q: The Courant Number
        :param prev: U at time step j - 1
        :param curr: U at time step j
        :param _u: value of the ghost point beyond boundary
        :return: U at time step j + 1
        """
    return -prev + 2 * curr + 2 * q ** 2 * (_u - curr)


def main(delta):
    """
    Applies the leapfrog algorithm
    :return: u values through time
    """
    """
    Define the x axis and the time interval, and the discretisation scheme
    num parameter is how many sub intervals we want, and the + 1 is for the last point
    """
    timeline, dt = np.linspace(0, 8, num=800 + 1, endpoint=True, retstep=True)
    xs, dx = np.linspace(-2, 2, num=400 + 1, endpoint=True, retstep=True)

    print("dt = {0}\n dx = {1}".format(dt, dx))
    # assert dt <= dx   # CFL safety

    """
    Specify which boundary condition to use,
    e.g.
    'boundary_condition': non_reflective => non non reflective boundary
    'boundary_condition': non reflective     => solid wall
    Note we are passing a function as an argument here.
    """
    params = {
        'dt': dt,
        'dx': dx,
        'delta': delta,
        'boundary_condition': reflective
    }

    """
    In the 1d case u at time t is a 1d vector, however since we march through time, 
    we want to stack these vectors on top of one another.
    Therefore u below is a 2d matrix which is indexed as: u[x axis index, time index].
    """
    u = first_pass(**params)    # Define U at t=0

    for _ in timeline[1:]:
        u = run(u, **params)    # Define U for all other t

    return xs, timeline, u, dx, dt


def plot(x, t, u, dx, dt):
    """
    Generates a 3d plot of the system where we see the wave transform through time
    :param x: x axis
    :param t: time axis
    :param u: u axis
    :return: None
    """
    surface = plt.figure(1)
    ax1 = surface.add_subplot(projection='3d',
                              title='1D surface plot with non reflective boundaries',
                              xlabel='displacement',
                              ylabel='time',
                              zlabel='amplitude')
    xm, tm = np.meshgrid(x, t)
    ax1.plot_surface(xm, tm, u.T, cmap='plasma')      # Transposition due to indexing

    cross_section = plt.figure(2)
    ax2 = cross_section.add_subplot(title="1D wave plot after 4 seconds with non reflective boundaries",
                                    xlabel="displacement",
                                    ylabel="amplitude")
    ax2.plot(x, u[:, -1:])


if __name__ == "__main__":
    start = time.process_time()
    plot(*main(0.5))   # Runs and plots the main function
    stop = time.process_time()
    print("time elapsed: {} seconds".format(stop - start))
    plt.show()
