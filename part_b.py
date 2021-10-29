import matplotlib.pyplot as plt
import numpy as np
import time


def first_pass(rs=None):
    delta = 0.2

    def calc_value(r):
        if r <= delta:
            return np.cos(np.pi * r / 2 / delta)
        else:
            return 0

    return np.vectorize(calc_value, otypes=[float])(rs)


def main_pass(**kwargs):
    u = kwargs['u']
    dx = kwargs['dx']
    dy = kwargs['dy']
    dt = kwargs['dt']
    q = kwargs['q']

    t, n, m = u.shape

    u_next = np.zeros((n, m))
    u_curr = u[t - 1, ...]
    u_prev = u[t - 2, ...] if t > 1 else np.zeros((n, m))

    u_next[1: -1, 1: -1] = 2 * u_curr[1:-1, 1:-1] - u_prev[1:-1, 1:-1] + dt ** 2 * (
        (u_curr[1:-1, 2:] - 2 * u_curr[1:-1, 1:-1] + u_curr[1:-1, :-2]) / dx ** 2 +
        q * (u_curr[2:, 1:-1] - 2 * u_curr[1:-1, 1:-1] + u_curr[:-2, 1:-1]) / dy ** 2
    )

    if t == 1:
        u_next = u_next / 2
        u_prev = u_next

    # At y = -2

    u_next[0, 1:-1] = u_next[2, 1:-1] - u_prev[2, 1:-1] + u_prev[0, 1:-1] - 4 * dt * dy * (
        (u_next[1, 1:-1] - 2 * u_curr[1, 1:-1] + u_prev[1, 1:-1]) / dt ** 2 -
        .5 * (u_curr[1, 2:] - 2 * u_curr[1, 1:-1] + u_curr[1, :-2]) / dx ** 2
    )

    # u_next[0, 1:-1] = u_next[2, 1:-1] - dy ** 2 * (
    #     (u_next[1, 1:-1] - u_prev[1, 1:-1]) / dt ** 2
    # )

    # At x = -2

    u_next[1:-1, 0] = u_next[1:-1, 2] - u_prev[1:-1, 2] + u_prev[1:-1, 0] - 4 * dt * dx * (
        (u_next[1:-1, 1] - 2 * u_curr[1:-1, 1] + u_prev[1:-1, 1]) / dt ** 2 -
        .5 * (u_curr[2:, 1] - 2 * u_curr[1:-1, 1] + u_curr[:-2, 1]) / dy ** 2
    )

    # u_next[1:-1, 0] = u_next[1:-1, 2] - dx ** 2 * (
    #     (u_next[1:-1, 1] - u_prev[1:-1, 1]) / dt ** 2
    # )

    # At y = 2

    u_next[-1, 1:-1] = u_next[-3, 1:-1] - u_prev[-3, 1:-1] + u_prev[-1, 1:-1] - 4 * dt * dy * (
        (u_next[-2, 1:-1] - 2 * u_curr[-2, 1:-1] + u_prev[-2, 1:-1]) / dt ** 2 -
        .5 * (u_curr[-2, 2:] - 2 * u_curr[-2, 1:-1] + u_curr[-2, :-2]) / dx ** 2
    )

    # u_next[-1, 1:-1] = u_next[-3, 1:-1] - dy ** 2 * (
    #     (u_next[-2, 1:-1] - u_prev[-2, 1:-1]) / dt ** 2
    # )

    # At x = 2

    u_next[1:-1, -1] = u_next[1:-1, -3] - u_prev[1:-1, -3] + u_prev[1:-1, -1] - 4 * dt * dx * (
        (u_next[1:-1, -2] - 2 * u_curr[1:-1, -2] + u_prev[1:-1, -2]) / dt ** 2 -
        .5 * (u_curr[2:, -2] - 2 * u_curr[1:-1, -2] + u_curr[:-2, -2]) / dy ** 2
    )

    # u_next[1:-1, -1] = u_next[1:-1, -3] - dx ** 2 * (
    #     (u_next[1:-1, -2] - u_prev[1:-1, -2]) / dt ** 2
    # )

    u_next = u_next[np.newaxis, ...]
    return np.vstack((u, u_next))


def main():
    start = time.process_time()

    timeline, dt = np.linspace(0, 5, num=500 + 1, endpoint=True, retstep=True)
    xs, dx = np.linspace(-2, 2, num=200 + 1, endpoint=True, retstep=True)
    ys, dy = np.linspace(-2, 2, num=200 + 1, endpoint=True, retstep=True)
    xm, ym = np.meshgrid(xs, ys)

    rs = np.sqrt(xm ** 2 + ym ** 2)  # returns grid of distances to origin
    u = first_pass(rs=rs)

    _u = np.zeros((u.shape[0] + 2, u.shape[1] + 2))     # padding for ghost points
    _u[1:-1, 1:-1] = u
    _u = _u[np.newaxis, ...]

    for _ in timeline:
        _u = main_pass(u=_u, dx=dx, dt=dt, dy=dy, q=1)

    print(dx, dy, dt)
    stop = time.process_time()
    print("time elapsed: {} seconds".format(stop - start))

    return xm, ym, _u[:, 1:-1, 1:-1]


def plot(xm, ym, u):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(xm, ym, u[-1, 1:-1, 1:-1], cmap='plasma')
    plt.show()


if __name__ == "__main__":
    plot(*main())
