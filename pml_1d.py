import numpy as np
import time


def first_pass(rs):
    delta = 0.5

    def calc_value_u(r):
        if - delta <= r <= delta:
            return np.cos(np.pi * r / 2 / delta)
        else:
            return 0

    return np.vectorize(calc_value_u, otypes=[float])(rs)


def main_pass(u, v, s=None, dx=0.1, dt=0.1):
    t, m = u.shape

    u_curr = u[t - 1, ...]
    v_curr = v[t - 1, ...]

    u_prev = np.zeros(m) if t == 1 else u[t - 2, ...]
    v_prev = np.zeros(m) if t == 1 else u[t - 2, ...]

    # Applying Neumann Hard Wall conditions at boundary
    v_next = v_prev - 2 * dt * s * v_curr
    v_next[1:-1] += dt * (u_curr[2:] - u_curr[:-2]) / dx

    u_next = u_prev - 2 * dt * s * u_curr
    u_next[1:-1] += dt * (v_curr[2:] - v_curr[:-2]) / dx
    u_next[0] += dt * v_curr[1] / dx
    u_next[-1] -= dt * v_curr[-1] / dx

    u_next = u_next.reshape((1, m))
    v_next = v_next.reshape((1, m))
    return np.vstack((u, u_next)), np.vstack((v, v_next))


def main():
    start = time.process_time()

    timeline, dt = np.linspace(0, 5, 500 + 1, endpoint=True, retstep=True)
    xs, dx = np.linspace(-2.5, 2.5, 250 + 1, endpoint=True, retstep=True)
    u = first_pass(xs)
    v = np.zeros(u.shape)
    s = np.zeros(u.shape)

    i = np.where(xs == -2)[0][0]
    s[:i] = (-xs[:i] - 2) ** 2
    s[-i:] = (xs[-i:] - 2) ** 2

    u = u[np.newaxis, ...]
    v = v[np.newaxis, ...]

    params = u, v
    for _ in timeline[1:]:
        params = main_pass(*params, s=s, dx=dx, dt=dt)

    stop = time.process_time()
    print("time elapsed: {} seconds".format(stop - start))

    return params[0], params[1], s, xs

