import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
import time


def first_pass(rs):
    delta = 1

    def calc_value(r):
        if r <= delta:
            return np.cos(np.pi * r / 2 / delta)
        else:
            return 0

    return np.vectorize(calc_value, otypes=[float])(rs)


def main_pass(u, v, w, px, py, pz, sx=None, sy=None, dt=0.01, dx=0.01, dy=0.01, q=1, i=0, j=0):
    t, n, m = u.shape

    u_curr = u[t - 1, ...]
    v_curr = v[t - 1, ...]
    w_curr = w[t - 1, ...]
    px_curr = px[t - 1, ...]
    py_curr = py[t - 1, ...]
    pz_curr = pz[t - 1, ...]

    u_prev = np.zeros((n, m)) if t == 1 else u[t - 2, ...]
    v_prev = np.zeros((n, m)) if t == 1 else v[t - 2, ...]
    w_prev = np.zeros((n, m)) if t == 1 else w[t - 2, ...]
    px_prev = np.zeros((n, m)) if t == 1 else px[t - 2, ...]
    py_prev = np.zeros((n, m)) if t == 1 else py[t - 2, ...]
    pz_prev = np.zeros((n, m)) if t == 1 else pz[t - 2, ...]

    v_next = v_prev - 2 * dt * sx * v_curr
    v_next[:, 1:-1] += dt * (u_curr[:, 2:] - u_curr[:, :-2]) / dx

    w_next = w_prev - 2 * dt * sy * w_curr
    w_next[1:-1, :] += q * dt * (u_curr[2:, :] - u_curr[:-2, :]) / dy

    px_next = px_prev
    px_next[1:-1, :] += dt * sx[1:-1, :] * (w_curr[2:, :] - w_curr[:-2, :]) / dy

    py_next = py_prev
    py_next[:, 1:-1] += dt * sy[:, 1:-1] * (v_curr[:, 2:] - v_curr[:, :-2]) / dx

    pz_next = pz_prev + 2 * dt * u_curr

    u_next = u_prev - 2 * dt * (sx + sy) * u_curr + 2 * dt * (px_curr + py_curr) - 2 * dt * sx * sy * pz_curr
    u_next[:, 1:-1] += dt * (v_curr[:, 2:] - v_curr[:, :-2]) / dx
    u_next[1:-1, :] += dt * (w_curr[2:, :] - w_curr[:-2, :]) / dy

    # Define inner points by wave equation
    # u_next[j:-j, i:-i] = 2 * u_curr[j:-j, i:-i] - u_prev[j:-j, i:-i] + dt**2 * (
    #     (u_curr[j:-j, i+1:-i+1] - 2 * u_curr[j:-j, i:-i] + u_curr[j:-j, i-1:-i-1]) / dx ** 2 +
    #     q * (u_curr[j+1:-j+1, i:-i] - 2 * u_curr[j:-j, i:-i] + u_curr[j-1:-j-1, i:-i]) / dy ** 2
    # )
    #
    # if t == 1:
    #     u_next[j:-j, i:-i] /= 2

    u_next = u_next[np.newaxis, ...]
    v_next = v_next[np.newaxis, ...]
    w_next = w_next[np.newaxis, ...]
    px_next = px_next[np.newaxis, ...]
    py_next = py_next[np.newaxis, ...]
    pz_next = pz_next[np.newaxis, ...]

    return np.vstack((u, u_next)), np.vstack((v, v_next)), np.vstack((w, w_next)), np.vstack((px, px_next)), np.vstack((py, py_next)), np.vstack((pz, pz_next))


def main():
    start = time.process_time()

    timeline, dt = np.linspace(0, 10, num=1000 + 1, endpoint=True, retstep=True)
    xs, dx = np.linspace(-4, 4, num=40 + 1, endpoint=True, retstep=True)
    ys, dy = np.linspace(-4, 4, num=40 + 1, endpoint=True, retstep=True)
    xm, ym = np.meshgrid(xs, ys)

    rs = np.sqrt(xm ** 2 + ym ** 2)  # returns grid of distances to origin
    u = first_pass(rs)

    v = np.zeros(u.shape)
    w = np.zeros(u.shape)
    px = np.zeros(u.shape)
    py = np.zeros(u.shape)
    pz = np.zeros(u.shape)

    sx = np.zeros(u.shape)
    sy = np.zeros(u.shape)

    i = np.where(xs == -2)[0][0]
    sx[:, :i] = (-xm[:, :i] - 2)**2
    sx[:, -i:] = (xm[:, -i:] - 2)**2

    j = np.where(ys == -2)[0][0]
    sy[:j, :] = (-ym[:j, :] - 2)**3
    sy[-j:, :] = (ym[-j:, :] - 2)**3

    u = u[np.newaxis, ...]
    v = v[np.newaxis, ...]
    w = w[np.newaxis, ...]
    px = px[np.newaxis, ...]
    py = py[np.newaxis, ...]
    pz = pz[np.newaxis, ...]

    params = u, v, w, px, py, pz

    print(dx, dy, dt)

    for _ in timeline[1:]:
        params = main_pass(*params, sx=sx, sy=sy, dt=dt, dy=dy, dx=dx, q=1, i=i, j=j)
    stop = time.process_time()
    print("time elapsed: {} seconds".format(stop - start))

    return params[0], params[1], params[2], xm, ym
    # return sx, xm, ym


if __name__ == "__main__":
    u, v, w, xm, ym = main()

    fig = plt.figure()

    t = -1
    ax = p3.Axes3D(fig)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('T={}'.format(t))
    ax.plot_surface(xm, ym, u[t, ...], cmap='plasma')

    plt.show()
