import matplotlib.pyplot as plt
import part_a
import numpy as np
import time

delta = 0.01

start = time.process_time()

xs, timeline, u, dx, dt = part_a.main(delta)

stop = time.process_time()

print("time taken: {} seconds".format(stop - start))


def graph(t):
    fig = plt.figure()
    ax = fig.add_subplot(xlabel='x')
    ax.set_ylim(bottom=-1, top=1)
    ax.plot(xs, u[..., t])
    plt.show()
    plt.cla()
    plt.clf()


def gen_graphs():
    q = dt / dx
    for t in np.linspace(0, 8, num=9):
        fig = plt.figure()
        title = 'delta = {0}, t = {1}, q={4}'.format(delta, t, dt, dx, q)
        ax = fig.add_subplot(
            xlabel='x', ylabel='u',
            title=title
        )
        ax.set_ylim(bottom=-0.2, top=1.2)
        t1 = np.where(timeline == t)[0][0]
        ax.plot(xs, u[..., t1])
        plt.savefig("./part_a_results/reflective/q={}/".format(q) + title + ".png")
        plt.cla()
        plt.clf()
        plt.close()


if __name__ == "__main__":
    gen_graphs()
