import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import pml
import pml_1d
import part_b

# u, v, w, xm, ym = pml.main()
# u, v, s, xs = pml_1d.main()
xm, ym, u = part_b.main()


def graph(t):
    fig = plt.figure()
    ax = p3.Axes3D(fig, xlabel='x', ylabel='y')
    ax.set_zlim3d(top=1)
    ax.plot_surface(xm, ym, u[t, ...], cmap='plasma')
    # ax = fig.add_subplot(xlabel='x')
    # ax.set_ylim(bottom=-1, top=1)
    # ax.plot(xs, u[t])
    plt.show()
    plt.cla()
    plt.clf()

