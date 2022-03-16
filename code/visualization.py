import numpy as np
import matplotlib.pyplot as plt


def ds_visual(x: np.array, y: np.array, z: np.array):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z)
    plt.show()

    t = np.linspace(0, len(x), len(x))

    figure = plt.figure(figsize=(12, 12))
    axes = figure.subplots(3, 2)
    axes[0, 0].plot(t, x, label='X(t)')
    axes[0, 0].grid()
    axes[0, 0].legend()
    axes[1, 0].plot(t, y, label='Y(t)')
    axes[1, 0].grid()
    axes[1, 0].legend()
    axes[2, 0].plot(t, z, label='Z(t)')
    axes[2, 0].grid()
    axes[2, 0].legend()

    axes[0, 1].plot(x, y, label='XY')
    axes[0, 1].grid()
    axes[0, 1].legend()
    axes[1, 1].plot(x, z, label='XZ')
    axes[1, 1].grid()
    axes[1, 1].legend()
    axes[2, 1].plot(y, z, label='YZ')
    axes[2, 1].grid()
    axes[2, 1].legend()
    plt.show()


def result_visual(ds, answ0, answ1):
    assert len(answ0) == len(answ1) == len(ds)

    answ0 = np.array(answ0)
    answ1 = np.array(answ1)
    t = np.arange(len(answ0))

    plt.plot(t, ds[:, 0], color='green', label='X')
    plt.plot(t, answ0[:, 0], color='blue', label='(3/p/3) X')
    plt.plot(t, answ1[:, 0], color='orange', label='(6/p/3) X')
    plt.legend()
    plt.grid()
    plt.show()

    plt.plot(t, ds[:, 1], color='green', label='Y')
    plt.plot(t, answ0[:, 1], color='blue', label='(3/p/3) Y')
    plt.plot(t, answ1[:, 1], color='orange', label='(6/p/3) Y')
    plt.legend()
    plt.grid()
    plt.show()

    plt.plot(t, ds[:, 2], color='green', label='Z')
    plt.plot(t, answ0[:, 2], color='blue', label='(3/p/3) Z')
    plt.plot(t, answ1[:, 2], color='orange', label='(6/p/3) Z')
    plt.legend()
    plt.grid()
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(ds[:, 0], ds[:, 1], ds[:, 2])
    ax.plot(answ0[:, 0], answ0[:, 1], answ0[:, 2])
    ax.plot(answ1[:, 0], answ1[:, 1], answ1[:, 2])
    plt.show()


def cascade_visual(ds, answ):
    answ = np.array(answ)
    t = np.arange(len(answ))

    plt.plot(t, ds[:, 0], color='green', label='X')
    plt.plot(t, answ[:, 0], color='blue', label='(3/p/3) X')
    plt.legend()
    plt.grid()
    plt.show()

    plt.plot(t, ds[:, 1], color='green', label='Y')
    plt.plot(t, answ[:, 1], color='blue', label='(3/p/3) Y')
    plt.legend()
    plt.grid()
    plt.show()

    plt.plot(t, ds[:, 2], color='green', label='Z')
    plt.plot(t, answ[:, 2], color='blue', label='(3/p/3) Z')
    plt.legend()
    plt.grid()
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(ds[:, 0], ds[:, 1], ds[:, 2])
    ax.plot(answ[:, 0], answ[:, 1], answ[:, 2])
    plt.show()
