import numpy as np


def GetDataSet(dxyz: tuple, size, h=0.01, learn_size=0.8, valid_size=0.1, test_size=0.1):
    assert learn_size + valid_size + test_size == 1

    data = [(0.1, 0.1, 0.1)]

    for i in range(size - 1):
        k1 = [0.0, 0.0, 0.0]
        k1[0] = dxyz[0](data[i][0], data[i][1], data[i][2])
        k1[1] = dxyz[1](data[i][0], data[i][1], data[i][2])
        k1[2] = dxyz[2](data[i][0], data[i][1], data[i][2])

        k2 = [0.0, 0.0, 0.0]
        k2[0] = dxyz[0](data[i][0] + k1[0] * (h / 2),
                        data[i][1] + k1[1] * (h / 2),
                        data[i][2] + k1[2] * (h / 2))
        k2[1] = dxyz[1](data[i][0] + k1[0] * (h / 2),
                        data[i][1] + k1[1] * (h / 2),
                        data[i][2] + k1[2] * (h / 2))
        k2[2] = dxyz[2](data[i][0] + k1[0] * (h / 2),
                        data[i][1] + k1[1] * (h / 2),
                        data[i][2] + k1[2] * (h / 2))

        k3 = [0.0, 0.0, 0.0]
        k3[0] = dxyz[0](data[i][0] + k2[0] * (h / 2),
                        data[i][1] + k2[1] * (h / 2),
                        data[i][2] + k2[2] * (h / 2))
        k3[1] = dxyz[1](data[i][0] + k2[0] * (h / 2),
                        data[i][1] + k2[1] * (h / 2),
                        data[i][2] + k2[2] * (h / 2))
        k3[2] = dxyz[2](data[i][0] + k2[0] * (h / 2),
                        data[i][1] + k2[1] * (h / 2),
                        data[i][2] + k2[2] * (h / 2))

        k4 = [0.0, 0.0, 0.0]
        k4[0] = dxyz[0](data[i][0] + k3[0] * h, data[i][1] + k3[1] * h, data[i][2] + k3[2] * h)
        k4[1] = dxyz[1](data[i][0] + k3[0] * h, data[i][1] + k3[1] * h, data[i][2] + k3[2] * h)
        k4[2] = dxyz[2](data[i][0] + k3[0] * h, data[i][1] + k3[1] * h, data[i][2] + k3[2] * h)

        data.append((data[i][0] + (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) * (h / 6),
                     data[i][1] + (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) * (h / 6),
                     data[i][2] + (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) * (h / 6)))

    learn_size = int(size * learn_size)
    valid_size = int(size * valid_size)
    test_size = int(size * test_size)

    learn_set = np.array(data[:learn_size])
    validation_set = np.array(data[learn_size: learn_size + valid_size])
    test_set = np.array(data[learn_size + valid_size: learn_size + valid_size + test_size])

    return learn_set, validation_set, test_set
