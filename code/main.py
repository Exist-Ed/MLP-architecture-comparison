import numpy as np

import visualization as v
import DataSet as ds
import NN
from config import *  # configuration values for dataset and NLP's


def main():
    learn_ds, valid_ds, test_ds = ds.GetDataSet((dx, dy, dz), ds_size, learn_size=learn_size, valid_size=valid_size,
                                                test_size=test_size, h=h)
    v.ds_visual(learn_ds[:, 0], learn_ds[:, 1], learn_ds[:, 2])

    mlp0 = NN.MLP(input_layer_size_0, hidden_layer_size_0, alpha=alpha_0, Em=Em_0)
    mlp1 = NN.MLP(input_layer_size_1, hidden_layer_size_1, alpha=alpha_1, Em=Em_1)

    mlp0.fit(learn_ds)
    mlp1.fit(learn_ds)

    answers0 = []
    answers1 = []
    error0 = []
    error1 = []
    for i in range(int(input_layer_size_1 / 3), len(valid_ds)):
        answers0.append(mlp0.predict(valid_ds[i - int(input_layer_size_0 / 3): i]))
        error0.append(np.sum((answers0[-1] - valid_ds[i]) ** 2))

        answers1.append(mlp1.predict(valid_ds[i - int(input_layer_size_1 / 3): i]))
        error1.append(np.sum((answers1[-1] - valid_ds[i]) ** 2))

    v.result_visual(valid_ds[int(input_layer_size_1 / 3):], answers0, answers1)
    with open('error_logs.txt', 'a+') as f:
        for i in (error0, error1):
            f.write(f'{0.5 * np.sum(i)}, ')
        f.write('\n')

    with open('timer_logs.txt', 'a+') as f:
        f.write('\n')

    answers = []
    answers.append(mlp0.predict(valid_ds[0]))
    for i in range(int(input_layer_size_0 / 3) + 1, len(valid_ds)):
        answers.append(mlp0.predict(answers[-1]))

    v.cascade_visual(valid_ds[int(input_layer_size_0 / 3):], answers)


if __name__ == '__main__':
    main()
