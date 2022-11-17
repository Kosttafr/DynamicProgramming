import matplotlib.pyplot as plt
import seaborn as sns
import game.algorithms as alg
import game.parameters as param
import numpy as np
import math

class GridSearch:
    def __init__(self):
        self.abs_val_delta = np.zeros((10, 10))
        self.rel_val_delta = np.zeros((10, 10))
        self.abs_time_delta = np.zeros((10, 10))
        self.rel_time_delta = np.zeros((10, 10))

    @staticmethod
    def launcher():
        for i in range(10):
            for j in range(10):
                p = (i + 1) * 0.1
                k = (j + 1) * 0.1
                print("p = " + str(p))
                print("k = " + str(k))
                vi = alg.ValueIteration()
                vi.launcher(p, k, False, True)

    def analysis(self):
        file_ref = open("data_exp/val" + str('{0:1.1f}'.format(1)) + "_" +
                        str('{0:1.1f}'.format(1)) + ".txt", "r")
        data_ref = []

        for line in file_ref:
            data_ref.append([float(x) for x in line.split()])

        file_ref.close()

        ref_norm = 0
        for m in range(param.JackCarParam.max_cars() + 1):
            for n in range(param.JackCarParam.max_cars() + 1):
                ref_norm += pow(data_ref[m][n], 2)
        ref_norm = math.sqrt(ref_norm)


        for i in range(10):
            for j in range(10):
                p = (i + 1) * 0.1
                k = (j + 1) * 0.1
                file = open("data_exp/val" + str('{0:1.1f}'.format(p)) + "_" +
                                             str('{0:1.1f}'.format(k)) + ".txt", "r")
                data = []

                for line in file:
                    data.append([float(x) for x in line.split()])

                file.close()

                delta = 0

                for m in range(param.JackCarParam.max_cars() + 1):
                    for n in range(param.JackCarParam.max_cars() + 1):
                        delta += pow(data_ref[m][n] - data[m][n], 2)

                delta = math.sqrt(delta)
                time_delta = data[param.JackCarParam.max_cars() + 1][0] - data_ref[param.JackCarParam.max_cars() + 1][0]

                self.abs_val_delta[i][j] = delta
                self.rel_val_delta[i][j] = delta / ref_norm
                self.abs_time_delta[i][j] = time_delta
                self.rel_time_delta[i][j] = time_delta / data_ref[param.JackCarParam.max_cars() + 1][0]

    @staticmethod
    def plot(data, data_label):
        axis_labels = ['{0:1.1f}'.format((i + 1) * 0.1) for i in range(10)]
        plt.figure(figsize=(18, 18))
        hm = sns.heatmap(data,
                         annot=True,
                         square = True,
                         linewidth=0.5,
                         xticklabels=axis_labels, yticklabels=axis_labels)
        hm.set_xlabel('k', fontsize=10)
        hm.set_ylabel('p', fontsize=10)
        hm.invert_yaxis()
        plt.savefig(data_label + '.svg')
        plt.close()

    def plot_analysis(self):
        self.plot(self.abs_val_delta, 'abs_val_delta')
        self.plot(self.rel_val_delta, 'rel_val_delta')
        self.plot(self.abs_time_delta, 'abs_time_delta')
        self.plot(self.rel_time_delta, 'rel_time_delta')




