import matplotlib.pyplot as plt
import seaborn as sns


class Plotter:
    def __init__(self):
        self.saved_policy_counter = 0
        self.saved_value_counter = 0

    def save_policy(self, policy):
        self.saved_policy_counter += 1
        ax = sns.heatmap(policy, linewidth=0.5)
        ax.invert_yaxis()
        plt.savefig('policy' + str(self.saved_policy_counter) + '.svg')
        plt.close()

    def save_value(self, value):
        self.saved_value_counter += 1
        ax = sns.heatmap(value, linewidth=0.5)
        ax.invert_yaxis()
        plt.savefig('value' + str(self.saved_value_counter) + '.svg')
        plt.close()
