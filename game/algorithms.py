import random
import time
import game.jack_car_rental_model as jcrm
import game.parameters as param
import game.utilits.plotter as plotter
import game.utilits.experiment_saver as exp
import sys


class PolicyIteration:
    def __init__(self):
        self.model = jcrm.Model()
        self.eps = 5

    def policy_evaluation(self):
        self.eps /= 10

        while 1:
            delta = 0

            for i in range(self.model.value.shape[0]):
                for j in range(self.model.value.shape[1]):
                    # self.model.value[i][j] denotes the self.model.value of the state [i,j]

                    old_val = self.model.value[i][j]
                    self.model.value[i][j] = self.model.expected_reward([i, j], self.model.policy[i][j])

                    delta = max(delta, abs(self.model.value[i][j] - old_val))
                    print('.', end='')
                    # print(delta)
                    sys.stdout.flush()
            print(delta)
            sys.stdout.flush()

            if delta < self.eps:
                break

    def policy_improvement(self):
        policy_stable = True
        for i in range(self.model.value.shape[0]):
            for j in range(self.model.value.shape[1]):
                old_action = self.model.policy[i][j]

                max_act_val = None
                max_act = None

                t12 = min(i, 5)  # if I have say 3 cars at the first location, then I can atmost move 3 from 1 to 2
                t21 = -min(j, 5)  # if I have say 2 cars at the second location, then I can atmost move 2 from 2 to 1

                for act in range(t21, t12 + 1):
                    sigma = self.model.expected_reward([i, j], act)
                    if max_act_val is None:
                        max_act_val = sigma
                        max_act = act
                    elif max_act_val < sigma:
                        max_act_val = sigma
                        max_act = act

                self.model.policy[i][j] = max_act

                if old_action != self.model.policy[i][j]:
                    policy_stable = False

        return policy_stable

    def launcher(self):
        plt = plotter.Plotter()
        while 1:
            self.policy_evaluation()
            stable = self.policy_improvement()
            plt.save_value(self.model.value)
            plt.save_policy(self.model.policy)
            if stable:
                break

class ValueIteration:
    def __init__(self):
        self.model = jcrm.Model()
        self.eps = 0.01

    def launcher(self, p, k, plot_is_on, save_experiment_is_on):
        # timer
        tic = time.perf_counter()

        # plotter
        if plot_is_on:
            plt = plotter.Plotter()

        while 1:
            delta = 0

            for i in range(self.model.value.shape[0]):
                for j in range(self.model.value.shape[1]):
                    # self.model.value[i][j] denotes the value of the state [i,j]

                    rand = random.random()
                    if rand > p:
                        continue

                    old_val = self.model.value[i][j]

                    # searching of max and min available action in state [i,j]
                    max_action = min(min(param.JackCarParam.max_move(), i), param.JackCarParam.max_move() - (
                                min(param.JackCarParam.max_move(), i) + j - param.JackCarParam.max_cars()))
                    min_action = -min(min(param.JackCarParam.max_move(), j), param.JackCarParam.max_move() - (
                                min(param.JackCarParam.max_move(), j) + i - param.JackCarParam.max_cars()))

                    max_ = 0
                    for n in range(min_action, max_action + 1):
                        if n == min_action:
                            max_ = self.model.expected_reward([i, j], n)
                        else:
                            buf = self.model.expected_reward([i, j], n)
                            if buf > max_:
                                max_ = buf

                    self.model.value[i][j] = max_

                    rand = random.random()
                    if rand < k:
                        delta = max(delta, abs(self.model.value[i][j] - old_val))
                    print('.', end='')
                    sys.stdout.flush()
            print(delta)
            sys.stdout.flush()

            # plotter
            if plot_is_on:
                plt.save_value(self.model.value)

            if delta < self.eps:
                break

        # argmax search
        for i in range(self.model.value.shape[0]):
            for j in range(self.model.value.shape[1]):
                max_action = min(min(param.JackCarParam.max_move(), i), param.JackCarParam.max_move() - (
                        min(param.JackCarParam.max_move(), i) + j - param.JackCarParam.max_cars()))
                min_action = -min(min(param.JackCarParam.max_move(), j), param.JackCarParam.max_move() - (
                        min(param.JackCarParam.max_move(), j) + i - param.JackCarParam.max_cars()))
                max_v_action = 0
                max_ = 0
                for n in range(min_action, max_action + 1):
                    if n == min_action:
                        max_ = self.model.expected_reward([i, j], n)
                        max_v_action = n
                    else:
                        buf = self.model.expected_reward([i, j], n)
                        if buf > max_:
                            max_ = buf
                            max_v_action = n

                self.model.policy[i, j] = max_v_action

        if plot_is_on:
            plt.save_policy(self.model.policy)

        # timer
        tac = time.perf_counter()

        if save_experiment_is_on:
            exp.save(p, k, self.model.value, tac - tic)
