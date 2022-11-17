import game.parameters as param
import game.destributions as dest
import numpy as np


class Location:

    def __init__(self, req, ret):
        self.a = req  # value of lambda for requests
        self.b = ret  # value of lambda for returns
        self.poisson_a = dest.Poisson(self.a)
        self.poisson_b = dest.Poisson(self.b)


class Model:
    def __init__(self):
        self.value = np.zeros((param.JackCarParam.max_cars() + 1, param.JackCarParam.max_cars() + 1))
        self.policy = self.value.copy().astype(int)
        self.A = Location(3, 3)
        self.B = Location(4, 2)

    @staticmethod
    def apply_action(state, action):
        return [max(min(state[0] - action, param.JackCarParam.max_cars()), 0),
                max(min(state[1] + action, param.JackCarParam.max_cars()), 0)]

    def expected_reward(self, state, action):
        """
        state  : It's a pair of integers, # of cars at A and at B
        action : # of cars transferred from A to B,  -5 <= action <= 5
        """

        psi = 0  # reward
        new_state = self.apply_action(state, action)

        # adding reward for moving cars from one location to another (which is negative)

        psi = psi + param.JackCarParam.moving_reward() * abs(action)

        # there are four discrete random variables which determine the probability distribution of the reward and
        # next state

        # print(self.A.poisson_a.vals)
        # a = input()

        for Aa in range(self.A.poisson_a.a, self.A.poisson_a.b):
            for Ba in range(self.B.poisson_a.a, self.B.poisson_a.b):
                for Ab in range(self.A.poisson_b.a, self.A.poisson_b.b):
                    for Bb in range(self.B.poisson_b.a, self.B.poisson_b.b):
                        """
                        Aa : sample of cars requested at location A
                        Ab : sample of cars returned at location A
                        Ba : sample of cars requested at location B
                        Bb : sample of cars returned at location B
                        xi  : probability of this event happening
                        """

                        # all four variables are independent of each other
                        xi = self.A.poisson_a.vals[Aa] * self.B.poisson_a.vals[Ba] * self.A.poisson_b.vals[Ab] * self.B.poisson_b.vals[Bb]

                        valid_requests_A = min(new_state[0], Aa)
                        valid_requests_B = min(new_state[1], Ba)

                        rew = (valid_requests_A + valid_requests_B) * (param.JackCarParam.credit_reward())

                        # calculating the new state based on the values of the four random variables
                        new_s = [0, 0]
                        new_s[0] = max(min(new_state[0] - valid_requests_A + Ab, param.JackCarParam.max_cars()), 0)
                        new_s[1] = max(min(new_state[1] - valid_requests_B + Bb, param.JackCarParam.max_cars()), 0)

                        # Bellman's equation
                        # print(new_s[0])
                        psi += xi * (rew + param.JackCarParam.gamma() * self.value[new_s[0]][new_s[1]])
                        # print(psi)

        return psi
