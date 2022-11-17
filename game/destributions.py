from scipy.stats import poisson


class Poisson:

    def __init__(self, lamb):
        self.lamb = lamb

        # [a, b] is the range of n's for which the pmf value is above eps
        self.a = 0
        self.b = 0
        self.vals = {}

        eps = 0.01
        state = 1
        summer = 0

        while 1:
            if state == 1:
                temp = poisson.pmf(self.a, self.lamb)
                if temp <= eps:
                    self.a += 1
                else:
                    self.vals[self.a] = temp
                    summer += temp
                    self.b = self.a + 1
                    state = 2
            elif state == 2:
                temp = poisson.pmf(self.b, self.lamb)
                if temp > eps:
                    self.vals[self.b] = temp
                    summer += temp
                    self.b += 1
                else:
                    break

        # normalizing the pmf, values of n outside of [a, b] have pmf = 0
        for key in self.vals:
            self.vals[key] /= summer

    def f(self, n):
        try:
            ret_value = self.vals[n]
        except KeyError:
            ret_value = 0
        finally:
            return ret_value
