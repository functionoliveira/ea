import numpy as np
from numpy.random import permutation, rand

class PoolLast:
    def __init__(self, options):
        """
        Constructor
        :param options:to store (initially the probability is equals)
        :return:
        """
        size = len(options)
        assert size > 0

        self.options = np.copy(options)
        self.improvements = []
        self.count_calls = 0
        self.first = permutation(self.options).tolist()

        self.new = None
        self.improvements = dict(zip(options, [0] * size))

    def reset(self):
        self.first = permutation(self.options).tolist()
        self.new = None
        options = self.options
        size = len(options)
        self.improvements = dict(zip(options, [0] * size))

    def has_no_improvement(self):
        return np.all([value == 0 for value in self.improvements.values()])

    def get_new(self):
        """
        Get one of the options, following the probabilities
        :return: one of the stored object
        """
        # First time it returns all
        if self.first:
            return self.first.pop()

        if self.new is None:
            self.new = self.update_prob()

        return self.new

    def is_empty(self):
        counts = self.improvements.values()
        return np.all(counts == 0)

    def improvement(self, obj, account, freq_update, minimum=0.15):
        """
        Received how much improvement this object has obtained (higher is better), it only update
        the method improvements

        :param object:
        :param account: improvement obtained (higher is better), must be >= 0
        :param freq_update: Frequency of improvements used to update the ranking
        :return: None
        """
        if account < 0:
            return

        if obj not in self.improvements:
            raise Exception("Error, object not found in PoolProb")

        previous = self.improvements[obj]
        self.improvements[obj] = account
        self.count_calls += 1
 
        if self.first:
            return

        if not self.new:
            self.new = self.update_prob()
        elif account == 0 or account < previous:
            self.new = self.update_prob()

    def update_prob(self):
        """
        update the probabilities considering improvements value, following the equation
        prob[i] = Improvements[i]/TotalImprovements

        :return: None
        """
        
        if np.all([value == 0 for value in self.improvements.values()]):
            new_method = np.random.choice(list(self.improvements.keys()))
            return new_method

        # Complete the ranking
        indexes = np.argsort(self.improvements.values())
        posbest = indexes[-1]
        best = list(self.improvements.keys())[posbest]
        return best
    
class PoolInc:
    def __init__(self, options):
        """
        Constructor
        :param options:to store (initially the probability is equals)
        :return:
        """
        self.options = options
        self.cumProb = []
        self.improvements = []
        self.count_calls = 0

        if len(options) > 0:
            size = len(options)
            prob = np.ones(size) / float(size)
            self.cumProb = prob.cumsum()
            self.improvements = dict(zip(options, [0.0] * size))
            self.count_total = dict(zip(options, [0.0] * size))

    def get_prob(self):
        return self.cumProb

    def get_new(self):
        """
        Get one of the options, following the probabilities
        :return: one of the stored object
        """
        if not self.options:
            raise Exception("There is no object")

        r = rand()
        position = self.cumProb.searchsorted(r)
        return self.options[position]

    def values(self):
        """
        Return the different values
        :return:
        """
        return self.options[:]

    def improvement(self, obj, account):
        """
        Received how much improvement this obj has obtained (higher is better), it only update
        the method improvements

        :param obj:
        :param account: improvement obtained (higher is better)
        :param freq_update: Frequency of run used to update the ranking
        :return: None
        """
        if obj not in self.improvements:
            raise Exception("Error, obj not found in PoolProb")

        self.count_total[obj] += 1

        if account > 0:
            self.improvements[obj] += 1

    def update_prob(self):
        """
        update the probabilities considering improvements value, following the equation
        prob[i] = Improvements[i]/TotalImprovements

        :return: None
        """
        size = len(self.options)

        # Complete the ranking
        improvements = np.array(self.improvements.values())
        totals = np.array(self.count_total.values())
        assert (np.all(totals > 0))
        ratio = improvements / totals + 0.01

        total_ratio = float(ratio.sum())
        prob = ratio / total_ratio

        self.cumProb = prob.cumsum()

    