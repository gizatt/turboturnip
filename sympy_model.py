import matplotlib.pyplot as plt
import numpy as np

from sympy.stats import Uniform, density, sample
from sympy.functions.elementary.piecewise import Piecewise
from sympy import Symbol, lambdify

'''
Analyzes Animal Crossing: New Horizons turnip pricing with sympy.
'''

p = Symbol("p")

class Rollout:
    def __init__(self, rollout_probability, pattern_name, prices):
        self.rollout_probability = rollout_probability
        self.pattern_name = pattern_name
        self.prices = prices
    def __repr__(self):
        s = "%s Rollout (weighted %0.3f): " % (self.pattern_name, self.rollout_probability)
        s += "\n\t" + str(self.prices) + "\n"
        return s

uniform_sym_dict = {}
def get_uniform_sym(name, low, high):
    if name in uniform_sym_dict.keys():
        return uniform_sym_dict[name][0]
    else:
        sym = Symbol(name)
        uniform_sym_dict[name] = (sym, low, high)
        return sym

pattern_index_to_name = [
    "Random",
    "Large Spike",
    "Decreasing",
    "Small Spike"
]
pattern_transition_matrix = np.array(
    [[0.2, 0.3, 0.15, 0.35],
     [0.5, 0.05, 0.2, 0.25],
     [0.25, 0.45, 0.05, 0.25],
     [0.45, 0.25, 0.15, 0.15]]).T

def generate_previous_pattern_steady_state():
    # The prior over patterns is the steady state of
    # the transition matrix
    w, v = np.linalg.eig(pattern_transition_matrix)
    assert(np.allclose(w[0], 1.))
    prior_over_patterns = v[:, 0]
    return prior_over_patterns / np.sum(prior_over_patterns)

def generate_random_pattern_rollouts(base_price, rollout_probability):
    return [] #Rollout(rollout_probability, "Random", None)]

def generate_large_spike_pattern_rollouts(base_price, rollout_probability):
    return [] #Rollout(rollout_probability, "Large Spike", None)]

def generate_decreasing_pattern_rollouts(base_price, rollout_probability):
    prices = []
    rate = 0.9
    rate -= Uniform("dec_rate_init", 0.0, 0.05)
    for k in range(12):
        prices.append(rate * base_price)
        rate -= 0.03
        rate -= Uniform("dec_rate_%d" % k, 0.0, 0.02)
    return [Rollout(rollout_probability, "Decreasing", np.array(prices))]

def generate_small_spike_pattern_rollouts(base_price, rollout_probability):
    return [] #Rollout(rollout_probability, "Small Spike", None)]

pattern_rollout_generators = [
    generate_random_pattern_rollouts,
    generate_large_spike_pattern_rollouts,
    generate_decreasing_pattern_rollouts,
    generate_small_spike_pattern_rollouts
]

def generate_all_turnip_patterns(previous_pattern_prior):
    assert(previous_pattern_prior.shape == (4,))
    assert(np.allclose(np.sum(previous_pattern_prior), 1.))

    all_rollouts = []

    # Random integer base price on [90, 110]
    base_price = Uniform("base_price", 90., 110.)

    # Probability of being in each pattern:
    pattern_probs = np.dot(pattern_transition_matrix, previous_pattern_prior)
    for next_pattern_k in range(4):

        all_rollouts += pattern_rollout_generators[next_pattern_k](
            base_price=base_price,
            rollout_probability=pattern_probs[next_pattern_k])

    return all_rollouts

def make_uniform_density(p, low, high):
    return Piecewise((1. / (high - low), ((p >= low) & (p <= high))),
                     (0., True))

if __name__ == "__main__":
    previous_pattern_prior = generate_previous_pattern_steady_state()
    all_rollouts = generate_all_turnip_patterns(previous_pattern_prior)
    
    rollout_1 = all_rollouts[0]
    
    plt.figure()
    for k in range(12):
        samples = [sample(rollout_1.prices[k]) for i in range(1000)]
        plt.scatter([k]*1000, samples)
    plt.show()

