import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import numpy as np
from scipy.stats import multivariate_normal

import time

import pydrake.symbolic as sym
import pydrake.common

'''
Analyzes Animal Crossing: New Horizons turnip pricing with sympy.
'''

class Rollout:
    def __init__(self, rollout_probability, pattern_name, prices):
        self.rollout_probability = rollout_probability
        self.pattern_name = pattern_name
        self.prices = prices
        self.means = None
        self.vars = None

    def __repr__(self):
        s = "%s Rollout (weighted %0.3f): " % (self.pattern_name, self.rollout_probability)
        s += "\n\t" + str(self.prices) + "\n"
        return s

    def fit_normal(self, num_samples):
        g = pydrake.common.RandomGenerator()
        v = [sym.Evaluate(self.prices, generator=g) for i in range(num_samples)]
        self.v = np.stack(v).reshape(num_samples, 12)
        self.means = np.mean(self.v, axis=0)
        self.vars = np.var(self.v, axis=0)

    def evaluate_pdf(self, x):
        assert(len(x) == 12)
        # Values of X can be float or None
        total_density = 1.
        for k, val in enumerate(x):
            if val is not None:
                total_density *= multivariate_normal.pdf(val, mean=self.means[k], cov=self.vars[k])
        return total_density

uniform_variable_store = {}
def random_uniform(name, low, high):
    if name not in uniform_variable_store.keys():
        var = sym.Variable(name, sym.Variable.Type.RANDOM_UNIFORM)
        uniform_variable_store[name] = (var, low, high)
    else:
        var, lows, highs = uniform_variable_store[name]
    return var*(high-low)+low

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
    return []#Rollout(rollout_probability, "Random", None)]

def generate_large_spike_pattern_rollouts(base_price, rollout_probability):
    out = []
    for peak_start in range(1, 8):
        prices = []
        reduced_prob = rollout_probability / 7.
        rate = random_uniform("large_spike_initial_rate", 0.85, 0.9)
        for k in range(peak_start):
            prices.append(rate * base_price)
            rate -= 0.03
            rate -= random_uniform("large_spike_dec_%d" % k, 0., 0.02)

        prices.append(random_uniform("large_spike_rate_1", 0.9, 1.4) * base_price)
        prices.append(random_uniform("large_spike_rate_2", 1.4, 2.0) * base_price)
        prices.append(random_uniform("large_spike_rate_3", 2.0, 6.0) * base_price)
        prices.append(random_uniform("large_spike_rate_4", 1.4, 2.0) * base_price)
        prices.append(random_uniform("large_spike_rate_5", 0.9, 1.4) * base_price)
        for k in range(12 - peak_start - 5):
            prices.append(random_uniform("large_spike_fin_rate_%d" % k, 0.4, 0.9) * base_price)
        out.append(Rollout(reduced_prob, "Large Spike", prices))
    return out

def generate_decreasing_pattern_rollouts(base_price, rollout_probability):
    prices = []
    rate = 0.9
    rate -= random_uniform("decreasing_dec_init", 0.0, 0.05)
    for k in range(12):
        prices.append(rate * base_price)
        rate -= 0.03
        rate -= random_uniform("decreasing_dec_%d" % k, 0.0, 0.02)
    return [Rollout(rollout_probability, "Decreasing", np.array(prices))]

def generate_small_spike_pattern_rollouts(base_price, rollout_probability):
    out = []
    for peak_start in range(8):
        prices = []
        reduced_prob = rollout_probability / 8.
        rate = random_uniform("small_spike_initial_rate", 0.4, 0.9)
        for k in range(peak_start):
            prices.append(rate * base_price)
            rate -= 0.03
            rate -= random_uniform("small_spike_dec_%d" % k, 0., 0.02)

        prices.append(random_uniform("small_spike_rate_1", 0.9, 1.4) * base_price)
        prices.append(random_uniform("small_spike_rate_2", 0.9, 1.4) * base_price)
        peak_rate = random_uniform("small_spike_rate_spike_ub", 1.4, 2.0)
        prices.append(random_uniform("small_spike_rate_pre_spike", 1.4, peak_rate) * base_price - 1)
        prices.append(peak_rate * base_price)
        prices.append(random_uniform("small_spike_rate_post_spike", 1.4, peak_rate) * base_price - 1)
        post_rate = random_uniform("small_spike_rate_final", 0.4, 0.9)
        for k in range(12 - peak_start - 5):
            prices.append(post_rate * base_price)
            post_rate -= 0.03
            post_rate -= random_uniform("small_spike_final_dec_%d" % k, 0.0, 0.02)
        out.append(Rollout(reduced_prob, "Small Spike", prices))
    return out


pattern_rollout_generators = [
    generate_random_pattern_rollouts,
    generate_large_spike_pattern_rollouts,
    generate_decreasing_pattern_rollouts,
    generate_small_spike_pattern_rollouts
]

def generate_all_turnip_patterns(previous_pattern_prior, base_price=None):
    assert(previous_pattern_prior.shape == (4,))
    assert(np.allclose(np.sum(previous_pattern_prior), 1.))

    all_rollouts = []

    # Random integer base price on [90, 110]
    if base_price is None:
        base_price = random_uniform("base_price", 90, 110)
    else:
        base_price = base_price
    print("generating with base price ", base_price)

    # Probability of being in each pattern:
    pattern_probs = np.dot(pattern_transition_matrix, previous_pattern_prior)
    for next_pattern_k in range(4):
        all_rollouts += pattern_rollout_generators[next_pattern_k](
            base_price=base_price,
            rollout_probability=pattern_probs[next_pattern_k])

    return all_rollouts

if __name__ == "__main__":
    #observed = [None, None,  # M
    #        None, None,  # T
    #        None, None,  # W
    #        None, None,  # R
    #        None, None,  # F
    #        None, None]  # S

    observed = [66, 61,  # M
            58, 53,  # T
            48, 114,  # W
            122, 137,  # R
            None, None,  # F
            None, None]  # S
    base_price = 109

    previous_pattern_prior = generate_previous_pattern_steady_state()
    all_rollouts = generate_all_turnip_patterns(previous_pattern_prior, base_price=base_price)
    
    g = pydrake.common.RandomGenerator()
    
    start = time.time()
    num_samples = 1000
    colors_by_type = {"Large Spike": np.array([0., 0., 1]), "Decreasing": np.array([1., 0., 0.]),
                     "Small Spike": np.array([0., 1., 0.]), "Random": np.array([1., 0., 1.]),
                     "Average Prediction": np.array([0.1, 0.1, 0.1])}
    colormaps_by_type = {"Large Spike": cm.get_cmap('winter'),
                         "Decreasing": cm.get_cmap('spring'),
                         "Small Spike": cm.get_cmap('summer'),
                         "Random": cm.get_cmap('cool')}

    # Fit a multivariate normal distribution to each rollout
    for rollout in all_rollouts:
        rollout.fit_normal(num_samples)

    # Adjust the probability of each rollout based on the normal dist
    for rollout in all_rollouts:
        density = rollout.evaluate_pdf(observed)
        rollout.rollout_probability *= density
        print("New prob: ", rollout.rollout_probability)
    
    # Normalize probabilities across remaining rollouts
    total_prob = sum([rollout.rollout_probability for rollout in all_rollouts])
    valid_rollouts = []
    for rollout in all_rollouts:
        rollout.rollout_probability /= total_prob
        if rollout.rollout_probability > 1E-4:
            valid_rollouts.append(rollout)

    # Combine the valid rollouts into a final model over prices
    mean = np.zeros(12)
    var = np.zeros(12)
    for rollout in valid_rollouts:
        print("Prob: ", rollout.rollout_probability)
        mean += rollout.means * rollout.rollout_probability
        var += rollout.rollout_probability**2. * rollout.vars
    for k, val in enumerate(observed):
        if val is not None:
            mean[k] = val
            var[k] = 0.
    print("Final mean and var: ", mean, var)

    # Finally, draw it
    plt.figure()
    plt.grid(True)
    for k, rollout in enumerate(valid_rollouts):
        if rollout.rollout_probability > 1E-3:
            plt.plot(range(12), rollout.means,
                     color=colors_by_type[rollout.pattern_name],
                     alpha=0.5, linestyle="--")
            plt.fill_between(
                range(12), rollout.means-1.96*np.sqrt(rollout.vars), rollout.means+1.96*np.sqrt(rollout.vars),
                alpha=0.125, edgecolor=colors_by_type[rollout.pattern_name],
                facecolor=colormaps_by_type[rollout.pattern_name](float(k)/len(valid_rollouts)))

    plt.plot(range(12), mean, 'k', color=colors_by_type["Average Prediction"])
    plt.fill_between(range(12), mean-1.96*np.sqrt(var), mean+1.96*np.sqrt(var),
        alpha=0.25, edgecolor=colors_by_type["Average Prediction"], facecolor=[0.25, 0.1, 0.25])

    label_lines = [Line2D([0], [0], color=colors_by_type[pattern_name], lw=4) for pattern_name in list(colors_by_type.keys())]
    plt.legend(label_lines, list(colors_by_type.keys()))

    #plt.figure()
   # bins = np.arange(1., 660, 10.)
   # #plt.figure()
   # rollout_hists = {}
   # for rollout_type_k, rollout_type in enumerate(pattern_index_to_name):
   #     #plt.subplot(4, 1, rollout_type_k+1)
   #     #plt.grid(True)
   #     vs = []
   #     ws = []
   #     for k, rollout in enumerate(all_rollouts):
   #         if rollout.pattern_name == rollout_type:
   #             v = []
   #             w = []
   #             for t in range(12):
   #                 g = pydrake.common.RandomGenerator(k)
   #                 v.append([rollout.prices[t].Evaluate(g) for i in range(num_samples)])
   #                 w.append([rollout.rollout_probability]*num_samples)
   #             vs.append(np.vstack(v))
   #             ws.append(np.stack(w))
   #     if len(vs) == 0:
   #         continue
   #     vs = np.hstack(vs).T
   #     ws = np.hstack(ws).T
   #     # Hist per column
   #     v_hist = [np.histogram(vs[:, k], bins=bins, density=True, weights=ws[:, k])[0] for k in range(12)]
   #     v_hist = [v / np.sum(v) for v in v_hist] # Normalize columns
   #     v_hist = np.vstack(v_hist).T
   #     rollout_hists[rollout_type] = v_hist
   #     #plt.imshow(1. - v_hist, origin='lower', extent=[1, 12, 0, 660], aspect='auto')
   #     #plt.hexbin(np.tile(np.array(range(12)), (vs.shape[0], 1)), vs)
   #     #plt.title(rollout_type)


    # Actually evaluate rollouts and get sequential info
    #plt.figure()
    #for rollout_type_k, rollout_type in enumerate(pattern_index_to_name):
    #    plt.subplot(4, 1, rollout_type_k+1)
    #    plt.grid(True)
    #    these_rollouts = [rollout for rollout in all_rollouts if rollout.pattern_name == rollout_type]
    #    for k, rollout in enumerate(these_rollouts):
    #        if rollout.pattern_name == rollout_type:
    #            g = pydrake.common.RandomGenerator(k)
    #            v = [sym.Evaluate(np.array(rollout.prices), generator=g) for i in range(num_samples)]
    #            v = np.hstack(v)
    #            plt.plot(range(12), v, color=colormaps_by_type[rollout_type](k / len(these_rollouts)), linewidth=1, alpha=0.05)
    #    plt.title(rollout_type)

    # Independent scatters per type
    #for k in range(12):
    #    # For each rollout...
    #    samples_by_type = {"Large Spike": [], "Decreasing": [], "Small Spike": [], "Random": []}
    #    weights_by_type = {"Large Spike": [], "Decreasing": [], "Small Spike": [], "Random": []}
    #    for rollout in all_rollouts:
    #        samples_by_type[rollout.pattern_name] += [rollout.prices[k].Evaluate(g) for i in range(num_samples)]
    #        weights_by_type[rollout.pattern_name] += [rollout.rollout_probability] * num_samples
    #    for offset, pattern_name in enumerate(pattern_index_to_name):
    #        samples = samples_by_type[pattern_name]
    #        weights = weights_by_type[pattern_name]
    #        plt.scatter([k + offset*0.2]*len(samples), samples, color=colors_by_type[pattern_name], alpha=0.01)
    #label_lines = [Line2D([0], [0], color=colors_by_type[pattern_name], lw=4) for pattern_name in pattern_index_to_name]
    #plt.legend(label_lines, pattern_index_to_name)
    
    print("Elapsed: ", time.time() - start)
    plt.show()