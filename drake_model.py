import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.cm as cm
import numpy as np
from scipy import interpolate
from scipy.stats import multivariate_normal, gaussian_kde
from scipy.special import logsumexp
from sklearn import mixture

import time
import sys

from pydrake.solvers.ipopt import IpoptSolver
from pydrake.solvers.snopt import SnoptSolver
from pydrake.solvers.osqp import OsqpSolver
from pydrake.solvers.mathematicalprogram import MathematicalProgram, Solve
import pydrake.symbolic as sym
import pydrake.common
from pydrake.common.containers import EqualToDict

'''
Analyzes Animal Crossing: New Horizons turnip pricing with sympy.


Optimization notes:

Every rollout type, with one exception, has each price linear
in the uniformly distributed variables. The exception is the
small spike, in which the immediately pre and post-spike values
have a bilinear term 

    small_spike_rate_spike_ub * small_spike_rate_pre_spike = C_N-1
    small_spike_rate_spike_ub * small_spike_rate_post_spike = C_N+1

When the true peak is observed, small_spike_rate_spike_ub is
directly observable as

    small_spike_rate_spike_ub  * base_price = C_N

but I can't guarantee both the off-peak /and/ peak terms will
be observed...

'''

class Rollout:
    def __init__(self, rollout_probability, pattern_name, prices):
        self.rollout_probability = rollout_probability
        self.pattern_name = pattern_name
        self.prices = prices
        self.means = None
        self.vars = None
        self.model_type = None
        self.feasible_vals = None

    def __repr__(self):
        s = "%s Rollout (weighted %0.3f): " % (self.pattern_name, self.rollout_probability)
        s += "\n\t" + str(self.prices) + "\n"
        return s

    def find_feasible_latents(self, observed):
        # Build an optimization to estimate the hidden variables
        try:    
            prog = MathematicalProgram()
            # Add in all the appropriate variables with their bounds
            all_vars = self.prices[0].GetVariables()
            for price in self.prices[1:]:
                all_vars += price.GetVariables()
            mp_vars = prog.NewContinuousVariables(len(all_vars))
            subs_dict = {}
            for v, mv in zip(all_vars, mp_vars):
                subs_dict[v] = mv
            lb = []
            ub = []
            prog.AddBoundingBoxConstraint(0., 1., mp_vars)
            prices_mp = [self.prices[k].Substitute(subs_dict) for k in range(12)]
            # Add the observation constraint
            for k, val in enumerate(observed[1:]):
                if val != 0:
                    prog.AddConstraint(prices_mp[k] >= val - 2.)
                    prog.AddConstraint(prices_mp[k] <= val + 2)

            # Find lower bounds
            prog.AddCost(sum(prices_mp))
            solver = SnoptSolver()
            result = solver.Solve(prog)

            if result.is_success():
                lb = [result.GetSolution(x).Evaluate() for x in prices_mp]
                lb_vars = result.GetSolution(mp_vars)
                # Find upper bound too
                prog.AddCost(-2.*sum(prices_mp))
                result = solver.Solve(prog)
                if result.is_success():
                    ub_vars = result.GetSolution(mp_vars)
                    ub = [result.GetSolution(x).Evaluate() for x in prices_mp]
                    self.price_ub = ub
                    self.price_lb = lb
                    subs_dict = {}
                    for k, v in enumerate(all_vars):
                        if lb_vars[k] == ub_vars[k]:
                            subs_dict[v] = lb_vars[k]
                        else:
                            new_var = sym.Variable("feasible_%d" % k, sym.Variable.Type.RANDOM_UNIFORM)
                            subs_dict[v] = new_var * (ub_vars[k] - lb_vars[k]) + lb_vars[k]
                    self.prices = [self.prices[k].Substitute(subs_dict) for k in range(12)]
                    return

        except RuntimeError as e:
            print("Runtime error: ", e)
        self.rollout_probability = 0.
        
    def do_rollouts(self, num_samples):
        g = pydrake.common.RandomGenerator()
        v = [sym.Evaluate(self.prices, generator=g) for i in range(num_samples)]
        self.v = np.stack(v).reshape(num_samples, 12)

    def fit_model(self, num_samples, model_type="gmm", model_params={}):
        self.do_rollouts(num_samples)

        self.model_type = model_type
        if model_type == "gmm":
            assert("n_components" in model_params.keys())
            n_components = model_params["n_components"]
            self.g = mixture.GaussianMixture(n_components=n_components, covariance_type='diag')
            self.g.fit(self.v)
        elif model_type == "kde":
            assert("bw_method" in model_params)
            self.kdes = [gaussian_kde(self.v[:, k], bw_method=model_params["bw_method"]) for k in range(12)]
        else:
            raise NotImplementedError("Model type %s" % model_type)
    
    def evaluate_componentwise_pdf(self, vals):
        assert(self.model_type is not None)
        assert(self.model_type == "gmm")
        scores = []
        for k in range(12):
            scores.append(
                sum([multivariate_normal.pdf(vals, self.g.means_[comp_k, k],
                     self.g.covariances_[comp_k, k])*self.g.weights_[comp_k]
                        for comp_k in range(len(self.g.weights_))]))
        return np.stack(scores)

    def evaluate_logpdf(self, x):
        assert(self.model_type is not None)
        assert(len(x) == 13)
        # Values of X can be float or None
        if self.model_type == "gmm":
            total_log_density = 0.
            logscores_by_comp = self.g._estimate_log_prob(np.array(x[1:]).reshape(1, -1)).flatten()
            log_weights = self.g._estimate_log_weights().flatten()
            # Correct those scores for unobserved points, which we can do because
            # of the diagonal covariances
            for k, val in enumerate(x[1:]):
                if val == 0:
                    for comp_k in range(len(logscores_by_comp)):
                        logscores_by_comp[comp_k] -= multivariate_normal.logpdf(
                            val, self.g.means_[comp_k, k],
                            self.g.covariances_[comp_k, k])
            return logsumexp(logscores_by_comp + log_weights)
        elif self.model_type == "kde":
            total_log_density = 0.
            for k, val in enumerate(x[1:]):
                if val != 0:
                    total_log_density += self.kdes[k].logpdf(val)
            return total_log_density

uniform_variable_store = {}
uniform_variable_to_range = EqualToDict()
def random_uniform(name, low, high):
    if name not in uniform_variable_store.keys():
        var = sym.Variable(name, sym.Variable.Type.RANDOM_UNIFORM)
        uniform_variable_store[name] = var
        uniform_variable_to_range[var] = (low, high)
    else:
        var = uniform_variable_store[name]
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

colors_by_type = {"Large Spike": np.array([0., 0., 1]), "Decreasing": np.array([1., 0., 0.]),
                 "Small Spike": np.array([0., 1., 0.]), "Random": np.array([1., 0., 1.]),
                 "Average Prediction": np.array([0.1, 0.1, 0.1])}
colormaps_by_type = {"Large Spike": cm.get_cmap('winter'),
                     "Decreasing": cm.get_cmap('autumn'),
                     "Small Spike": cm.get_cmap('summer'),
                     "Random": cm.get_cmap('cool')}

def generate_previous_pattern_steady_state():
    # The prior over patterns is the steady state of
    # the transition matrix
    w, v = np.linalg.eig(pattern_transition_matrix)
    assert(np.allclose(w[0], 1.))
    prior_over_patterns = v[:, 0]
    return prior_over_patterns / np.sum(prior_over_patterns)

def generate_random_pattern_rollouts(base_price, rollout_probability):
    # Random is in 5 phases: inc, dec, inc, dec, inc
    # Lengths add up to 12
    # Dec_1 is [2,3] long
    # Dec_2 is 5 - Dec_1 long
    # Inc_1 is [0-6] long
    # Inc_2and3 is 7 - Inc_1 long
    # Inc_3 is [0, Inc_2and3] long
    # Inc_2 is Inc_2and3 - Inc_3 long
    out = []
    for inc_1_length in range(7):
        inc_2and3_length = 7 - inc_1_length
        for inc_3_length in range(0, inc_2and3_length):
            inc_2_length = inc_2and3_length - inc_3_length
            for dec_1_length in [2, 3]:
                reduced_prob = 1. / 7. / inc_2and3_length / 2.
                dec_2_length = 5 - dec_1_length

                # Actual generation of this rollout
                prices = []
                
                # INC 1
                for k in range(inc_1_length):
                    prices.append(random_uniform("inc_1_rate_%d" % k, 0.9, 1.4) * base_price)
                
                # DEC 1
                rate = random_uniform("dec_1_initial_rate", 0.6, 0.8)
                for k in range(dec_1_length):
                    prices.append(rate * base_price)
                    rate -= 0.04
                    rate -= random_uniform("dec_1_dec_%d" % k, 0.0, 0.06)

                # INC 2
                for k in range(inc_2_length):
                    prices.append(random_uniform("inc_2_rate_%d" % k, 0.9, 1.4) * base_price)
                
                # DEC 2
                rate = random_uniform("dec_2_initial_rate", 0.6, 0.8)
                for k in range(dec_2_length):
                    prices.append(rate * base_price)
                    rate -= 0.04
                    rate -= random_uniform("dec_2_dec_%d" % k, 0.0, 0.06)

                # INC 3
                for k in range(inc_3_length):
                    prices.append(random_uniform("inc_3_rate_%d" % k, 0.9, 1.4) * base_price)

                out.append(Rollout(reduced_prob, "Random", prices))
    return out

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
    if base_price == 0:
        base_price = random_uniform("base_price", 90, 110)
    else:
        base_price = base_price
    
    # Probability of being in each pattern:
    pattern_probs = np.dot(pattern_transition_matrix, previous_pattern_prior)
    for next_pattern_k in range(4):
        all_rollouts += pattern_rollout_generators[next_pattern_k](
            base_price=base_price,
            rollout_probability=pattern_probs[next_pattern_k])

    return all_rollouts

def plot_results(valid_rollouts, observed, num_samples=0, subplots=False):
    # Finally, draw it
    plt.grid(True)

    if subplots:
        for k in range(4):
            plt.subplot(5, 1, k+1)
            plt.title(pattern_index_to_name[k])
            plt.grid(True)

    for k, rollout in enumerate(valid_rollouts):
        if subplots:
            plt.subplot(5, 1, pattern_index_to_name.index(rollout.pattern_name) + 1)
        if num_samples > 0:
            rollout.do_rollouts(num_samples=num_samples)
            plt.plot(range(12), rollout.v.T, color=colormaps_by_type[rollout.pattern_name](k / len(valid_rollouts)), linewidth=2, alpha=0.05)
        plt.fill_between(
            range(12), rollout.price_lb, rollout.price_ub,
            alpha=0.5, edgecolor=colors_by_type[rollout.pattern_name],
            facecolor=colormaps_by_type[rollout.pattern_name](float(k)/len(valid_rollouts)))

    ks = []
    vals = []
    for k, val in enumerate(observed[1:]):
        if val != 0:
            ks.append(k)
            vals.append(val)  

    for k in range(4):
        if subplots:
            plt.subplot(5, 1, k+1)
        plt.scatter(ks, vals, color="k")
        if not subplots:
            break
    if not subplots:
        label_lines = [Line2D([0], [0], color=colors_by_type[pattern_name], lw=4) for pattern_name in list(colors_by_type.keys())]
        plt.legend(label_lines, list(colors_by_type.keys()))

def do_analysis(observed):
    previous_pattern_prior = generate_previous_pattern_steady_state()
    all_rollouts = generate_all_turnip_patterns(previous_pattern_prior, observed[0])

    # Fit a model to reach rollout
    for rollout in all_rollouts:
        rollout.find_feasible_latents(observed)
        #rollout.fit_model(
        #    num_samples,
        #    model_type=model_type,
        #    model_params=model_params)

    # Normalize probabilities across remaining rollouts
    # These numbers are tame enough that we don't realy need log prob messiness
    total_prob = np.sum([rollout.rollout_probability for rollout in all_rollouts])
    valid_rollouts = []
    total_prob_of_types = {}
    for rollout in all_rollouts:
        rollout.rollout_probability /= total_prob
        print("Prob of type %s: " % rollout.pattern_name, rollout.rollout_probability)
        if rollout.pattern_name not in total_prob_of_types.keys():
            total_prob_of_types[rollout.pattern_name] = rollout.rollout_probability
        else:
            total_prob_of_types[rollout.pattern_name] += rollout.rollout_probability
        if rollout.rollout_probability > 0.001:
            valid_rollouts.append(rollout)
    print("Total prob of types: ", total_prob_of_types)
    return valid_rollouts



if __name__ == "__main__":
    observed = [0, # base price, was 109, but first week functional buy price is random
            66, 61,  # M
            58, 53,  # T
            48, 114,  # W
            122, 137,  # R
            138, 136,  # F
            0, 0]  # S
    subplots = False
    num_samples = 100
    #observed = [109, # base price
    #        98, 94,  # M
    #        91, 86,  # T
    #        82, 78,  # W
    #        0, 0,  # R
    #        0, 0,  # F
    #        0, 0]  # S


    #observed_data = np.loadtxt("example_data.csv", dtype=int, delimiter=",", skiprows=1, usecols=range(1, 1+13))
    #observed = observed_data[39, :]
    print("Observed: ", observed)
    plt.figure(dpi=300).set_size_inches(12, 12)
    for k in range(13):
        sub_observed = [0] * 13
        sub_observed[0] = observed[0]
        sub_observed[1:(1+k)] = observed[1:(1+k)]
        print(sub_observed)
        valid_rollouts = do_analysis(sub_observed)
        plt.gca().clear()
        plot_results(valid_rollouts, sub_observed, num_samples=num_samples, subplots=subplots)
        plt.ylim([0, 660])
        plt.title("With %d observations" % sum([x != 0 for x in sub_observed[1:]]))
        plt.savefig("%d.jpg" % k)
    

    #plt.figure()
    #bins = np.arange(1., 660, 10.)
    ##plt.figure()
    #rollout_hists = {}
    #for rollout_type_k, rollout_type in enumerate(pattern_index_to_name):
    #    plt.subplot(4, 1, rollout_type_k+1)
    #    plt.grid(True)
    #    vs = []
    #    ws = []
    #    for k, rollout in enumerate(all_rollouts):
    #        if rollout.pattern_name == rollout_type:
    #            v = []
    #            w = []
    #            for t in range(12):
    #                g = pydrake.common.RandomGenerator(k)
    #                v.append([rollout.prices[t].Evaluate(g) for i in range(num_samples)])
    #                w.append([rollout.rollout_probability]*num_samples)
    #            vs.append(np.vstack(v))
    #            ws.append(np.stack(w))
    #    if len(vs) == 0:
    #        continue
    #    vs = np.hstack(vs).T
    #    ws = np.hstack(ws).T
    #    # Hist per column
    #    v_hist = [np.histogram(vs[:, k], bins=bins, density=True, weights=ws[:, k])[0] for k in range(12)]
    #    v_hist = [v / np.sum(v) for v in v_hist] # Normalize columns
    #    v_hist = np.vstack(v_hist).T
    #    rollout_hists[rollout_type] = v_hist
    #    plt.imshow(1. - v_hist, origin='lower', extent=[1, 12, 0, 660], aspect='auto')
    #    #plt.hexbin(np.tile(np.array(range(12)), (vs.shape[0], 1)), vs)
    #    #plt.title(rollout_type)


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
    #            plt.plot(range(12), v, color=colormaps_by_type[rollout_type](k / len(these_rollouts)), linewidth=2, alpha=0.05)
    #    plt.title(rollout_type)

    # Independent scatters per type
    #plt.figure()
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

    #plt.show()