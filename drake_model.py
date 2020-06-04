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
            #for var in all_vars:
                #low, high = uniform_variable_to_range[var]
                #if not isinstance(low, float):
                #    low = low.Substitute(subs_dict)
                #if not isinstance(high, float):
                #    high = high.Substitute(subs_dict)
                #lb.append(low)
                #ub.append(high)
            prog.AddBoundingBoxConstraint(0., 1., mp_vars)
            #for k in range(len(lb)):
                #print("C1: ", mp_vars[k] >= lb[k])
                #print("C2: ", mp_vars[k] <= ub[k])
                #prog.AddLinearConstraint(mp_vars[k] >= lb[k])
                #prog.AddLinearConstraint(mp_vars[k] <= ub[k])
                #prog.AddLinearConstraint(mp_vars[k] <= ub[k])
                #prog.AddLinearConstraint(mp_vars[k] <= ub[k])
            # Add the observation constraint
            for k, val in enumerate(observed[1:]):
                if val != 0:
                    print("Val: ", self.prices[k] == val)
                    prog.AddConstraint(self.prices[k].Substitute(subs_dict) >= val - 2.)
                    prog.AddConstraint(self.prices[k].Substitute(subs_dict) <= val + 2)

            # Find lower bounds
            prog.AddCost(np.sum(mp_vars))
            #print(prog)
            solver = IpoptSolver()
            result = solver.Solve(prog)
            if result.is_success():
                lb = result.GetSolution(mp_vars)
                
                # Find upper bound too
                prog.AddCost(-2.*np.sum(mp_vars))
                result = solver.Solve(prog)
                if result.is_success():
                    ub = result.GetSolution(mp_vars)
                    subs_dict = {}
                    for k, v in enumerate(all_vars):
                        if lb[k] == ub[k]:
                            subs_dict[v] = lb[k]
                        else:
                            new_var = sym.Variable("feasible_%d" % k, sym.Variable.Type.RANDOM_UNIFORM)
                            subs_dict[v] = new_var * (ub[k] - lb[k]) + lb[k]
                    
                    self.prices = [self.prices[k].Substitute(subs_dict) for k in range(12)]
                    print("New prices: ", self.prices)
                    return

        except RuntimeError as e:
            print("Runtime error: ", e)
        self.rollout_probability = 0.
        
    def fit_model(self, num_samples, model_type="gmm", model_params={}):
        g = pydrake.common.RandomGenerator()
        v = [sym.Evaluate(self.prices, generator=g) for i in range(num_samples)]
        self.v = np.stack(v).reshape(num_samples, 12)

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
    reduced_prob = 1.
    for inc_1_length in range(7):
        reduced_prob /= 7.
        inc_2and3_length = 7 - inc_1_length
        for inc_3_length in range(0, inc_2and3_length):
            reduced_prob /= inc_2and3_length
            inc_2_length = inc_2and3_length - inc_3_length
            for dec_1_length in [2, 3]:
                reduced_prob /= 2
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
    print("Input base price: ", base_price)
    if base_price == 0:
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

def plot_gmm_results(valid_rollouts, observed, subplots=False):
    # Combine the valid rollouts into a final model over prices
    # by combining the modes
    all_means = []
    all_covars = []
    all_weights = []
    for rollout in valid_rollouts:
        all_means.append(rollout.g.means_)
        all_covars.append(rollout.g.covariances_)
        all_weights.append(rollout.g.weights_)
    all_means = np.vstack(all_means)
    all_covars = np.vstack(all_covars)
    all_weights = np.concatenate(all_weights)
    all_weights /= np.sum(all_weights)
    for k, val in enumerate(observed[1:]):
        if val != 0:
            all_means[:, k] = val
            all_covars[:, k] = 0.

    # Finally, draw it
    plt.figure()
    plt.grid(True)

    if subplots:
        for k in range(4):
            plt.subplot(5, 1, k+1)
            plt.title(pattern_index_to_name[k])
            plt.grid(True)

    for k, rollout in enumerate(valid_rollouts):
        if subplots:
            plt.subplot(5, 1, pattern_index_to_name.index(rollout.pattern_name) + 1)

        for comp_k in range(len(rollout.g.weights_)):
            mean = rollout.g.means_[comp_k, :]
            covar = rollout.g.covariances_[comp_k, :]
            plt.plot(range(12), mean,
                     color=colors_by_type[rollout.pattern_name],
                     alpha=0.5, linestyle="--")
            plt.fill_between(
                range(12), mean-1.96*np.sqrt(covar), mean+1.96*np.sqrt(covar),
                alpha=0.5, edgecolor=colors_by_type[rollout.pattern_name],
                facecolor=colormaps_by_type[rollout.pattern_name](float(k)/len(valid_rollouts)))

    for comp_k in range(len(all_weights)):
        mean = all_means[comp_k, :]
        covar = all_covars[comp_k, :]
        weight = all_weights[comp_k]


    # Mean of means
    if subplots:
        plt.subplot(5, 1, 5)
    plt.plot(range(12), np.sum(all_weights.T*all_means.T, axis=1), color=colors_by_type["Average Prediction"], alpha=0.5)

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

def plot_kde_results(valid_rollouts, observed, subplots=False):
    # Finally, draw it
    plt.figure()
    plt.grid(True)

    if subplots:
        for k in range(4):
            plt.subplot(5, 1, k+1)
            plt.title(pattern_index_to_name[k])
            plt.grid(True)

    grid_cells = np.arange(0., 660., 5.)

    all_prob_images = []
    regions = []
    for k, rollout in enumerate(valid_rollouts):
        if subplots:
            plt.subplot(5, 1, pattern_index_to_name.index(rollout.pattern_name) + 1)

        this_prob_image = np.vstack([kde.logpdf(grid_cells) for kde in rollout.kdes])
        region_bin = (this_prob_image > -1e3)
        region_low = grid_cells[np.argmax(region_bin, axis=1)]
        region_high = grid_cells[-1 - np.argmax(np.flip(region_bin, axis=1), axis=1)]
        regions.append((region_low, region_high))
        all_prob_images.append(this_prob_image + rollout.rollout_probability)

    total_prob_image = sum(all_prob_images) - logsumexp(all_prob_images)
    maxval = np.max(total_prob_image)
    minval = maxval - 1E-3

    if subplots:
        plt.subplot(5, 1, 5)
    maxes = grid_cells[np.argmax(total_prob_image, axis=1)]
    plt.imshow(np.log(-total_prob_image.T),
                   origin='lower', extent=[0, 11, 0, 660], aspect='auto', cmap="YlOrRd", interpolation="bilinear",
                   alpha=1.0)
    plt.plot(range(12), maxes, color=colors_by_type["Average Prediction"], alpha=0.5)
    print("Maxes: ", maxes)

    # Push to after initial imshow so these are on top
    for k, rollout in enumerate(valid_rollouts):
        if subplots:
            plt.subplot(5, 1, pattern_index_to_name.index(rollout.pattern_name) + 1)
        plt.fill_between(
            range(12), regions[k][0], regions[k][1],
            alpha=0.25, edgecolor=colors_by_type[rollout.pattern_name],
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

if __name__ == "__main__":
    model_type = "gmm"
    model_params = {"n_components": 1,
                    "bw_method": 0.05}
    subplots = False
    observed = [0, # base price, was 109, but first week functional buy price is random
            66, 61,  # M
            58, 53,  # T
            48, 114,  # W
            122, 137,  # R
            138, 136,  # F
            0, 0]  # S

    #observed = [109, # base price
    #        98, 94,  # M
    #        91, 86,  # T
    #        82, 78,  # W
    #        0, 0,  # R
    #        0, 0,  # F
    #        0, 0]  # S


    #observed_data = np.loadtxt("example_data.csv", dtype=int, delimiter=",", skiprows=1, usecols=range(1, 1+13))
    #observed = observed_data[8, :]

    print("Observed: ", observed)
    previous_pattern_prior = generate_previous_pattern_steady_state()
    all_rollouts = generate_all_turnip_patterns(previous_pattern_prior, observed[0])
    
    g = pydrake.common.RandomGenerator()
    
    start = time.time()
    num_samples = 250

    # Fit a model to reach rollout
    for rollout in all_rollouts:
        rollout.find_feasible_latents(observed)
        rollout.fit_model(
            num_samples,
            model_type=model_type,
            model_params=model_params)

        # Report the optimization degrees
        #for k, price in enumerate(rollout.prices):
        #    if price.is_polynomial():
        #        deg = sym.Polynomial(price).TotalDegree()
        #        if deg > 1:
        #            print("Price %d in %s of degree %d: " % (k, rollout.pattern_name, deg), price)
        #    else:
        #        print("Price %d in %s is not polynomial: " % (k, rollout.pattern_name), price)
        
    # Adjust the probability of each rollout based on the normal dist
    for rollout in all_rollouts:
        log_density = rollout.evaluate_logpdf(observed)
        rollout.rollout_probability = np.log(rollout.rollout_probability) + log_density
        print("New log prob for %s: " % rollout.pattern_name, rollout.rollout_probability)
        #sys.exit(0)
    
    # Normalize probabilities across remaining rollouts
    total_log_prob = logsumexp([rollout.rollout_probability for rollout in all_rollouts])
    valid_rollouts = []
    for rollout in all_rollouts:
        rollout.rollout_probability -= total_log_prob
        if rollout.rollout_probability > np.log(1e-4):
            valid_rollouts.append(rollout)

    if model_type == "gmm":
        plot_gmm_results(valid_rollouts, observed, subplots=subplots)
    elif model_type == "kde":
        plot_kde_results(valid_rollouts, observed, subplots=subplots)

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
    
    print("Elapsed: ", time.time() - start)
    plt.show()