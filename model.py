import torch
import pyro
import pyro.distributions as dist
from torch.distributions import constraints
from pyro import poutine
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO, config_enumerate, infer_discrete
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.ops.indexing import Vindex

import pyro.poutine as poutine
from pyro.infer import MCMC, NUTS

'''

Implements Animal Crossing: New Horizons turnip pricing as a model
in Pyro.

Based primarily on the data mining by
https://gist.github.com/Treeki/85be14d297c80c8b3c0a76375743325b
and the writeup at
https://tinyurl.com/y76p2zzk.

'''

pattern_index_to_name = [
  "Random",
  "Large Spike",
  "Decreasing",
  "Small Spike"
]

pattern_transition_matrix = torch.tensor(
    [[0.2, 0.3, 0.15, 0.35],
     [0.5, 0.05, 0.2, 0.25],
     [0.25, 0.45, 0.05, 0.25],
     [0.45, 0.25, 0.15, 0.15]]).T

def calculate_turnip_prices():
    # Random integer base price on [90, 110]
    basePrice = pyro.sample(
      "basePrice",
      dist.Uniform(90, 110))

    # The prior over patterns is the steady state of
    # the transition matrix
    out = torch.eig(pattern_transition_matrix, eigenvectors=True)
    assert(torch.isclose(out.eigenvalues[0, 0], torch.tensor([1.])))
    prior_over_patterns = out.eigenvectors[:, 0]
    prior_over_patterns = prior_over_patterns / torch.sum(prior_over_patterns)

    # Select previous pattern from the steady-state prior
    prevPattern = pyro.sample(
      "prevPattern",
      dist.Categorical(prior_over_patterns),
      infer={"enumerate": "sequential"})
    # And select next pattern based on the transition probabilities
    nextPattern = pyro.sample(
      "nextPattern",
      dist.Categorical(pattern_transition_matrix[prevPattern, :]),
      infer={"enumerate": "sequential"},
      obs=torch.tensor([0]))

    print("basePrice: ", basePrice)
    print("prevPattern: ", prevPattern)
    print("nextPattern: ", nextPattern)

    # Now observe the 12 half-day prices based on the pattern.
    if nextPattern == 0:
      ### Random: high, decreasing, high, decreasing, high
      random_decPhaseLen1 = pyro.sample("random_decPhase1Len",
                                        dist.Categorical(torch.tensor([0.5, 0.5])),
                                        infer={"enumerate": "sequential"}) + 2
      random_decPhaseLen2 = 5 - random_decPhaseLen1
      random_hiPhaseLen1 = pyro.sample("random_hiPhase1Len",
                                       dist.Categorical(torch.ones(7)),
                                       infer={"enumerate": "sequential"})
      random_hiPhaseLen2and3 = 7 - random_hiPhaseLen1
      random_hiPhaseLen3 = pyro.sample("random_hiPhaseLen3",
                                       dist.Categorical(torch.ones(random_hiPhaseLen2and3)),
                                       infer={"enumerate": "sequential"})

      sellprices = []
      for k in range(random_hiPhaseLen1):
        rate = pyro.sample("random_hiPhase1_rate_%d" % k, dist.Uniform(0.9, 1.4))
        sellprices.append(rate * basePrice)

      rate = pyro.sample("random_decPhase1_initial_rate",
                         dist.Uniform(0.6, 0.8))
      for k in range(random_decPhaseLen1):
        sellprices.append(rate * basePrice)
        rate -= 0.04
        rate -= pyro.sample("random_decPhase1_dec_%d" % k, dist.Uniform(0.0, 0.06))

      for k in range(random_hiPhaseLen2and3 - random_hiPhaseLen3):
        rate = pyro.sample("random_hiPhase2_rate_%d" % k, dist.Uniform(0.9, 1.4))
        sellprices.append(rate * basePrice)

      rate = pyro.sample("random_decPhase2_initial_rate",
                         dist.Uniform(0.6, 0.8))
      for k in range(random_decPhaseLen2):
        sellprices.append(rate * basePrice)
        rate -= 0.04
        rate -= pyro.sample("random_decPhase2_dec_%d" % k, dist.Uniform(0.0, 0.06))

      for k in range(random_hiPhaseLen3):
        rate = pyro.sample("random_hiPhase3_rate_%d" % k, dist.Uniform(0.9, 1.4))
        sellprices.append(rate * basePrice)

      if len(sellprices[0].shape) >= 1:
        sellprices = torch.stack(sellprices, dim=1)
      else:
        sellprices = torch.stack(sellprices).reshape(1, 12)

      sellprices = torch.ceil(sellprices)
      print("Random sellprices: ", sellprices)

    elif nextPattern == 1:
      ### Pattern 1: Large Spike
      sellprices = []
      large_spike_peakStart = pyro.sample("large_spike_peakStart",
                                          dist.Categorical(torch.ones(7))) + 1
      rate = pyro.sample("large_spike_initial_rate",
                         dist.Uniform(0.85, 0.9))
      for k in range(large_spike_peakStart):
        sellprices.append(rate * basePrice)
        rate -= 0.03
        rate -= pyro.sample("large_spike_dec_%d" % k, dist.Uniform(0., 0.02))
      sellprices.append(pyro.sample("large_spike_rate_1", dist.Uniform(0.9, 1.4)) * basePrice)
      sellprices.append(pyro.sample("large_spike_rate_2", dist.Uniform(1.4, 2.0)) * basePrice)
      sellprices.append(pyro.sample("large_spike_rate_3", dist.Uniform(2.0, 6.0)) * basePrice)
      sellprices.append(pyro.sample("large_spike_rate_4", dist.Uniform(1.4, 2.0)) * basePrice)
      sellprices.append(pyro.sample("large_spike_rate_5", dist.Uniform(0.9, 1.4)) * basePrice)
      for k in range(12 - large_spike_peakStart - 5):
        sellprices.append(pyro.sample("large_spike_fin_rate_%d" % k, dist.Uniform(0.4, 0.9)) * basePrice)
      sellprices = torch.ceil(torch.stack(sellprices))
      print("Large spike sellprices: ", sellprices)

    elif nextPattern == 2:
      ### Pattern 2: Decreasing
      sellprices = []
      rate = 0.9
      rate -= pyro.sample("decreasing_dec_init", dist.Uniform(0.0, 0.05))
      for k in range(12):
        sellprices.append(rate * basePrice)
        rate -= 0.03
        rate -= pyro.sample("decreasing_dec_%d" % k, dist.Uniform(0.0, 0.02))
      
      sellprices = torch.ceil(torch.stack(sellprices))
      print("Decreasing sellprices: ", sellprices)

    elif nextPattern == 3:
      ### Pattern 3: Small Spike
      sellprices = []
      small_spike_peakStart = pyro.sample("small_spike_peakStart",
                                          dist.Categorical(torch.ones(8)))
      rate = pyro.sample("small_spike_initial_rate",
                         dist.Uniform(0.4, 0.9))
      for k in range(small_spike_peakStart):
        sellprices.append(rate * basePrice)
        rate -= 0.03
        rate -= pyro.sample("small_spike_dec", dist.Uniform(0., 0.02))
      sellprices.append(pyro.sample("small_spike_rate_1", dist.Uniform(0.9, 1.4)) * basePrice)
      sellprices.append(pyro.sample("small_spike_rate_2", dist.Uniform(0.9, 1.4)) * basePrice)
      rate = pyro.sample("small_spike_rate_spike_ub", dist.Uniform(1.4, 2.0))
      sellprices.append(pyro.sample("small_spike_rate_spike_pre", dist.Uniform(1.4, rate)) * basePrice - 1)
      sellprices.append(rate * basePrice)
      sellprices.append(pyro.sample("small_spike_rate_spike_post", dist.Uniform(1.4, rate)) * basePrice - 1)
      rate = pyro.sample("small_spike_rate_post_spike", dist.Uniform(0.4, 0.9))
      for k in range(12 - small_spike_peakStart - 5):
        sellprices.append(rate * basePrice)
        rate -= 0.03
        rate -= pyro.sample("small_spike_final_dec_%d" % k, dist.Uniform(0., 0.02))
      sellprices = torch.ceil(torch.stack(sellprices))
      print("Small spike sellprices: ", sellprices)
    else:
      raise ValueError("Invalid nextPattern %d" % nextPattern)
    pyro.sample("obs", dist.Delta(sellprices).to_event(1))

if __name__ == "__main__":
  calculate_turnip_prices()

  conditioned_model = poutine.condition(
    calculate_turnip_prices,
    data={"obs": torch.tensor([87.,  83.,  79.,  75.,  72.,  68.,  64., 106., 115., 144., 185., 138.])})
  nuts_kernel = NUTS(conditioned_model)
  mcmc = MCMC(nuts_kernel,
              num_samples=1000,
              warmup_steps=100,
              num_chains=1)
  mcmc.run()
  mcmc.summary(prob=0.5)
