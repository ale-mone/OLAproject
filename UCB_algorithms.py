from environment import *
from reward import *
from algorithms import *
from simulation import *
from helpers import *
from context_generation import *

from scipy.special import kl_div
import copy
import numpy as np


class Learner:
    
    """ Base class common to all learners
    """
    
    def __init__(self, env : Environment):
        self.env            = env # environment
        self.reward_history = []  # history of rewards found
        
    
class UCB_Learner(Learner):
    
    """ Upper Confidence Bound 1 algorithm
    """
    
    def __init__(self, env : Environment, unknown_names):
        super().__init__(env)                                           # super-class constructor (Learner)
        self.t                  = 1                                     # time
        self.unknown_names      = unknown_names                         # names of unknown parameters
        self.means              = {name : [] for name in unknown_names} # estimates of the unknown parameter means
        self.widths             = {name : [] for name in unknown_names} # estimates of the unknown parameter means
        self.initialize_intervals()
        self.params_history     = {name : [] for name in unknown_names} # history of parameters found
        self.simulation_history = {name : [] for name in unknown_names} # history of simulation results
        self.simulation_history['n_users'] = []
        # upper bounds for parameters
        self.learning_rate        = 3
        self.MAX_conversion_rates = 1
        self.MAX_n_prods_param    = 1000
        self.MAX_graph_probs      = 1
        
    def initialize_intervals(self):
        
        # initialize values of each parameter
        if self.unknown_names.count('conversion_rates'):
            self.means['conversion_rates']  = np.ones((len(self.env.products), len(self.env.products[0].prices)))
            self.widths['conversion_rates'] = np.ones((len(self.env.products), len(self.env.products[0].prices))) * np.inf
        if self.unknown_names.count('alphas'):
            self.means['alphas']  = np.ones(len(self.env.products)) * 0.2
            self.widths['alphas'] = np.ones(len(self.env.products)) * 1000
        if self.unknown_names.count('n_prods_param'):
            self.means['n_prods_param']  = np.ones(len(self.env.products))
            self.widths['n_prods_param'] = np.ones(len(self.env.products)) * np.inf
        if self.unknown_names.count('graph_probs'):
            self.means['graph_probs']  = np.ones((len(self.env.products), len(self.env.products)))
            self.widths['graph_probs'] = np.ones((len(self.env.products), len(self.env.products))) * np.inf
                                                           
    def reset(self):
        
        # initialize mean and width of each parameter
        self.initialize_intervals()
        
        # initialize history of simulation results
        self.simulation_history = {name : [] for name in self.unknown_names}
        self.simulation_history['n_users'] = []
        
        # initialize time
        self.t = 1
        
    def sample_conversion_rates(self):
        
        # if conversion rates are assumed to be known...
        if not self.unknown_names.count('conversion_rates'):
            return None
        
        # ...otherwise simulate them
        return np.expand_dims(np.minimum(np.sum([self.means['conversion_rates'], self.widths['conversion_rates']], axis = 0),
                                         self.MAX_conversion_rates)
                              , axis = 0)
        
    def sample_alphas(self):
        
        # if alphas are assumed to be known...
        if not self.unknown_names.count('alphas'):
            return None
        
        # ...otherwise simulate them
        return np.expand_dims(np.divide(np.sum([self.means['alphas'], self.widths['alphas']], axis = 0),
                                        np.sum([self.means['alphas'], self.widths['alphas']]))
                              , axis = 0)
        
    def sample_n_prods_param(self):
        
        # if alphas are assumed to be known...
        if not self.unknown_names.count('n_prods_param'):
            return None
        
        # ...otherwise simulate them
        return np.minimum(np.sum([self.means['n_prods_param'], self.widths['n_prods_param']], axis = 0),
                          [np.ones((len(self.env.products))) * self.MAX_n_prods_param])
        
    def sample_graph_probs(self):
        
        # if alphas are assumed to be known...
        if not self.unknown_names.count('graph_probs'):
            return None
        
        # ...otherwise simulate them
        return np.expand_dims(np.minimum(np.sum([self.means['graph_probs'], self.widths['graph_probs']], axis = 0),
                                         self.MAX_graph_probs)
                              , axis = 0)
    
    def update_parameters(self, result_simulation, arms_pulled):
        
        # increase time
        self.t += 1
        
        for prod_idx in range(len(self.env.products)):
            if self.unknown_names.count('conversion_rates'):
                # update mean
                n_sold     = np.sum(self.simulation_history['conversion_rates'], axis = 0)[0][prod_idx, arms_pulled[prod_idx]]
                n_observed = np.sum(self.simulation_history['conversion_rates'], axis = 0)[1][prod_idx, arms_pulled[prod_idx]]
                self.means['conversion_rates'][prod_idx, arms_pulled[prod_idx]] = n_sold / ( n_observed + 1e-10 )
                
                # update width
                for price_idx in range(len(self.env.products[0].prices)):
                    n_observed = np.sum(self.simulation_history['conversion_rates'], axis = 0)[1][prod_idx, price_idx]
                    n = n_observed / (np.mean(self.simulation_history['n_users']) + 1e-10)
                    if n > 0:
                        self.widths['conversion_rates'][prod_idx, price_idx] = 1/self.learning_rate * np.sqrt(2 * np.log(self.t) / (n * (self.t - 1)))
                    else:
                        self.widths['conversion_rates'][prod_idx, price_idx] = np.inf
                
            if self.unknown_names.count('alphas'):
                # update mean
                self.means['alphas'] = np.mean(self.simulation_history['alphas'], axis = 0)
                
                # update width
                n = len(self.simulation_history['alphas'])
                if n > 0:
                    self.widths['alphas'][prod_idx] = np.sqrt(2 * np.log(self.t) / (n * (self.t - 1)))
                else:
                    self.widths['alphas'][prod_idx] = np.inf
                
            if self.unknown_names.count('n_prods_param'):
                # update mean
                self.means['n_prods_param'] = np.mean(self.simulation_history['n_prods_param'], axis = 0)
                
                # update width
                n = len(self.simulation_history['n_prods_param'])
                if n > 0:
                    self.widths['n_prods_param'][prod_idx] = np.sqrt(2 * np.log(self.t) / (n * (self.t - 1)))
                else:
                    self.widths['n_prods_param'][prod_idx] = np.inf
                
            if self.unknown_names.count('graph_probs'):
                # update mean
                self.means['graph_probs'] = np.divide(np.sum(np.multiply(np.array(self.simulation_history['graph_probs'])[:, 0], np.array(self.simulation_history['graph_probs'])[:, 1]), axis = 0), np.maximum(np.sum(np.array(self.simulation_history['graph_probs'])[:, 1], axis = 0), 1))
                
                # update width
                for prod_idx_2 in range(len(self.env.products)):
                    n = np.sum(np.array(self.simulation_history['graph_probs'])[:, 1][:, prod_idx, prod_idx_2])
                    if n > 0:
                        self.widths['graph_probs'][prod_idx, prod_idx_2] = np.sqrt(2 * np.log(self.t) / (n * (self.t - 1)))
                    else:
                        self.widths['graph_probs'][prod_idx, prod_idx_2] = np.inf
                
    def save_simulation_results(self, result_simulation, arms_pulled):
        
        """ Append the results found to the history for each new run of the algorithm
        """
        
        # append number of users
        self.simulation_history['n_users'].append(result_simulation['n_users'])
        
        # append parameter results
        if self.unknown_names.count('conversion_rates'):
            n_sold         = np.zeros((len(self.env.products), len(self.env.products[0].prices)))
            n_observations = np.zeros((len(self.env.products), len(self.env.products[0].prices)))
            n_sold[np.arange(len(self.env.products)), arms_pulled]         = result_simulation['conversion_rates'][0]
            n_observations[np.arange(len(self.env.products)), arms_pulled] = result_simulation['conversion_rates'][1]
            self.simulation_history['conversion_rates'].append(np.array([np.array(n_sold), np.array(n_observations)]))
            
        if self.unknown_names.count('alphas'):
            self.simulation_history['alphas'].append(np.divide(result_simulation['alphas'], np.sum(result_simulation['alphas'])))
            
        if self.unknown_names.count('n_prods_param'):
            self.simulation_history['n_prods_param'].append(result_simulation['n_prods_param'][0]/(result_simulation['n_prods_param'][1]+1e-10))
            
        if self.unknown_names.count('graph_probs'):
            estimate = np.divide(result_simulation['graph_probs'][0], np.maximum(result_simulation['graph_probs'][1], 1))
            self.simulation_history['graph_probs'].append([estimate.tolist(),
                                                           result_simulation['graph_probs'][1].tolist()])
        
    def daily_step(self, daily_users):
        
        """ Apply upper confidence bound algorithm for a given time t
            Algorithm:
                1. Sample parameters from means and widths of the confidence intervals
                2. Select the arms to pull that return the highest expected
                    return for the given sampled parameters
                3. Observe the realization simulating the daily user interaction
                4. Update the means and widths accordingly
        """
        
        # sample parameters + find prices that maximize the expected rewards given the sampled parameters
        arms_pulled = greedy_optimizer(self.env,
                                       conversion_rates = self.sample_conversion_rates(),
                                       alphas           = self.sample_alphas(),
                                       n_prods_param    = self.sample_n_prods_param(),
                                       graph_probs      = self.sample_graph_probs())['combination']
        
        # simulate the daily user interaction
        result_simulation = simulate_daily_interaction(self.env, arms_pulled, daily_users, self.unknown_names)
        
        # save simulation result
        self.save_simulation_results(result_simulation, arms_pulled)
        
        # update distribution parameters
        self.update_parameters(result_simulation, arms_pulled)
        
        return arms_pulled
    
    def save_results(self, rewards):
        
        """ Append the results found to the history for each new run of the algorithm
        """
        
        # append rewards found
        self.reward_history.append(rewards)
        
        # append parameters
        for param_name in self.unknown_names:
            self.params_history[param_name].append(self.means[param_name])
        
    def run(self, n_days, daily_users):
        
        # initialize rewards for this run
        rewards = []
        
        # reinitialize parameters at each run
        self.reset()
        
        for day in range(n_days):
            # apply UCB for a specific day
            arms_pulled = self.daily_step(daily_users)
            # append expected reward of the optimal combination (the pulled arms)
            # found considering the updated parameters
            rewards.append(compute_reward(self.env, arms_pulled))
        
        # save the results of the run
        self.save_results(rewards)
        
    def compute_regret_bound(self, T, eps):
        
        """ Compute UCB theoretical regret upper bound
        """
        
        # initialize combination and reward lists
        all_combinations = []
        reward_list = []
        
        for i1 in range(4):
            for i2 in range(4):
                for i3 in range(4):
                    for i4 in range(4):
                        for i5 in range(4):
                            all_combinations.append([i1, i2, i3, i4, i5])
                            reward_list.append(compute_reward(self.env, [i1, i2, i3, i4, i5]))
        
        # turn into array
        all_combinations = np.array(all_combinations)
        reward_list      = np.array(reward_list)
        
        # find maximum reward and associated maximum combination
        max_reward      = np.max(reward_list)
        idx_max_reward  = np.argmax(reward_list)
        opt_combination = all_combinations[idx_max_reward]
        
        # initialize delta for each arm
        n = len(self.env.products)
        m = len(self.env.products[0].prices)
        delta_a = np.zeros((n, m-1))
    
        # compute delta for each sub optimal arm
        for prod_idx in range(n):
            max_price = opt_combination[prod_idx]
            i = 0
            for price_idx in range(m):
                if price_idx != max_price:
                    idx_combination      = np.where(all_combinations[:, prod_idx] == price_idx)[0]
                    mean_reward          = np.mean(reward_list[idx_combination])
                    delta_a[prod_idx, i] = max_reward - mean_reward
                    i += 1
    
        # initialize regret's upper bound
        regret_upper_bound = np.zeros(T)
        
        # compute regret upper bound for each time instant
        for t in range(T):
            regret_upper_bound[t] = np.sum( 4 * np.log(t+1) / delta_a + 8 * delta_a )
    
        return regret_upper_bound
    
    
class SW_UCB_Learner(UCB_Learner):
    
    """ Sliding window Upper Confidence Bound 1 algorithm
    """
    
    def __init__(self, env : Environment, unknown_names, SW_length, demand_curve_params):
        super().__init__(env, unknown_names)           # super-class constructor (UCB_Learner)
        self.SW_length           = SW_length           # length of sliding window
        self.demand_curve_params = demand_curve_params # different demand curves
    
    def SW_save_simulation_results(self, result_simulation, arms_pulled):
        
        """ Append the results found to the history for each new run of the algorithm
        """
        
        # save simulation results as before
        super().save_simulation_results(result_simulation, arms_pulled)
        
        # check number of observations for sliding window
        if self.unknown_names.count('conversion_rates'):
            if len(self.simulation_history['conversion_rates']) > self.SW_length:
                self.simulation_history['conversion_rates'].pop(0)
                self.t = self.SW_length
        if self.unknown_names.count('alphas'):
            if len(self.simulation_history['alphas']) > self.SW_length:
                self.simulation_history['alphas'].pop(0)
                self.t = self.SW_length
        if self.unknown_names.count('n_prods_param'):
            if len(self.simulation_history['n_prods_param']) > self.SW_length:
                self.simulation_history['n_prods_param'].pop(0)
                self.t = self.SW_length
        if self.unknown_names.count('graph_probs'):
            if len(self.simulation_history['graph_probs']) > self.SW_length:
                self.simulation_history['graph_probs'].pop(0)
                self.t = self.SW_length
                
    def daily_step(self, daily_users):
        
        """ Apply upper confidence bound algorithm with change detection for a given time t
            Algorithm:
                1. Sample parameters from means and widths of the confidence intervals
                2. Select the arms to pull that return the highest expected
                    return for the given sampled parameters
                3. Observe the realization simulating the daily user interaction
                4. Update the means and widths accordingly checking if some parameters
                    have to be discarded due to the sliding window
        """
        
        # sample parameters + find prices that maximize the expected rewards given the sampled parameters
        arms_pulled = greedy_optimizer(self.env,
                                       conversion_rates = self.sample_conversion_rates(),
                                       alphas           = self.sample_alphas(),
                                       n_prods_param    = self.sample_n_prods_param(),
                                       graph_probs      = self.sample_graph_probs())['combination']
        
        # simulate the daily user interaction
        result_simulation = simulate_daily_interaction(self.env, arms_pulled, daily_users, self.unknown_names)
        
        # save simulation result taking into account the sliding window
        self.SW_save_simulation_results(result_simulation, arms_pulled)
        
        # update distribution parameters
        super().update_parameters(result_simulation, arms_pulled)
        
        return arms_pulled
        
    def run(self, n_days, daily_users):
        
        # initialize rewards for this run
        rewards = []
        
        # reinitialize parameters at each run
        self.reset()
        
        for day in range(n_days):
            if day in self.demand_curve_params.keys():
                change_demand_curve(self.env, self.demand_curve_params[day])
            # apply UCB for a specific day
            arms_pulled = self.daily_step(daily_users)
            # compute expected reward of the optimal combination (the pulled arms)
            # found considering the updated parameters
            rewards.append(compute_reward(self.env, arms_pulled))
        
        # save the results of the run
        super().save_results(rewards)
    
    
class CD_UCB_Learner(UCB_Learner):
    
    """ Change detection Upper Confidence Bound 1 algorithm
    """
    
    def __init__(self, env : Environment, unknown_names, CD_params, demand_curve_params):
        super().__init__(env, unknown_names)           # super-class constructor (UCB_Learner)
        self.CD_params = copy.deepcopy(CD_params)      # change detection parameters
        self.initialize_CD_params()
        self.demand_curve_params = demand_curve_params # different demand curves
    
    def initialize_CD_params(self):
        self.CD_params['t']      = np.ones((len(self.env.products), len(self.env.products[0].prices)))
        self.CD_params['means']  = np.zeros((len(self.env.products), len(self.env.products[0].prices)))
        self.CD_params['S_high'] = np.zeros((len(self.env.products), len(self.env.products[0].prices)))
        self.CD_params['S_low']  = np.zeros((len(self.env.products), len(self.env.products[0].prices)))
        
    def reset_CD_params(self, change_detected_mask):
        self.CD_params['t'][change_detected_mask]      = 1
        self.CD_params['means'][change_detected_mask]  = 0
        self.CD_params['S_high'][change_detected_mask] = 0
        self.CD_params['S_low'][change_detected_mask]  = 0
        
    def reset_UCB_params(self, change_detected_mask):
        self.means['conversion_rates'][change_detected_mask]  = 0
        self.widths['conversion_rates'][change_detected_mask] = np.inf
        for elem in self.simulation_history['conversion_rates']:
            elem[0][change_detected_mask] = 0
            elem[1][change_detected_mask] = 1e-10
            
    def CD_reset(self):
        
        # initialize mean and width of each parameter
        self.initialize_intervals()
        
        # initialize change detection parameters
        self.initialize_CD_params()
        
        # initialize history of simulation results
        self.simulation_history = {name : [] for name in self.unknown_names}
        self.simulation_history['n_users'] = []
        
        # initialize time
        self.t = 1
        
    def detect_change(self, conversion_rates_estimate, arms_pulled):
        
        # initialize change_detected
        n_prods = len(self.env.products)
        change_detected = np.full((n_prods, len(self.env.products[0].prices)), False)
        
        for prod_idx, price_idx in zip(range(n_prods), arms_pulled):
            # check if enough time has passed
            if self.CD_params['t'][prod_idx, price_idx] <= self.CD_params['T']:
                # update reference mean value
                self.CD_params['means'][prod_idx, price_idx] += conversion_rates_estimate[prod_idx, price_idx] / self.CD_params['T']
            else:
                # update reference mean value
                self.CD_params['means'][prod_idx, price_idx] = ( self.CD_params['means'][prod_idx, price_idx] * (self.CD_params['t'][prod_idx, price_idx] - 1) + conversion_rates_estimate[prod_idx, price_idx] ) / self.CD_params['t'][prod_idx, price_idx]
                # check if concept drift has happened
                z = ( conversion_rates_estimate[prod_idx, price_idx] - self.CD_params['means'][prod_idx, price_idx] )
                self.CD_params['S_high'][prod_idx, price_idx] = max(0, self.CD_params['S_high'][prod_idx, price_idx] + z - self.CD_params['w'])
                self.CD_params['S_low'][prod_idx, price_idx]  = max(0, self.CD_params['S_low'][prod_idx, price_idx] - z - self.CD_params['w'])
                change_detected[prod_idx, price_idx] = (self.CD_params['S_low'][prod_idx, price_idx] > self.CD_params['threshold']) or (self.CD_params['S_high'][prod_idx, price_idx] > self.CD_params['threshold'])
            
            # enlarge mask to all prices if change is detected for the pulled price
            if change_detected[prod_idx, price_idx]:
                change_detected[prod_idx, :] = np.full(len(self.env.products[0].prices), True)
            
            # increase time
            self.CD_params['t'][prod_idx, price_idx] += 1
        
        return change_detected
        
    def CD_update_parameters(self, result_simulation, arms_pulled):
        
        # increase time
        self.t += 1
        
        # check if concept drift happened
        if self.unknown_names.count('conversion_rates'):
            last_estimate = self.simulation_history['conversion_rates'][-1][0] / ( self.simulation_history['conversion_rates'][-1][1] + 1e-10 )
            change_detected_mask = self.detect_change(last_estimate, arms_pulled)
            self.reset_CD_params(change_detected_mask)
            self.reset_UCB_params(change_detected_mask)
        
        for prod_idx in range(len(self.env.products)):
            if self.unknown_names.count('conversion_rates'):
                # update mean
                n_sold     = np.sum(self.simulation_history['conversion_rates'], axis = 0)[0][prod_idx, arms_pulled[prod_idx]]
                n_observed = np.sum(self.simulation_history['conversion_rates'], axis = 0)[1][prod_idx, arms_pulled[prod_idx]]
                self.means['conversion_rates'][prod_idx, arms_pulled[prod_idx]] = n_sold / ( n_observed + 1e-10 )
            
                # update width
                for price_idx in range(len(self.env.products[0].prices)):
                    n_observed = np.sum(self.simulation_history['conversion_rates'], axis = 0)[1][prod_idx, price_idx]
                    n = n_observed / np.mean(self.simulation_history['n_users'])
                    if n > 0:
                        self.widths['conversion_rates'][prod_idx, price_idx] = 1/self.learning_rate * np.sqrt(2 * np.log(self.t) / (n * (self.t - 1)))
                    else:
                        self.widths['conversion_rates'][prod_idx, price_idx] = np.inf
            
            if self.unknown_names.count('alphas'):
                # update mean
                self.means['alphas'] = np.mean(self.simulation_history['alphas'], axis = 0)
                
                # update width
                n = len(self.simulation_history['alphas'])
                if n > 0:
                    self.widths['alphas'][prod_idx] = np.sqrt(2 * np.log(self.t) / (n * (self.t - 1)))
                else:
                    self.widths['alphas'][prod_idx] = np.inf
                
            if self.unknown_names.count('n_prods_param'):
                # update mean
                self.means['n_prods_param'] = np.mean(self.simulation_history['n_prods_param'], axis = 0)
                
                # update width
                n = len(self.simulation_history['n_prods_param'])
                if n > 0:
                    self.widths['n_prods_param'][prod_idx] = np.sqrt(2 * np.log(self.t) / (n * (self.t - 1)))
                else:
                    self.widths['n_prods_param'][prod_idx] = np.inf
                
            if self.unknown_names.count('graph_probs'):
                # update mean
                self.means['graph_probs'] = np.divide(np.sum(np.multiply(np.array(self.simulation_history['graph_probs'])[:, 0], np.array(self.simulation_history['graph_probs'])[:, 1]), axis = 0), np.maximum(np.sum(np.array(self.simulation_history['graph_probs'])[:, 1], axis = 0), 1))
                # update width
                for prod_idx_2 in range(len(self.env.products)):
                    # total number of samples on the secondary product [product_idx_2] for [prod_idx_1] as primary
                    n = np.sum(np.array(self.simulation_history['graph_probs'])[:, 1][:, prod_idx, prod_idx_2])
                    if n > 0:
                        self.widths['graph_probs'][prod_idx, prod_idx_2] = np.sqrt(2 * np.log(self.t) / (n * (self.t - 1)))
                    else:
                        self.widths['graph_probs'][prod_idx, prod_idx_2] = np.inf
                        
    def daily_step(self, daily_users):
        
        """ Apply upper confidence bound algorithm with change detection for a given time t
            Algorithm:
                1. Sample parameters from means and widths of the confidence intervals
                2. Select the arms to pull that return the highest expected
                    return for the given sampled parameters
                3. Observe the realization simulating the daily user interaction
                4. Update the means and widths accordingly checking for concept drifts
        """
        
        # sample parameters + find prices that maximize the expected rewards given the sampled parameters
        arms_pulled = greedy_optimizer(self.env,
                                       conversion_rates = self.sample_conversion_rates(),
                                       alphas           = self.sample_alphas(),
                                       n_prods_param    = self.sample_n_prods_param(),
                                       graph_probs      = self.sample_graph_probs())['combination']
        
        # simulate the daily user interaction
        result_simulation = simulate_daily_interaction(self.env, arms_pulled, daily_users, self.unknown_names)
        
        # save simulation result taking into account the sliding window
        super().save_simulation_results(result_simulation, arms_pulled)
        
        # update distribution parameters
        self.CD_update_parameters(result_simulation, arms_pulled)
        
        return arms_pulled
        
    def run(self, n_days, daily_users):
        
        # initialize rewards for this run
        rewards = []
        
        # reinitialize all parameters at each run
        self.CD_reset()
        
        for day in range(n_days):
            if day in self.demand_curve_params.keys():
                change_demand_curve(self.env, self.demand_curve_params[day])
            # apply UCB for a specific day
            arms_pulled = self.daily_step(daily_users)
            # compute expected reward of the optimal combination (the pulled arms)
            # found considering the updated parameters
            rewards.append(compute_reward(self.env, arms_pulled))
        
        # save the results of the run
        super().save_results(rewards)
    
class CG_group_UCB_Learner(Learner):

    """ Context generation Upper Confidence Bound algorithm for a specific group
    """

    def __init__(self, env: Environment, group_params, group_list, t):
        super().__init__(env)                           # super-class constructor (Learner)
        self.group_params = copy.deepcopy(group_params) # initialize learner parameters
        self.means        = {}                          # estimates of the unknown parameter means
        self.widths       = {}                          # estimates of the unknown parameter means
        self.t            = t                           # initialize time
        self.initialize_intervals()
        self.group_list   = copy.deepcopy(group_list)   # copy list of users
        self.daily_users  = []                          # initialize number of users
        # upper bounds for parameters
        self.learning_rate        = 3
        self.MAX_conversion_rates = 1
        self.MAX_alphas           = 0.2
        self.MAX_n_prods_param    = 10000

    def initialize_intervals(self):
        
        # initialize means of each parameter
        self.means['conversion_rates'] = self.group_params['conversion_rates'][0, :, :] / self.group_params['conversion_rates'][1, :, :]
        self.means['alphas']           = self.group_params['alphas'] / np.sum(self.group_params['alphas'])
        self.means['n_prods_param']    = self.group_params['n_prods_param'][0] / self.group_params['n_prods_param'][1]
        
        # initialize widths of each parameter
        self.widths['conversion_rates'] = np.ones((len(self.env.products), len(self.env.products[0].prices))) * np.inf
        self.widths['alphas']           = np.ones(len(self.env.products)) * np.inf
        self.widths['n_prods_param']    = np.ones(len(self.env.products)) * np.inf
            
        # update widths of each parameter
        for prod_idx in range(len(self.env.products)):
            for price_idx in range(len(self.env.products[0].prices)):
                
                # conversion rates
                n = self.group_params['conversion_rates'][1, prod_idx, price_idx]
                if n >= 1 and self.t > 1:
                    self.widths['conversion_rates'][prod_idx, price_idx] = np.sqrt(2 * np.log(self.t) / (n * (self.t - 1)))
                        
                # alphas
                n = self.group_params['alphas'][prod_idx]
                if n >= 1 and self.t > 1:
                    self.widths['alphas'][prod_idx] = np.sqrt(2 * np.log(self.t) / (n * (self.t - 1)))
                        
                # Poisson parameters
                n = self.group_params['n_prods_param'][1, prod_idx]
                if n >= 1 and self.t > 1:
                    self.widths['n_prods_param'][prod_idx] = np.sqrt(2 * np.log(self.t) / (n * (self.t - 1)))

    def iteration(self, est_cat_prob):
        
        # sample parameters
        sampled_params = {}
        sampled_params['conversion_rates'] = np.minimum(np.array([self.means['conversion_rates'] + self.widths['conversion_rates']]), self.MAX_conversion_rates)
        if np.sum([self.means['alphas'], self.widths['alphas']]) == np.inf:
            sampled_params['alphas'] = self.MAX_alphas * np.ones(len(self.means['alphas']))
        else:
            sampled_params['alphas'] = np.divide(np.sum([self.means['alphas'], self.widths['alphas']], axis = 0), np.sum([self.means['alphas'], self.widths['alphas']]))
        sampled_params['n_prods_param'] = np.minimum(np.sum([self.means['n_prods_param'], self.widths['n_prods_param']], axis = 0), self.MAX_n_prods_param)
        
        # pull arms
        arms_pulled = greedy_optimizer(self.env,
                                       conversion_rates = list(sampled_params['conversion_rates']) * len(self.group_list),
                                       alphas           = [sampled_params['alphas']]               * len(self.group_list),
                                       n_prods_param    = [sampled_params['n_prods_param']]        * len(self.group_list),
                                       group_list       = self.group_list,
                                       cat_prob         = est_cat_prob)['combination']

        return arms_pulled
    
    def update_parameters(self, group_result_simulation, arms_pulled):
        
        # time
        self.t += 1

        # number of users
        n_users = np.sum(group_result_simulation['alphas'])
        self.daily_users.append(n_users)

        # conversion rates
        self.group_params['conversion_rates'][0, np.arange(5), arms_pulled] += group_result_simulation['conversion_rates'][0]
        self.group_params['conversion_rates'][1, np.arange(5), arms_pulled] += group_result_simulation['conversion_rates'][1]
        
        # alpha ratios
        self.group_params['alphas'] += group_result_simulation['alphas']
        
        # Poisson parameters
        self.group_params['n_prods_param'] += group_result_simulation['n_prods_param']
        
        # update parameters
        for prod_idx in range(len(self.env.products)):
            for price_idx in arms_pulled:
                # update means
                self.means['conversion_rates'][prod_idx, price_idx] = self.group_params['conversion_rates'][0, prod_idx, price_idx] / self.group_params['conversion_rates'][1, prod_idx, price_idx]
                
                # update widths
                n = self.group_params['conversion_rates'][1, prod_idx, price_idx] / ( np.mean(self.daily_users) / self.learning_rate )
                if n > 0 and self.t > 1:
                    self.widths['conversion_rates'][prod_idx, price_idx] = np.sqrt(2 * np.log(self.t) / (n * (self.t - 1)))
                else:
                    self.widths['conversion_rates'][prod_idx, price_idx] = np.inf

            # update means
            self.means['alphas'][prod_idx] = self.group_params['alphas'][prod_idx] / np.sum(self.group_params['alphas'])

            # update widths
            n = self.group_params['alphas'][prod_idx]
            if n > 0 and self.t > 1:
                self.widths['alphas'][prod_idx] = np.sqrt(np.divide(2 * np.log(self.t), (n * (self.t - 1))))
            else:
                self.widths['alphas'][prod_idx] = np.inf

            # update means
            self.means['n_prods_param'][prod_idx] = self.group_params['n_prods_param'][0, prod_idx] / self.group_params['n_prods_param'][1, prod_idx]

            # update widths
            n = self.group_params['n_prods_param'][1, prod_idx]
            if n > 0 and self.t > 1:
                self.widths['n_prods_param'][prod_idx] = np.sqrt(np.divide(2 * np.log(self.t), (n * (self.t - 1))))
            else:
                self.widths['n_prods_param'][prod_idx] = np.inf
                
    
class CG_UCB_Learner(Learner):

    """ Context generation Upper Confidence Bound algorithm
    """

    def __init__(self, env: Environment, unknown_names, unknown_params, context_unknown_params, confidence):
        super().__init__(env)                                       # super-class constructor (Learner)
        self.unknown_names  = unknown_names                         # names of unknown parameters
        self.unknown_params = copy.deepcopy(unknown_params)         # initial value of unknown parameters
        self.params_history = {name : [] for name in unknown_names} # history of parameters found
        self.group_learners = []                                    # TS learner for each group
        
        # simulation history
        self.init_user_simulation_history = {'n_users'        : 0,                # number of users that interacted with the site
                                             'n_sold'         : np.zeros((5, 4)), # number of times products i was sold
                                             'n_observed'     : np.zeros((5, 4)), # number of times products i was observed
                                             'n_initial_prod' : np.zeros(5),      # number of times product i was shown as primary
                                             'n_prod_sold'    : np.zeros((2, 5))} # number of products of type i that were sold (and number of products sold)
        self.init_simulation_history = {'00' : copy.deepcopy(self.init_user_simulation_history),
                                        '01' : copy.deepcopy(self.init_user_simulation_history),
                                        '10' : copy.deepcopy(self.init_user_simulation_history),
                                        '11' : copy.deepcopy(self.init_user_simulation_history)}
        self.simulation_history = copy.deepcopy(self.init_simulation_history)
        
        # context generation
        self.context_unknown_params = copy.deepcopy(context_unknown_params) # initial value of context unknown parameters
        self.new_context_params     = copy.deepcopy(context_unknown_params) # estimates of context unknown parameters
        self.context_generator      = Context_Generator(self.env, confidence, self.simulation_history, context_unknown_params['cat_prob'], unknown_params)

    def get_group_simulation_history(self, group):
        
        """ Aggregate group simulation history according to the context
        """
        
        n_sold         = np.zeros((5, 4))
        n_observed     = np.zeros((5, 4))
        n_initial_prod = np.zeros(5)
        n_prod_sold    = np.zeros((2, 5))
        
        i_values, j_values = np.where(self.new_context_params['context'] == group)
        
        for k in range(len(i_values)):
            # indexes of category
            cat_idxs_as_key = str(i_values[k]) + str(j_values[k])
            
            n_sold         += self.simulation_history[cat_idxs_as_key]['n_sold']
            n_observed     += self.simulation_history[cat_idxs_as_key]['n_observed']
            n_initial_prod += self.simulation_history[cat_idxs_as_key]['n_initial_prod']
            n_prod_sold    += self.simulation_history[cat_idxs_as_key]['n_prod_sold']

        conversion_rates = np.array([n_sold, n_observed])
        
        return conversion_rates, n_initial_prod, n_prod_sold

    def update_group_learners(self, t):
        
        """ Update list of learners for each group according to the current context
        """
        
        self.group_learners = []
        
        n_groups = self.new_context_params['context'].max() + 1
        
        group_lists = get_features_list(self.new_context_params['context'])
        
        for group in range(n_groups):
            conversion_rates, n_initial_prod, n_prod_sold = self.get_group_simulation_history(group)
            
            # add initial parameters
            conversion_rates += self.unknown_params['conversion_rates']
            n_initial_prod   += self.unknown_params['alphas'][0]
            n_prod_sold      += self.unknown_params['n_prods_param']
            
            # create parameter dictionary
            group_simulation_history = {}
            group_simulation_history['conversion_rates'] = conversion_rates.copy()
            group_simulation_history['alphas']           = n_initial_prod.copy()
            group_simulation_history['n_prods_param']    = n_prod_sold.copy()
            
            self.group_learners.append(CG_group_UCB_Learner(self.env, group_simulation_history, group_lists[group], t))

    def reset(self):
        
        # reset initial parameters
        self.new_context_params['context'] = copy.deepcopy(self.context_unknown_params['context'])
        self.simulation_history = copy.deepcopy(self.init_simulation_history)
        
        # initialize list of TS learners
        self.update_group_learners(1)

    def check_context_generation(self, t):
        
        # update context generator parameters
        self.context_generator.update_parameters(self.simulation_history, self.new_context_params['cat_prob'])
                
        # obtain new context
        self.new_context_params['context'] = self.context_generator.run()
        
        # in case the context has changed the learners have to be updated
        self.update_group_learners(t)

    def aggregate_result_simulation(self, result_simulation):
        
        """ Aggregate the results of the given simulation according to the context
        """

        groups_result_simulation = []
        
        n_groups = self.new_context_params['context'].max() + 1
        
        for group in range(n_groups):
            i_values, j_values = np.where(self.new_context_params['context'] == group)
            cat_idxs_as_key = str(i_values[0]) + str(j_values[0])
            groups_result_simulation.append(copy.deepcopy(result_simulation[cat_idxs_as_key]))
            if len(i_values > 1):
                for k in range(1, len(i_values)):
                    cat_idxs_as_key = str(i_values[k]) + str(j_values[k])
                    temp_result_simulation = result_simulation[cat_idxs_as_key]
                    for key in temp_result_simulation.keys():
                        groups_result_simulation[-1][key] += temp_result_simulation[key]

        return groups_result_simulation

    def save_simulation_results(self, result_simulation, list_arms_pulled):
        
        """ Save the results of the current simulation
        """
        
        for cat_idxs_as_key in result_simulation.keys():
            
            # find arms pulled for specific category chosen
            i, j = int(cat_idxs_as_key[0]), int(cat_idxs_as_key[1])
            group = self.new_context_params['context'][i, j]
            arms_pulled = list_arms_pulled[group]
            n = len(arms_pulled)
            
            # update simulation history
            self.simulation_history[cat_idxs_as_key]['n_users']                               += result_simulation[cat_idxs_as_key]['n_users']
            self.simulation_history[cat_idxs_as_key]['n_sold'][np.arange(n), arms_pulled]     += result_simulation[cat_idxs_as_key]['conversion_rates'][0]
            self.simulation_history[cat_idxs_as_key]['n_observed'][np.arange(n), arms_pulled] += result_simulation[cat_idxs_as_key]['conversion_rates'][1]
            self.simulation_history[cat_idxs_as_key]['n_initial_prod']                        += result_simulation[cat_idxs_as_key]['alphas']
            self.simulation_history[cat_idxs_as_key]['n_prod_sold']                           += result_simulation[cat_idxs_as_key]['n_prods_param']

    def update_cat_prob(self):
        
        """ Update matrix of probability of observing a specific couple of features (category of user)
        """
        
        est_cat_prob = np.zeros((2, 2))
        
        for cat_idxs_as_key in self.simulation_history.keys():
            # find indexes for specific category chosen
            i, j = int(cat_idxs_as_key[0]), int(cat_idxs_as_key[1])
            est_cat_prob[i, j] += self.simulation_history[cat_idxs_as_key]['n_users']

        self.new_context_params['cat_prob'] = est_cat_prob / np.sum(est_cat_prob)

    def daily_step(self, daily_users):
        
        list_arms_pulled = []
        
        # find prices that maximize the expected rewards for each group
        for group_learner in self.group_learners:
            arms_pulled = group_learner.iteration(self.new_context_params['cat_prob'])
            list_arms_pulled.append(arms_pulled.copy())
            
        # simulate the daily user interaction
        result_simulation = simulate_daily_interaction_CG(self.env, list_arms_pulled, daily_users, self.new_context_params['context'])
        
        # save simulation results
        self.save_simulation_results(result_simulation, list_arms_pulled)
        
        # update estimate of category probability
        self.update_cat_prob()
        
        # aggregate for each group the results of the simulation
        groups_result_simulation = self.aggregate_result_simulation(result_simulation)
        
        # update distribution parameters for each learner
        for group_learner_idx, group_learner in enumerate(self.group_learners):
            group_learner.update_parameters(groups_result_simulation[group_learner_idx], list_arms_pulled[group_learner_idx])
        
        return list_arms_pulled

    def compute_reward(self, list_arms_pulled):
        
        reward = 0.
        feature_list = get_features_list(self.new_context_params['context'])
        
        for k, group_list in enumerate(feature_list):
            # compute the probability of observing a user from the group
            group_prob   = np.sum(find_user_prob_from_list(group_list, self.env.cat_prob))
            
            # compute group reward
            group_reward = compute_reward(self.env,
                                          list_arms_pulled[k],
                                          group_list = group_list)
            
            # update reward
            reward += group_prob * group_reward

        return reward

    def save_results(self, rewards):
        
        """ Append the results found to the history for each new run of the algorithm
        """
        
        # append rewards found
        self.reward_history.append(rewards)

    def run(self, n_days, daily_users):
        
        # initialize rewards for this run
        rewards = []
        
        # reinitialize parameters at each run
        self.reset()
        
        for day in range(n_days):
            
            # check if context generation has to be applied
            if day % 14 == 0 and day != 0:
                self.check_context_generation(day)
                
            # apply TS for each group for a specific day
            list_arms_pulled = self.daily_step(daily_users)
            
            # append expected reward of the optimal combination (the pulled arms)
            # found considering the updated parameters
            rewards.append(self.compute_reward(list_arms_pulled))
        
        # save the results of the run
        self.save_results(rewards)

    