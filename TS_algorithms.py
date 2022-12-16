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
    
    
class TS_Learner(Learner):
    
    """ Thompson sampling algorithm
    """
    
    def __init__(self, env : Environment, unknown_names, unknown_params):
        super().__init__(env)                                       # super-class constructor (Learner)
        self.unknown_names  = unknown_names                         # names of unknown parameters
        self.unknown_params = unknown_params                        # initial value of unknown parameters
        self.new_params     = copy.deepcopy(unknown_params)         # estimates of the unknown parameters
        self.params_history = {name : [] for name in unknown_names} # history of parameters found
        
    def reset(self):
        self.new_params = copy.deepcopy(self.unknown_params)
        
    def sample_conversion_rates(self):
        
        # if conversion rates are assumed to be known...
        if not self.unknown_names.count('conversion_rates'):
            return None
        
        # ...otherwise simulate them
        return [np.random.beta(self.new_params['conversion_rates'][0],
                               self.new_params['conversion_rates'][1])]
        
    def sample_alphas(self):
        
        # if alphas are assumed to be known...
        if not self.unknown_names.count('alphas'):
            return None
        
        # ...otherwise simulate them
        sampled_alphas = np.random.beta(self.new_params['alphas'][0],
                                        self.new_params['alphas'][1])
        sampled_alphas /= np.sum(sampled_alphas)

        return [sampled_alphas]
        
    def sample_n_prods_param(self):
        
        # if alphas are assumed to be known...
        if not self.unknown_names.count('n_prods_param'):
            return None
        
        # ...otherwise simulate them
        return [np.random.gamma(shape = self.new_params['n_prods_param'][0],
                                scale = 1 / self.new_params['n_prods_param'][1])]
        
    def sample_graph_probs(self):
        
        # if alphas are assumed to be known...
        if not self.unknown_names.count('graph_probs'):
            return None
        
        # ...otherwise simulate them
        compact_sampled_graph_probs = np.random.beta(self.new_params['graph_probs'][0],
                                                     self.new_params['graph_probs'][1])
        
        # generate probability matrix
        n = np.shape(compact_sampled_graph_probs)[0]
        sampled_graph_probs = np.zeros((n, n))
        for i, j in enumerate(list(self.env.secondary_prods.values())):
            sampled_graph_probs[i, j] = compact_sampled_graph_probs[i, :]
        
        return [sampled_graph_probs]
    
    def update_parameters(self, result_simulation, arms_pulled):
        
        for prod_idx in range(len(self.env.products)):
            
            # select parameters to update according to arm pulled (price selected)
            price_idx = arms_pulled[prod_idx]
            
            # update values of each parameter
            if self.unknown_names.count('conversion_rates'):
                # update beta parameters as:
                # a <- a + number of product i bought
                # b <- b + number of times users observed product i - number of product i bought
                self.new_params['conversion_rates'][0][prod_idx, price_idx] += result_simulation['conversion_rates'][0, prod_idx]
                self.new_params['conversion_rates'][1][prod_idx, price_idx] += result_simulation['conversion_rates'][1, prod_idx] - result_simulation['conversion_rates'][0, prod_idx]
            
            if self.unknown_names.count('alphas'):
                # update beta parameters as:
                # a <- a + number of times product i was the initial product
                # b <- b + number of times other products were the initial product
                self.new_params['alphas'][0, prod_idx] += result_simulation['alphas'][prod_idx]
                self.new_params['alphas'][1, prod_idx] += np.sum(result_simulation['alphas']) - result_simulation['alphas'][prod_idx]
                
            if self.unknown_names.count('n_prods_param'):
                # update number of products bought and number of products observed
                self.new_params['n_prods_param'] += result_simulation['n_prods_param']
                
            if self.unknown_names.count('graph_probs'):
                for scnd_position, scnd_idx in enumerate(list(self.env.secondary_prods.values())[prod_idx]):
                    # scnd_position : position of secondary product | first secondary = 0, second secondary = 1
                    # scnd_idx      : product index of secondary product | 0, 1, 2, 3, 4
                    # update beta parameters as:
                    # a <- a + number of clicks on product j shown as secondary of product i
                    # b <- b + number of times product j shown as secondary of product i has not been clicked
                    self.new_params['graph_probs'][0][prod_idx, scnd_position] += result_simulation['graph_probs'][0][prod_idx, scnd_idx]
                    self.new_params['graph_probs'][1][prod_idx, scnd_position] += result_simulation['graph_probs'][1][prod_idx, scnd_idx] - result_simulation['graph_probs'][0][prod_idx, scnd_idx]
        
    def daily_step(self, daily_users):
        
        """ Apply Thompson sampling algorithm for a given time t
            Algorithm:
                1. Sample parameters from a given distribution
                2. Select the arms to pull that return the highest expected
                    return for the given sampled parameters
                3. Observe the realization simulating the daily user interaction
                4. Update the distribution accordingly
        """
        
        # sample parameters + find prices that maximize the expected rewards given the sampled parameters
        arms_pulled = greedy_optimizer(self.env,
                                       conversion_rates = self.sample_conversion_rates(),
                                       alphas           = self.sample_alphas(),
                                       n_prods_param    = self.sample_n_prods_param(),
                                       graph_probs      = self.sample_graph_probs())['combination']
        
        # simulate the daily user interaction
        result_simulation = simulate_daily_interaction(self.env, arms_pulled, daily_users, self.unknown_names)
        
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
            if param_name == 'n_prods_param':
                estimate = self.new_params[param_name][0] / self.new_params[param_name][1]
            elif param_name == 'alphas':
                a = self.new_params[param_name][0, :]
                b = self.new_params[param_name][1, :]
                estimate = a / (a + b)
            else:
                a = self.new_params[param_name][0]
                b = self.new_params[param_name][1]
                estimate = a / (a + b)
                
            self.params_history[param_name].append(estimate)
        
    def run(self, n_days, daily_users):
        
        # initialize rewards for this run
        rewards = []
        
        # reinitialize parameters at each run
        self.reset()
        
        for day in range(n_days):
            
            # apply TS for a specific day
            arms_pulled = self.daily_step(daily_users)
            
            # append expected reward of the optimal combination (the pulled arms)
            # found considering the updated parameters
            rewards.append(compute_reward(self.env, arms_pulled))
        
        # save the results of the run
        self.save_results(rewards)
        
    def compute_regret_bound(self, T, eps):
        
        """ Compute TS theoretical regret upper bound
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
        
        # initialize delta and Kullback-Leibler divergence for each arm
        n = len(self.env.products)
        m = len(self.env.products[0].prices)
        delta_a = np.zeros((n, m-1))
        KL_a    = np.zeros((n, m-1))
    
        # compute delta and Kullback-Leibler divergence for each sub optimal arm
        for prod_idx in range(n):
            max_price = opt_combination[prod_idx]
            i = 0
            for price_idx in range(m):
                if price_idx != max_price:
                    idx_combination      = np.where(all_combinations[:, prod_idx] == price_idx)[0]
                    mean_reward          = np.mean(reward_list[idx_combination])
                    delta_a[prod_idx, i] = max_reward - mean_reward
                    KL_a[prod_idx, i]    = kl_div(max_reward, mean_reward)
                    i += 1
    
        # initialize regret's upper bound
        regret_upper_bound = np.zeros(T)
        
        # compute regret upper bound for each time instant
        for t in range(1, T):
            regret_upper_bound[t] = (1 + eps) * np.sum( delta_a * ( np.log(t+1) + np.log(np.log(t+1)) ) / KL_a )
        
        regret_upper_bound[0] = 0.8 * regret_upper_bound[1]
    
        return regret_upper_bound
        

class CG_group_TS_Learner(TS_Learner):
    
    """ Context generation Thompson sampling algorithm for a specific group
    """
    
    def __init__(self, env : Environment, unknown_names, unknown_params, group_list):
        super().__init__(env, unknown_names, unknown_params) # super-class constructor (TS_Learner)
        self.group_list = copy.deepcopy(group_list)          # copy list of users
        
    def iteration(self, est_cat_prob):
        
        """ Run one iteration of the Thompson sampling algorithm
            Algorithm:
                1. Sample parameters from a given distribution
                2. Select the arms to pull that return the highest expected
                    return for the given sampled parameters
        """
        
        # sample parameters + find prices that maximize the expected rewards given the sampled parameters
        arms_pulled = greedy_optimizer(self.env,
                                       conversion_rates = self.sample_conversion_rates() * len(self.group_list),
                                       alphas           = self.sample_alphas()           * len(self.group_list),
                                       n_prods_param    = self.sample_n_prods_param()    * len(self.group_list),
                                       group_list       = self.group_list,
                                       cat_prob         = est_cat_prob)['combination']
        
        return arms_pulled
    
    
class CG_TS_Learner(Learner):
    
    """ Context generation Thompson sampling algorithm
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
            n_observed     += self.simulation_history[cat_idxs_as_key]['n_observed'] - self.simulation_history[cat_idxs_as_key]['n_sold']
            n_initial_prod += self.simulation_history[cat_idxs_as_key]['n_initial_prod']
            n_prod_sold    += self.simulation_history[cat_idxs_as_key]['n_prod_sold']

        conversion_rates = np.array([n_sold, n_observed])
        
        return conversion_rates, n_initial_prod, n_prod_sold

    def update_group_learners(self):
        
        """ Update list of learners for each group according to the current context
        """
        
        self.group_learners = []
        
        n_groups = self.new_context_params['context'].max() + 1
        
        group_lists = get_features_list(self.new_context_params['context'])
        
        for group in range(n_groups):
            conversion_rates, n_initial_prod, n_prod_sold = self.get_group_simulation_history(group)
            
            # add initial parameters
            conversion_rates += self.unknown_params['conversion_rates']
            n_prod_sold      += self.unknown_params['n_prods_param']
            
            # create parameter dictionary
            group_simulation_history = {}
            group_simulation_history['conversion_rates'] = conversion_rates.copy()
            group_simulation_history['alphas']           = np.zeros((2, len(self.env.products)))
            group_simulation_history['alphas'][0]        = n_initial_prod.copy()
            group_simulation_history['alphas'][1]        = np.sum(n_initial_prod.copy()) - n_initial_prod.copy()
            group_simulation_history['alphas']          += self.unknown_params['alphas']
            group_simulation_history['n_prods_param']    = n_prod_sold.copy()
            
            self.group_learners.append(CG_group_TS_Learner(self.env, ['conversion_rates', 'alphas', 'n_prods_param'], group_simulation_history, group_lists[group]))

    def reset(self):
        
        # reset initial parameters
        self.new_context_params['context'] = copy.deepcopy(self.context_unknown_params['context'])
        self.simulation_history = copy.deepcopy(self.init_simulation_history)
        
        # initialize list of TS learners
        self.update_group_learners()

    def check_context_generation(self):
        
        # update context generator parameters
        self.context_generator.update_parameters(self.simulation_history, self.new_context_params['cat_prob'])
                
        # obtain new context
        self.new_context_params['context'] = self.context_generator.run()
        
        # in case the context has changed the learners have to be updated
        self.update_group_learners()

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
                self.check_context_generation()
                
            # apply TS for each group for a specific day
            list_arms_pulled = self.daily_step(daily_users)
            
            # append expected reward of the optimal combination (the pulled arms)
            # found considering the updated parameters
            rewards.append(self.compute_reward(list_arms_pulled))
        
        # save the results of the run
        self.save_results(rewards)
        
        