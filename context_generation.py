from environment import *
from algorithms import *
from helpers import get_features_list

import numpy as np
import copy


class Context_Generator():
    def __init__(self, env: Environment, confidence, simulation_history, est_cat_prob, unknown_params):
        self.env                = env                           # environment
        self.confidence         = confidence                    # confidence for lower confidence bound
        self.unknown_params     = copy.deepcopy(unknown_params) # initial value of unknown parameters
        self.simulation_history = simulation_history            # simulation results
        self.est_cat_prob       = est_cat_prob                  # probability of observing a certain couple of features
    
    def update_parameters(self, simulation_history, est_cat_prob):
        
        self.simulation_history = simulation_history
        self.est_cat_prob       = est_cat_prob
    
    def get_group_params(self, group_list):
        
        # initialize group's users parameters
        n_sold         = np.zeros((5, 4)) # number of times products i was sold
        n_observed     = np.zeros((5, 4)) # number of times products i was observed
        n_initial_prod = np.zeros(5)      # number of times product i was shown as primary
        n_prod_sold    = np.zeros((2, 5)) # number of products of type i that were sold (and number of products sold)
        group_prob     = 0.
        
        # initialize group parameters
        conversion_rates = []
        alphas           = []
        n_prods_param    = []
        group_params     = {}
        
        for cat_idxs in group_list:
            # indexes of category
            cat_idxs_as_key = str(cat_idxs[0]) + str(cat_idxs[1])
            
            # retrieve simulation history
            n_sold         += self.simulation_history[cat_idxs_as_key]['n_sold'].copy()
            n_observed     += self.simulation_history[cat_idxs_as_key]['n_observed'].copy()
            n_initial_prod += self.simulation_history[cat_idxs_as_key]['n_initial_prod'].copy()
            n_prod_sold    += self.simulation_history[cat_idxs_as_key]['n_prod_sold'].copy()
            group_prob     += self.est_cat_prob[cat_idxs[0], cat_idxs[1]]
            
            # estimate conversion rates
            n_sold     += self.unknown_params['conversion_rates'][0].copy()
            n_observed += self.unknown_params['conversion_rates'][0].copy() + self.unknown_params['conversion_rates'][1].copy()
            est_cr      = n_sold / n_observed
            conversion_rates.append(est_cr.copy())
            
            # estimate alpha ratios
            n_initial_prod += self.unknown_params['alphas'][0].copy()
            est_alpha       = n_initial_prod / np.sum(n_initial_prod)
            alphas.append(est_alpha.copy())
            
            # estimate number of product sold
            n_prod_sold     += self.unknown_params['n_prods_param'].copy()
            est_n_prod_sold  = n_prod_sold[0] / n_prod_sold[1]
            n_prods_param.append(est_n_prod_sold.copy())
        
        # build return dictionary
        group_params['conversion_rates'] = conversion_rates
        group_params['alphas']           = alphas
        group_params['n_prods_param']    = n_prods_param
        group_params['group_prob']       = group_prob
        
        return group_params
    
    def lower_bound(self, data, group_list):
        
        # compute the dimension of the set of data
        data_dim = 0
        for cat_idxs in group_list:
            cat_idxs_as_key = str(cat_idxs[0]) + str(cat_idxs[1])
            data_dim += self.simulation_history[cat_idxs_as_key]['n_users']
        
        # compute the lower confidence bound
        LB = max(0, data - np.sqrt( - np.log(self.confidence) / (2 * data_dim) ))
        
        return LB
    
    def context_value(self, group_list):
        
        # retrieve group parameters
        group_params = self.get_group_params(group_list)
        conversion_rates = group_params['conversion_rates']
        alphas           = group_params['alphas']
        n_prods_param    = group_params['n_prods_param']
        group_prob       = group_params['group_prob']
        
        # compute optimal expected reward
        reward = greedy_optimizer(self.env,
                                  conversion_rates = conversion_rates,
                                  alphas           = alphas,
                                  n_prods_param    = n_prods_param,
                                  group_list       = group_list,
                                  cat_prob         = self.est_cat_prob)['expected_reward']
        
        # compute the group probability
        if group_prob == 1:
            LB_group_prob = group_prob
        else:
            LB_group_prob = self.lower_bound(group_prob, group_list)
        
        # compute the context value
        context_value = LB_group_prob * self.lower_bound(reward, group_list)
        
        return context_value
    
    def make_split(self, current_context, group_to_split, features_to_split):
        
        # compute context value for current group to split
        features_current_group = get_features_list(current_context)[group_to_split]
        current_context_value  = self.context_value(features_current_group)
        
        # initialize lower bound of the expected reward for the groups
        expected_reward_LB = np.zeros(len(features_to_split))
        
        # initialize indexes of groups possibly subject to a split
        to_split_list = []
        
        # find indexes associated to group to split
        i_values, j_values = np.where(current_context == group_to_split)
        
        # consider the possible splits
        for (k, feature) in enumerate(features_to_split):
            
            # create list of indexes of splitted groups
            if feature == 0: # splitting on feature 1
                group_1 = list(zip(i_values[i_values == 0], j_values[i_values == 0]))
                group_2 = list(zip(i_values[i_values == 1], j_values[i_values == 1]))
            if feature == 1: # splitting on feature 2
                group_1 = list(zip(i_values[j_values == 0], j_values[j_values == 0]))
                group_2 = list(zip(i_values[j_values == 1], j_values[j_values == 1]))
            
            # compute lower bound of the expected reward we have when splitting
            expected_reward_LB[k] = self.context_value(group_1) + self.context_value(group_2)
            
            # add indexes of new possible group that has to be splitted
            to_split_list.append(copy.deepcopy(group_2))
        
        # find index of group with maximum lower bound of the expected reward
        to_split_idx = np.argmax(expected_reward_LB)
        
        # check if split has to be made
        split_made = expected_reward_LB[to_split_idx] > current_context_value
        
        # update context if split is made
        if split_made:
            
            new_group = int(current_context.max()) + 1
            
            for cat_idxs in to_split_list[to_split_idx]:
                current_context[cat_idxs[0], cat_idxs[1]] = new_group
            
            # if we have not already splitted cancel the splitted feature
            if len(features_to_split) == 2:
                features_to_split.pop(to_split_idx)

        return split_made
    
    def run(self):
        
        # initialize context (all users aggregated)
        context = np.array([[0, 0], [0, 0]])
        
        # at first both features can be splitted
        features_to_split = [0, 1]
        
        # 1. split on group 0
        if self.make_split(context, 0, features_to_split):
            
            # 2. split again on group 0
            self.make_split(context, 0, features_to_split)
            
            # 3. split on the newly created group 1
            self.make_split(context, 1, features_to_split)
        
        return context
  
