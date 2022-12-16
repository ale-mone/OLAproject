from environment import *
from network import *
from helpers import find_user_prob_from_list

import numpy as np
import copy


def product_reward(env : Environment, prod_idx, user_idx, prices_idxs):
    
    # explore paths
    paths_list = []
    explore_path(env, paths_list, None, prod_idx, prices_idxs, user_idx)
    
    # initialize reward 
    reward = 0.
    
    for path in paths_list:
        reward += path.expected_return()
        
    return reward


def user_reward(env : Environment, user_idx, prices_idxs):
    
    # initialize reward 
    reward = 0.
    
    for i in range(len(env.products)):
        alpha_i = env.alphas[user_idx][i]
        reward += alpha_i * product_reward(env, i, user_idx, prices_idxs)
        
    return reward


def compute_reward(env : Environment, prices_idxs,
                   conversion_rates = None, alphas = None, n_prods_param = None, graph_probs = None,
                   user_idx = None, group_list = None, cat_prob = None):
    
    """ Compute the expected reward given a set of prices
    """
    
    # change parameters that are passed, consider theoretical values
    if conversion_rates is not None:
        env.conversion_rates = copy.deepcopy(conversion_rates)
    else:
        env.conversion_rates = copy.deepcopy(env.theoretical_values['conversion_rates'])
    if alphas is not None:
        env.alphas = copy.deepcopy(alphas)
    else:
        env.alphas = copy.deepcopy(env.theoretical_values['alphas'])
    if n_prods_param is not None:
        env.n_prods_param = copy.deepcopy(n_prods_param)
    else:
        env.n_prods_param = copy.deepcopy(env.theoretical_values['n_prods_param'])
    if graph_probs is not None:
        env.graph_probs = copy.deepcopy(graph_probs)
    else:
        env.graph_probs = copy.deepcopy(env.theoretical_values['graph_probs'])
    
    # fix the parameters passed if the user list is changed
    if group_list is not None:
        if conversion_rates is None:
            env.update_conversion_rates(group_list)
        if alphas is None:
            env.update_alphas(group_list)
        if n_prods_param is None:
            env.update_n_prods_param(group_list)
        if graph_probs is None:
            env.update_graph_probs(group_list)
    
    # initialize reward 
    reward = 0.
    
    # check situation to change the specific computations of the reward
    if group_list is None:
        if user_idx is None:
            for user_i in range(len(env.users)):
                reward += env.user_prob[user_i] * user_reward(env, user_i, prices_idxs)
        else:
            reward = user_reward(env, user_idx, prices_idxs)
    else:
        if cat_prob is None:
            user_prob = find_user_prob_from_list(group_list, env.cat_prob)
        else:
            user_prob = find_user_prob_from_list(group_list, cat_prob)
        for user_i in range(len(group_list)):
            reward += user_prob[user_i] * user_reward(env, user_i, prices_idxs)
        reward /= np.sum(user_prob)

    return reward

