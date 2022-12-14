from reward import *

import numpy as np


def greedy_optimizer(env : Environment,
                     conversion_rates = None, alphas = None, n_prods_param = None, graph_probs = None,
                     user_idx = None, group_list = None, cat_prob = None):
    
    """ Find the best price combination exploring combinations using a greedy approach
        changing price combination only if the new one provides a marginal increase
        in the expected reward
    """

    # for every product start from the lowest price
    prices_idxs = [0 for x in env.products]
    
    # initialize algorithm
    highest_reward = compute_reward(env, prices_idxs,
                                    conversion_rates, alphas, n_prods_param, graph_probs,
                                    user_idx, group_list, cat_prob)
    new_reward     = highest_reward
    reward_history = []
    reward_history.append(highest_reward)

    updated = True

    while updated == True:

        updated = False

        for i in range(5): # cycle over products
            temp_prices_idxs = prices_idxs.copy() # reset indexes

            if temp_prices_idxs[i] < 3:
                temp_prices_idxs[i] += 1 # increase price for product ith
                new_reward = compute_reward(env, temp_prices_idxs,
                                            conversion_rates, alphas, n_prods_param, graph_probs,
                                            user_idx, group_list, cat_prob)

                if new_reward > highest_reward:
                    highest_reward = new_reward.copy()
                    updated_prices_idxs = temp_prices_idxs.copy()
                    updated = True

        if updated == True:
            prices_idxs = updated_prices_idxs.copy() # if there is an update then save the new price indexes
            reward_history.append(highest_reward)

    result = {'expected_reward' : highest_reward,
              'combination'     : prices_idxs}
    
    return result
    

def brute_force_optimizer(env : Environment,
                          user_idx = None, disaggregated = False):

    """ Find the best price combination exploring all the possible combinations
        and taking the one with the highest expected reward
    """
    
    if disaggregated :
        highest_rewards = np.zeros(len(env.users))
        prices_idxs     = [[]] * len(env.users)
        
        for user_i in range(len(env.users)):
            result_i = brute_force_optimizer(env, user_idx = user_i)
            highest_rewards[user_i] = result_i['optimal_reward']
            prices_idxs[user_i]     = result_i['combination']

        result = {'optimal_reward' : highest_rewards,
                  'combination'    : prices_idxs}
        
        return result

    optimal_combination = [0, 0, 0, 0, 0]
    highest_reward = 0.
    reward = 0.
    
    # enumerate all possible combinations of prices (1024 in total)
    all_combinations = []
    for i1 in range(4):
        for i2 in range(4):
            for i3 in range(4):
                for i4 in range(4):
                    for i5 in range(4):
                        all_combinations.append([i1, i2, i3, i4, i5])

    for prices_idxs in all_combinations:
        reward = compute_reward(env, prices_idxs,    # compute reward for the price index
                                user_idx = user_idx)
        if highest_reward < reward:                  # update reward if bigger than previous one
            highest_reward = reward
            opt_prices_idxs = prices_idxs.copy()

    result = {'optimal_reward' : highest_reward,
              'combination'    : opt_prices_idxs}
    
    return result
  
