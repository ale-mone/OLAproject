import numpy as np
import copy

from environment import *


def fix_probs(Q, secondary, Lambda):
        
    new_Q = Q.copy()
    
    for i in range(max(np.shape(Q))):
        j = list(secondary.values())[i][1]
        new_Q[i, j] *= Lambda
    
    return new_Q


def find_user_prob_from_list(group_list, features_mat):
    
    user_prob = []
    
    for features_couple in group_list:
        i, j = features_couple
        user_prob.append(features_mat[i, j])
        
    return np.array(user_prob)


def change_demand_curve(env : Environment, new_res_prices_params):

    # update user reservation prices
    for user in env.users:
        user.res_prices_params = copy.deepcopy(new_res_prices_params)
        user.sample_res_prices()

    # update environment conversion rates
    #env.set_conversion_rates()
    conversion_rates = []
    cr_user = []
    cr_prod = []
    for user in env.users:
        for product in env.products:
            for price in product.prices:
                prod_idx = product.index
                cr_prod.append(user.get_buy_prob(price, prod_idx))
            cr_user.append(cr_prod.copy())
            cr_prod = []
        cr_user = np.array(cr_user)
        conversion_rates.append(cr_user.copy())
        cr_user = []
    env.theoretical_values['conversion_rates'] = copy.deepcopy(conversion_rates)
    

def compute_regret_ratio(optimal_reward, reward_history, regret_UB, algorithm):

    regret                 = optimal_reward - reward_history
    cumulative_regret      = np.cumsum(regret, axis = 1)
    mean_cumulative_regret = np.mean(cumulative_regret, axis = 0)
    regret_ratio           = mean_cumulative_regret/regret_UB

    print('Regret ratio - ' + algorithm + ' : %f' %regret_ratio[-1])


def get_features_list(cat_matrix):

    features_list = []
    n_groups = int(cat_matrix.max()) + 1

    for group in range(n_groups):
        i_values, j_values = np.where(cat_matrix == group)
        group_list = []
        for k in range(len(i_values)):
            group_list.append([i_values[k], j_values[k]])
        features_list.append(group_list.copy())

    return features_list

    