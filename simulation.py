from user import *
from environment import *

import numpy as np
import copy


def simulate_user(env : Environment, user_type, arms_pulled, prod_idx, user_result_simulation):
    
    # user considered
    user = env.users[user_type]
    

    # PRIMARY PRODUCT --
    
    # select current price of the considered product
    price_idx = arms_pulled[prod_idx]
    
    # check if the product is bought
    primary_bought = user.buy(env.products[prod_idx].get_prices(price_idx), prod_idx)
    
    # update simulation parameters
    if 'conversion_rates' in user_result_simulation:
        user_result_simulation['conversion_rates'][0][prod_idx] += primary_bought
        user_result_simulation['conversion_rates'][1][prod_idx] += 1
    
    # if it is not bought stop, otherwise continue with the simulation
    if not primary_bought:
        return
    
    # check number of products bought
    n_prod_bought = user.sample_n_prods(prod_idx)
    
    # update simulation parameters
    if 'n_prods_param' in user_result_simulation:
        user_result_simulation['n_prods_param'][0][prod_idx] += n_prod_bought
        user_result_simulation['n_prods_param'][1][prod_idx] += 1
    
    
    # SECONDARY PRODUCTS --
    
    # retrieve secondary products
    first_scnd_idx   = env.secondary_prods[env.products[prod_idx].get_name()][0]
    first_scnd_name  = env.products[first_scnd_idx]
    second_scnd_idx  = env.secondary_prods[env.products[prod_idx].get_name()][1]
    second_scnd_name = env.products[first_scnd_idx]
    
    # check if the first secondary is clicked
    first_clicked = (first_scnd_name not in user.visited_products) * (np.random.uniform() < user.trans_probs[prod_idx, first_scnd_idx])
        
    # update simulation parameters
    if 'graph_probs' in user_result_simulation:
        
        # if the product was clicked increase number of clicks
        if first_clicked:
            user_result_simulation['graph_probs'][0][prod_idx, first_scnd_idx] += 1
            
        # if the product was observed (namely it had not been already observed) increase number of observations
        if first_scnd_name not in user.visited_products:
            user_result_simulation['graph_probs'][1][prod_idx, first_scnd_idx] += 1
            
    # if it is clicked add it to the list and continue simulation with the first secondary as primary
    if first_clicked:
        user.visited_products.append(first_scnd_name)
        simulate_user(env, user_type, arms_pulled, first_scnd_idx, user_result_simulation)
        
    # check if the second secondary is clicked
    second_clicked = (second_scnd_name not in user.visited_products) * (np.random.uniform() < user.trans_probs[prod_idx, second_scnd_idx])
        
    # update simulation parameters
    if 'graph_probs' in user_result_simulation:
        
        # if the product was clicked increase number of clicks
        if second_clicked:
            user_result_simulation['graph_probs'][0][prod_idx, second_scnd_idx] += 1
            
        # if the product was observed (namely it had not been already observed) increase number of observations
        if second_scnd_name not in user.visited_products:
            user_result_simulation['graph_probs'][1][prod_idx, second_scnd_idx] += 1
            
    # if it is clicked add it to the list and continue simulation with the first secondary as primary
    if second_clicked:
        user.visited_products.append(second_scnd_name)
        simulate_user(env, user_type, arms_pulled, second_scnd_idx, user_result_simulation)
    
    return

    
def user_step(env : Environment, user_type, arms_pulled, user_result_simulation):
    
    # user considered
    user = env.users[user_type]
    
    # sample reservation prices
    user.sample_res_prices()
    
    # sample starting product
    starting_prod = user.starting_product()
    
    # initialize list of visited products
    user.visited_products = [env.products[starting_prod]]
    
    # update simulation parameters
    if 'alphas' in user_result_simulation:
        user_result_simulation['alphas'][starting_prod] += 1
        
    # simulate user interaction
    simulate_user(env, user_type, arms_pulled, starting_prod, user_result_simulation)
    
    # empty the visited products of the structure
    user.empty_visited_products()


def simulate_daily_interaction(env : Environment, arms_pulled, daily_users, unknown_names):
    
    # number of products
    n = len(arms_pulled)
    
    # initialize simulation parameters for a specific user type
    user_result_simulation = {name : [] for name in unknown_names}
    
    if unknown_names.count('conversion_rates'):
        # 0: number of times products i was sold
        # 1: number of times products i was observed
        user_result_simulation['conversion_rates'] = np.zeros((2, n))
    
    if unknown_names.count('alphas'):
        # number of times product i was shown as primary
        user_result_simulation['alphas'] = np.zeros(n)
    
    if unknown_names.count('n_prods_param'):
        # 0: number of products of type i that were sold
        # 1: number of times products i was sold
        user_result_simulation['n_prods_param'] = np.zeros((2, n))
    
    if unknown_names.count('graph_probs'):
        # 0: number of clicks for each secondary product j given primary product i
        # 1: number of visualizations for each secondary product j given primary product i
        user_result_simulation['graph_probs'] = np.array([np.zeros((n, n)), np.zeros((n, n))])
    
    # initialize total simulation result (non aggregated)
    temp_result_simulation = []
    
    # generate result of the simulation for each user
    for user in env.users:
        temp_result_simulation.append(copy.deepcopy(user_result_simulation))
        
    # initialize total simulation result (aggregated)
    result_simulation = copy.deepcopy(user_result_simulation)
        
    # generate alphas for each user for the given day
    for user in env.users:
        user.sample_alphas()
        
    # simulate number of users
    n_users = np.random.poisson(lam = daily_users)
    
    # cycle over all the users
    for user_i in range(n_users):
        
        # sample user type
        if len(env.users) > 1:
            user_idxs = list(range(len(env.users)))
            user_type = np.random.choice(user_idxs, p = env.user_prob)
        else:
            user_type = 0
        
        # simulate website interaction for a given user
        user_step(env, user_type, arms_pulled, temp_result_simulation[user_type])
    
    # aggregate results
    if unknown_names.count('conversion_rates'):
        result_simulation['conversion_rates'] = np.sum([user_result['conversion_rates'] for user_result in temp_result_simulation], axis = 0)

    if unknown_names.count('alphas'):
        result_simulation['alphas'] = np.sum([user_result['alphas'] for user_result in temp_result_simulation], axis = 0)
        
    if unknown_names.count('n_prods_param'):
        result_simulation['n_prods_param'] = np.sum([user_result['n_prods_param'] for user_result in temp_result_simulation], axis = 0)
        
    if unknown_names.count('graph_probs'):
        result_simulation['graph_probs'][0] = np.sum([user_result['graph_probs'][0] for user_result in temp_result_simulation], axis = 0)
        result_simulation['graph_probs'][1] = np.sum([user_result['graph_probs'][1] for user_result in temp_result_simulation], axis = 0)
        
    # save number of users simulated
    result_simulation['n_users'] = n_users
    
    return result_simulation


def simulate_daily_interaction_CG(env : Environment, arms_pulled_list, daily_users, context):
    
    # number of products
    n = len(arms_pulled_list[0])
    
    # initialize simulation parameters for a specific user type
    user_result_simulation = {
                              # number of users that interacted with the site
                              'n_users'          : 0,
                                   
                              # 0: number of times products i was sold
                              # 1: number of times products i was observed
                              'conversion_rates' : np.zeros((2, n)),
                                   
                              # number of times product i was shown as primary
                              'alphas'           : np.zeros(n),
                                   
                              # 0: number of products of type i that were sold
                              # 1: number of times products i was sold
                              'n_prods_param'    : np.zeros((2, n))}
    
    # initialize simulation results
    result_simulation = {'00' : copy.deepcopy(user_result_simulation),
                         '01' : copy.deepcopy(user_result_simulation),
                         '10' : copy.deepcopy(user_result_simulation),
                         '11' : copy.deepcopy(user_result_simulation)}
    
    # generate alphas for each user for the given day
    for user in env.users:
        user.sample_alphas()
        
    # simulate number of users
    n_users = np.random.poisson(lam = daily_users)
    
    # cycle over all the users
    for user_i in range(n_users):
        
        # sample first feature
        feature_1 = np.random.binomial(1, 1 - np.sum(env.cat_prob[:, 0]))
        
        # sample second feature
        feature_2 = np.random.binomial(1, 1 - np.sum(env.cat_prob[0, :]))
        
        # write category indexes as key
        cat_idxs_as_key = str(feature_1) + str(feature_2)
        
        # increase number of user of the simulated category
        result_simulation[cat_idxs_as_key]['n_users'] += 1
        
        # find simulated user type 
        user_type = int(env.cat_matrix[feature_1, feature_2])
        
        # find arms pulled for simulated user type according to current context
        arms_pulled = arms_pulled_list[int(context[feature_1, feature_2])]
        
        # simulate website interaction for a given user
        user_step(env, user_type, arms_pulled, result_simulation[cat_idxs_as_key])
    
    return result_simulation
  
