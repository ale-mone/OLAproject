#----------------------#
#   IMPORT LIBRARIES   #
#----------------------#

# built-in libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle 

# main structures
from product import *
from user import *
from environment import *

# algorithms
from algorithms import *

# online learning algorithms
from TS_algorithms import *
from UCB_algorithms import *
from simulation import *

# utilities
from helpers import *
from plotters import *


#----------------------#
#  CREATE ENVIRONMENT  #
#----------------------#

np.random.seed(2022)

# import data
products_data    = pd.read_excel(r'Parameters_OLAProject.xlsx',
                                 sheet_name = 'Products',
                                 index_col = None, usecols = "A:J")

environment_data = pd.read_excel(r'Parameters_OLAProject.xlsx',
                                 sheet_name = 'Environment',
                                 index_col = None, usecols = "A:E")

user0_data       = pd.read_excel(r'Parameters_OLAProject.xlsx',
                                 sheet_name = 'Young Home Cook',
                                 index_col = None, usecols = "A:F")

user1_data       = pd.read_excel(r'Parameters_OLAProject.xlsx',
                                 sheet_name = 'Elder Home Cook',
                                 index_col = None, usecols = "A:F")

user2_data       = pd.read_excel(r'Parameters_OLAProject.xlsx',
                                 sheet_name = 'Experienced Cook',
                                 index_col = None, usecols = "A:F")

user_agg_data    = pd.read_excel(r'Parameters_OLAProject.xlsx',
                                 sheet_name = 'Aggregated',
                                 index_col = None, usecols = "A:F")

# 1. DEFINE PRODUCTS

# product names
prod_names = products_data['Names'][0:5].to_list()

# product prices
prod_prices = products_data.iloc[0:5, 1:5].values.astype(float).tolist()

# product production cost
prod_cost = products_data.iloc[0:5, 5].values.astype(float).tolist()

# product margins
prod_margins = products_data.iloc[0:5, 6:11].values.astype(float).tolist()

# create products vector
products = []
for i in range(len(prod_names)):
    products.append(Product(prod_names[i], prod_prices[i], prod_margins[i], i))

# probability to observe second secondary product
Lambda = environment_data.iloc[6, 1]

# secondary products
temp            = products_data.iloc[9:14, 0:3].values
secondary_prods = dict(zip(temp[:, 0], temp[:, 1:].astype(int).tolist()))
connectivity_matrix = products_data.iloc[17:23, 1:6].values.astype(int)


# 2. DEFINE USERS

# 2.1. Young Home Cook

# reservation prices parameters
temp              = user0_data.iloc[0:2, 0:6].values
res_prices_params = dict(zip(temp[:, 0], temp[:, 1:].astype(float).tolist()))
plot_prices(res_prices_params, products, prod_prices, 'Young Home Cook')

# average number of products bought (Poisson parameters)
n_prods_param = user0_data.iloc[3, 1:6].values.astype(float).tolist()
plot_n_prods(n_prods_param, products, 'Young Home Cook')

# transition probabilities
Q = user0_data.iloc[5:10, 1:6].values.astype(float).tolist()
Q = fix_probs(np.array(Q), secondary_prods, Lambda) # update probabilities of secondary products according to lambda
Q = Q * connectivity_matrix

# first product from which the user starts (Dirichelet parameters)
alphas = user0_data.iloc[11, 1:6].values.astype(float).tolist()

# create user
user0 = User('Young Home Cook', res_prices_params, alphas, n_prods_param, Q)

# 2.2. Elder Home Cook

# reservation prices parameters
temp              = user1_data.iloc[0:2, 0:6].values
res_prices_params = dict(zip(temp[:, 0], temp[:, 1:].astype(float).tolist()))
plot_prices(res_prices_params, products, prod_prices, 'Elder Home Cook')

# average number of products bought (Poisson parameters)
n_prods_param = user1_data.iloc[3, 1:6].values.astype(float).tolist()
plot_n_prods(n_prods_param, products, 'Elder Home Cook')

# transition probabilities
Q = user1_data.iloc[5:10, 1:6].values.astype(float).tolist()
Q = fix_probs(np.array(Q), secondary_prods, Lambda) # update probabilities of secondary products according to lambda
Q = Q * connectivity_matrix

# first product from which the user starts (Dirichelet parameters)
alphas = user1_data.iloc[11, 1:6].values.astype(float).tolist()

# create user
user1 = User('Elder Home Cook', res_prices_params, alphas, n_prods_param, Q)

# 2.3. Experienced Chef

# reservation prices parameters
temp              = user2_data.iloc[0:2, 0:6].values
res_prices_params = dict(zip(temp[:, 0], temp[:, 1:].astype(float).tolist()))
plot_prices(res_prices_params, products, prod_prices, 'Experienced Chef')

# average number of products bought (Poisson parameters)
n_prods_param = user2_data.iloc[3, 1:6].values.astype(float).tolist()
plot_n_prods(n_prods_param, products, 'Experienced Chef')

# transition probabilities
Q = user2_data.iloc[5:10, 1:6].values.astype(float).tolist()
Q = fix_probs(np.array(Q), secondary_prods, Lambda) # update probabilities of secondary products according to lambda
Q = Q * connectivity_matrix

# first product from which the user starts (Dirichelet parameters)
alphas = user2_data.iloc[11, 1:6].values.astype(float).tolist()

# create user
user2 = User('Experienced Chef', res_prices_params, alphas, n_prods_param, Q)

# 2.4. Aggregated User

# reservation prices parameters
temp                  = user_agg_data.iloc[0:2, 0:6].values
res_prices_params_agg = dict(zip(temp[:, 0], temp[:, 1:].astype(float).tolist()))
plot_prices(res_prices_params_agg, products, prod_prices, 'Aggregated')

# average number of products bought (Poisson parameters)
n_prods_param_agg = user_agg_data.iloc[3, 1:6].values.astype(float).tolist()
plot_n_prods(n_prods_param_agg, products, 'Aggregated')

# transition probabilities
Q_agg = user_agg_data.iloc[5:10, 1:6].values.astype(float).tolist()
Q_agg = fix_probs(np.array(Q_agg), secondary_prods, Lambda) # update probabilities of secondary products according to lambda
Q_agg = Q_agg * connectivity_matrix

# first product from which the user starts (Dirichelet parameters)
alphas_agg = user_agg_data.iloc[11, 1:6].values.astype(float).tolist()

# create user
user_agg = User('Aggregated', res_prices_params_agg, alphas_agg, n_prods_param_agg, Q_agg)


# 3. DEFINE ENVIRONMENT

# 1. Aggregated environment

# probability of having a specific feature
p_young           = environment_data.iloc[2, 1]
p_not_experienced = environment_data.iloc[1, 2]
feature_prob      = [p_young, p_not_experienced]

# matrix specifying the type to which the different categories of users belong
cat_matrix = environment_data.iloc[10:12, 3:5].values.astype(int)

# list of users
users = [user_agg]

# build environment
env_agg = Environment(users, products, secondary_prods, cat_matrix, feature_prob)


# 2. Complete environment

# probability of having a specific feature
p_young           = environment_data.iloc[2, 1]
p_not_experienced = environment_data.iloc[1, 2]
feature_prob      = [p_young, p_not_experienced]

# matrix specifying the type to which the different categories of users belong
cat_matrix = environment_data.iloc[10:12, 0:2].values.astype(int)

# list of users
users = [user0, user1, user2]

env = Environment(users, products, secondary_prods, cat_matrix, feature_prob)


#--------------------------------------------------------#
#  STEP 2: Optimization                                  #
#--------------------------------------------------------#

# compute greedy combination and reward

greedy_opt = greedy_optimizer(env_agg)

# compute optimal combination and reward
brute_force_opt = brute_force_optimizer(env_agg)

# print results
print('\n\nGreedy optimizer -')
print('Expected reward     : ', greedy_opt['expected_reward'])
print('Optimal combination : ', greedy_opt['combination'])

print('\nBrute force optimizer -')
print('Optimal reward      : ', brute_force_opt['optimal_reward'])
print('Optimal combination : ', brute_force_opt['combination'])
print('\n')

#--------------------------------------------------------#
#  STEP 3: Optimization with uncertain conversion rates  #
#--------------------------------------------------------#

# parameters for TS and UCB algorithm execution
n_runs      = 100
daily_users = 100
n_days      = 180

# upper bounds for the cumulative regret
eps = 1e-3
TS_learner   = TS_Learner(env_agg, [], {})
UCB_learner  = UCB_Learner(env_agg, [])
regretUB_TS  = TS_learner.compute_regret_bound(n_days, eps)
regretUB_UCB = UCB_learner.compute_regret_bound(n_days, eps)


# 1 -- Thompson Sampling

# parameters
# Note: initial beta parameters of the conversion rates are taken
#       in such a way that we have a uniform distribution on [0, 1]
a = np.ones((5, 4)) * 25
b = np.ones((5, 4))
init_conversion_rates_params = np.array([a, b])
unknowns_names = ['conversion_rates']
unknowns       = {'conversion_rates' : init_conversion_rates_params}

# define TS learner class
step3_TS = TS_Learner(env_agg, unknowns_names, unknowns)

for run in range(n_runs):
    step3_TS.run(n_days, daily_users)

# plot results
plot_results(brute_force_opt['optimal_reward'], step3_TS.reward_history, 'Thompson Sampling (step 3)')

# last iteration regret ratio with respect to theoretical upper bound
compute_regret_ratio(brute_force_opt['optimal_reward'], step3_TS.reward_history, regretUB_TS, 'Thompson Sampling (step 3)     ')

# 2 -- Upper Confidence Bound 1

# parameters
unknowns_names = ['conversion_rates']

# define UCB learner class
step3_UCB = UCB_Learner(env_agg, unknowns_names)

for run in range(n_runs):
    step3_UCB.run(n_days, daily_users)

# plot results
plot_results(brute_force_opt['optimal_reward'], step3_UCB.reward_history, 'Upper Confidence Bound (step 3)')

# last iteration regret ratio with respect to theoretical upper bound
compute_regret_ratio(brute_force_opt['optimal_reward'], step3_UCB.reward_history, regretUB_UCB, 'Upper Confidence Bound (step 3)')


# results
print('\nTrue values --')
print(env_agg.theoretical_values['conversion_rates'])

print('\nResults --')
print('Conversion rates:')
print('Thompson Sampling (step 3):')
print(step3_TS.params_history['conversion_rates'][-1])
print('Upper Confidence Bound (step 3):')
print(step3_UCB.params_history['conversion_rates'][-1])
print('\n\n')


#--------------------------------------------------------#
#  STEP 4: Optimization with uncertain conversion rates, #
#          alpha ratios and number of items              #
#          sold per product                              #
#--------------------------------------------------------#

# 1 -- Thompson Sampling

# parameters
# Note: initial beta parameters of the conversion rates are taken
#       in such a way that we have a uniform distribution on [0, 1]
a = np.ones((5, 4)) * 25
b = np.ones((5, 4))
init_conversion_rates_params = np.array([a, b])
init_alphas                  = np.ones((2, 5))
init_n_prods_param           = np.ones((2, 5))
unknowns_names = ['conversion_rates', 'alphas', 'n_prods_param']
unknowns       = {'conversion_rates' : init_conversion_rates_params,
                  'alphas'           : init_alphas,
                  'n_prods_param'    : init_n_prods_param}

# define TS learner class
step4_TS = TS_Learner(env_agg, unknowns_names, unknowns)

for run in range(n_runs):
    step4_TS.run(n_days, daily_users)

# plot results
plot_results(brute_force_opt['optimal_reward'], step4_TS.reward_history, 'Thompson Sampling (step 4)')

# last iteration regret ratio with respect to theoretical upper bound
compute_regret_ratio(brute_force_opt['optimal_reward'], step4_TS.reward_history, regretUB_TS, 'Thompson Sampling (step 4)     ')

# 2 -- Upper Confidence Bound 1

# parameters
unknowns_names = ['conversion_rates', 'alphas', 'n_prods_param']

# define UCB learner class
step4_UCB = UCB_Learner(env_agg, unknowns_names)

for run in range(n_runs):
    step4_UCB.run(n_days, daily_users)

# plot results
plot_results(brute_force_opt['optimal_reward'], step4_UCB.reward_history, 'Upper Confidence Bound (step 4)')

# last iteration regret ratio with respect to theoretical upper bound
compute_regret_ratio(brute_force_opt['optimal_reward'], step4_UCB.reward_history, regretUB_UCB, 'Upper Confidence Bound (step 4)')


# results
print('\nTrue values --')
print('Conversion rates:')
print(env_agg.theoretical_values['conversion_rates'])
print('Alpha ratios:')
print(env_agg.theoretical_values['alphas'])
print('Poisson parameters:')
print(env_agg.theoretical_values['n_prods_param'])

print('\nResults --')
print('Conversion rates:')
print('Thompson Sampling (step 4):')
print(step4_TS.params_history['conversion_rates'][-1])
print('Upper Confidence Bound (step 4):')
print(step4_UCB.params_history['conversion_rates'][-1])
print('Alpha ratios:')
print('Thompson Sampling (step 4):')
print(step4_TS.params_history['alphas'][-1])
print('Upper Confidence Bound (step 4):')
print(step4_UCB.params_history['alphas'][-1])
print('Poisson parameters:')
print('Thompson Sampling (step 4):')
print(step4_TS.params_history['n_prods_param'][-1])
print('Upper Confidence Bound (step 4):')
print(step4_UCB.params_history['n_prods_param'][-1])
print('\n\n')


#--------------------------------------------------------#
#  STEP 5: Optimization with uncertain graph weights     #
#--------------------------------------------------------#

# 1 -- Thompson Sampling

# parameters
# Note: initial beta parameters of the conversion rates are taken
#       in such a way that we have a uniform distribution on [0, 1]
a = np.ones((5, 4)) * 25
b = np.ones((5, 4))
c = np.ones((5, 2)) * 25
d = np.ones((5, 2))
init_conversion_rates_params = np.array([a, b])
init_graph_probs             = np.array([c, d])
unknowns_names = ['conversion_rates', 'graph_probs']
unknowns       = {'conversion_rates' : init_conversion_rates_params,
                  'graph_probs'      : init_graph_probs}

# define TS learner class
step5_TS = TS_Learner(env_agg, unknowns_names, unknowns)

for run in range(n_runs):
    step5_TS.run(n_days, daily_users)

# plot results
plot_results(brute_force_opt['optimal_reward'], step5_TS.reward_history, 'Thompson Sampling (step 5)')

# last iteration regret ratio with respect to theoretical upper bound
compute_regret_ratio(brute_force_opt['optimal_reward'], step5_TS.reward_history, regretUB_TS, 'Thompson Sampling (step 5)     ')

# 2 -- Upper Confidence Bound 1

# parameters
unknowns_names = ['conversion_rates', 'graph_probs']

# define UCB learner class
step5_UCB = UCB_Learner(env_agg, unknowns_names)

for run in range(n_runs):
    step5_UCB.run(n_days, daily_users)

# plot results
plot_results(brute_force_opt['optimal_reward'], step5_UCB.reward_history, 'Upper Confidence Bound (step 5)')

# last iteration regret ratio with respect to theoretical upper bound
compute_regret_ratio(brute_force_opt['optimal_reward'], step5_UCB.reward_history, regretUB_UCB, 'Upper Confidence Bound (step 5)')


# results
print('\nTrue values --')
print('Conversion rates:')
print(env_agg.theoretical_values['conversion_rates'])
print('Graph probabilities:')
print(env_agg.theoretical_values['graph_probs'])

print('\nResults --')
print('Conversion rates:')
print('Thompson Sampling (step 5):')
print(step5_TS.params_history['conversion_rates'][-1])
print('Upper Confidence Bound (step 5):')
print(step5_UCB.params_history['conversion_rates'][-1])
print('Graph probabilities:')
print('Thompson Sampling (step 5):')
print(step5_TS.params_history['graph_probs'][-1])
print('Upper Confidence Bound (step 5):')
print(step5_UCB.params_history['graph_probs'][-1])
print('\n\n')


#--------------------------------------------------------#
#  STEP 6: Non-stationary demand curve                   #
#--------------------------------------------------------#

# demand curve shifts
demand_curve_params = {}
temp_1 = user_agg_data.iloc[15:17, 0:6].values
dict_1 = dict(zip(temp_1[:, 0], temp_1[:, 1:].astype(float).tolist()))
demand_curve_params[user_agg_data.iloc[14, 1].astype(int)] = dict_1

temp_2 = user_agg_data.iloc[19:21, 0:6].values
dict_2 = dict(zip(temp_2[:, 0], temp_2[:, 1:].astype(float).tolist()))
demand_curve_params[user_agg_data.iloc[18, 1].astype(int)] = dict_2

# 0 -- Optimal reward history

# define vector of optimal rewards
optimal_rewards = np.zeros(n_days)
for day in range(n_days):
    if day in demand_curve_params.keys():
        change_demand_curve(env_agg, demand_curve_params[day])
    opt_reward = brute_force_optimizer(env_agg)['optimal_reward']
    optimal_rewards[day] = opt_reward

# 1 -- Sliding window Upper Confidence Bound 1

# parameters
SW_length      = 45
unknowns_names = ['conversion_rates']

# define UCB learner class
step6_SW_UCB = SW_UCB_Learner(env_agg, unknowns_names, SW_length, demand_curve_params)

for run in range(n_runs):
    # initialize starting demand curve
    change_demand_curve(env_agg, res_prices_params_agg)
    # run algorithm
    step6_SW_UCB.run(n_days, daily_users)
 
# plot results
plot_results_CD(optimal_rewards, step6_SW_UCB.reward_history, demand_curve_params, 'Sliding Window Upper Confidence Bound (step 6)')

# 2 -- Change detection Upper Confidence Bound 1

# parameters
change_detection_params = {}
change_detection_params['T']         = 45
change_detection_params['w']         = 0.02
change_detection_params['threshold'] = 0.3
unknowns_names = ['conversion_rates']

# define UCB learner class
step6_CD_UCB = CD_UCB_Learner(env_agg, unknowns_names, change_detection_params, demand_curve_params)

for run in range(n_runs):
    # initialize starting demand curve
    change_demand_curve(env_agg, res_prices_params_agg)
    # run algorithm
    step6_CD_UCB.run(n_days, daily_users)

# plot results
plot_results_CD(optimal_rewards, step6_CD_UCB.reward_history, demand_curve_params, 'Change Detection Upper Confidence Bound (step 6)')


#--------------------------------------------------------#
#  STEP 7: Context generation                            #
#--------------------------------------------------------#

# 0 -- Optimal rewards

# compute optimal combination and reward - disaggregated version
disagg_brute_force_opt = brute_force_optimizer(env, disaggregated = True)

# compute optimal combination and reward - aggregated version
agg_brute_force_opt_naive = brute_force_optimizer(env, disaggregated = False)['optimal_reward']

# compute optimal combination and reward - aggregated version from disaggregated result
agg_brute_force_opt_mean = np.sum(disagg_brute_force_opt['optimal_reward'] * env.user_prob)

# print results
print('\n\nBrute force optimizer - disaggregated version')
print('Optimal reward      : ', disagg_brute_force_opt['optimal_reward'])
print('Optimal combination : ', disagg_brute_force_opt['combination'])

print('\nBrute force optimizer - aggregated version found as mean')
print('Optimal reward      : ', agg_brute_force_opt_mean)

print('\nBrute force optimizer - aggregated version')
print('Optimal reward      : ', agg_brute_force_opt_naive)
print('\n')

# 1 -- Thompson Sampling

# parameters
# Note: initial beta parameters of the conversion rates are taken
#       in such a way that we have a uniform distribution on [0, 1]
a = np.ones((5, 4)) * 25
b = np.ones((5, 4))
init_conversion_rates_params = np.array([a, b])
init_alphas                  = np.ones((2, 5))
init_n_prods_param           = np.ones((2, 5))
unknowns_names = ['conversion_rates', 'alphas', 'n_prods_param']
unknowns       = {'conversion_rates' : init_conversion_rates_params,
                  'alphas'           : init_alphas,
                  'n_prods_param'    : init_n_prods_param}

# context parameters
confidence    = 0.05
init_cat_prob = np.ones((2, 2)) * 0.25
init_context  = np.array([[0, 0], [0, 0]])
context_unknowns = {'cat_prob' : init_cat_prob,
                    'context'  : init_context}

# define TS learner class
step7_TS = CG_TS_Learner(env, unknowns_names, unknowns, context_unknowns, confidence)

for run in range(n_runs):
    step7_TS.run(n_days, daily_users)

# plot results
plot_results_CG(agg_brute_force_opt_mean, agg_brute_force_opt_naive, step7_TS.reward_history, 'Thompson Sampling (step 7)')

# 2 -- Upper Confindence Bound 1

# parameters
init_conversion_rates_params = np.ones((2, 5, 4))
init_alphas                  = np.ones((2, 5))
init_n_prods_param           = np.ones((2, 5))
unknowns_names = ['conversion_rates', 'alphas', 'n_prods_param']
unknowns       = {'conversion_rates' : init_conversion_rates_params,
                  'alphas'           : init_alphas,
                  'n_prods_param'    : init_n_prods_param}

# context parameters
confidence    = 0.05
init_cat_prob = np.ones((2, 2)) * 0.25
init_context  = np.array([[0, 0], [0, 0]])
context_unknowns = {'cat_prob' : init_cat_prob,
                    'context'  : init_context}

# define UCB learner class
step7_UCB = CG_UCB_Learner(env, unknowns_names, unknowns, context_unknowns, confidence)

for run in range(n_runs):
    step7_UCB.run(n_days, daily_users)

# plot results
plot_results_CG(agg_brute_force_opt_mean, agg_brute_force_opt_naive, step7_UCB.reward_history, 'Upper Confidence Bound (step 7)')

