import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import factorial


def plot_prices(res_prices_params, products, prod_prices, algorithm):
    
    prices = prod_prices
    x = np.arange(0, 1.2 * np.max(prices), 0.01)
    colors = ['forestgreen', 'darkorange', 'dodgerblue', 'darkviolet', 'crimson']
    for i in range(len(products)):
        plt.plot(x, norm.pdf(x, res_prices_params['mu'][i], res_prices_params['sigma'][i]), label = products[i].get_name(), color = colors[i])
        plt.plot(prices[i], 0.05 * np.ones(4), 'o', color = colors[i])
    plt.legend()
    plt.title('Reservation prices - ' + algorithm)
    plt.show()
    

def plot_n_prods(n_prods_param, products, algorithm):
    
    def poisson_pdf(x, mu):
        mu = mu + 1
        return np.exp(-mu) * np.power(mu, x) / factorial(x)
        
    x = np.arange(0, 15, 0.01)
    colors = ['forestgreen', 'darkorange', 'dodgerblue', 'darkviolet', 'crimson']
    for i in range(len(n_prods_param)):
        plt.plot(x, poisson_pdf(x, n_prods_param[i]), label = products[i].get_name(), color = colors[i])
    plt.legend()
    plt.title('Number of products - ' + algorithm)
    plt.show()
    

def plot_results(optimal_reward, reward_history, algorithm):
    
    regret                 = optimal_reward - reward_history
    
    std_deviation_regret   = np.std(regret, axis = 0)
    
    cumulative_regret      = np.cumsum(regret, axis = 1)
    mean_cumulative_regret = np.mean(cumulative_regret, axis = 0)
    std_cumulative_regret  = np.std(cumulative_regret, axis = 0)
    
    mean_reward            = np.mean(reward_history, axis = 0)
    
    # cumulative regret
    plt.figure
    plt.grid(True, color = 'whitesmoke', linestyle = '-', linewidth = 1)
    plt.plot(mean_cumulative_regret, color = 'black') 
    plt.xlabel('t (days)')
    plt.title('Cumulative regret - ' + algorithm)
    plt.show()
    
    # standard deviation
    plt.figure
    plt.grid(True, color = 'whitesmoke', linestyle = '-', linewidth = 1)
    plt.plot(std_deviation_regret, color = 'darkgrey')
    plt.xlabel('t (days)')
    plt.title('Standard deviation - ' + algorithm)
    plt.show()

    # cumulative regret and standard deviation
    plt.figure
    plt.grid(True, color = 'whitesmoke', linestyle = '-', linewidth = 1)
    plt.fill_between(range(np.shape(reward_history)[1]),
                      mean_cumulative_regret - std_cumulative_regret,
                      mean_cumulative_regret + std_cumulative_regret,
                      alpha = 0.4, color = 'darkgrey')
    plt.plot(mean_cumulative_regret, color = 'black') 
    plt.xlabel('t (days)')
    plt.title('Cumulative regret with standard deviation - \n' + algorithm)
    plt.show()
    
    # mean reward versus optimal reward
    plt.figure
    plt.grid(True, color = 'whitesmoke', linestyle = '-', linewidth = 1)
    plt.axhline(optimal_reward, color = 'forestgreen')
    plt.plot(mean_reward, color = 'black')
    plt.xlabel('t (days)')
    plt.ylabel('Expected Reward(t)')
    plt.title('Mean reward - ' + algorithm)
    plt.legend(['Optimal Reward', 'Mean Expected Reward'])
    plt.show()
    

def plot_results_CG(optimal_reward, optimal_reward_naive, reward_history, algorithm):
    
    regret                 = optimal_reward - reward_history
    
    std_deviation_regret   = np.std(regret, axis = 0)
    
    cumulative_regret      = np.cumsum(regret, axis = 1)
    mean_cumulative_regret = np.mean(cumulative_regret, axis = 0)
    std_cumulative_regret  = np.std(cumulative_regret, axis = 0)
    
    mean_reward            = np.mean(reward_history, axis = 0)
    
    # cumulative regret
    plt.figure
    plt.grid(True, color = 'whitesmoke', linestyle = '-', linewidth = 1)
    plt.plot(mean_cumulative_regret, color = 'black') 
    plt.xlabel('t (days)')
    plt.title('Cumulative regret - ' + algorithm)
    plt.show()
    
    # standard deviation
    plt.figure
    plt.grid(True, color = 'whitesmoke', linestyle = '-', linewidth = 1)
    plt.plot(std_deviation_regret, color = 'darkgrey')
    plt.xlabel('t (days)')
    plt.title('Standard deviation - ' + algorithm)
    plt.show()

    # cumulative regret and standard deviation
    plt.figure
    plt.grid(True, color = 'whitesmoke', linestyle = '-', linewidth = 1)
    plt.fill_between(range(np.shape(reward_history)[1]),
                      mean_cumulative_regret - std_cumulative_regret,
                      mean_cumulative_regret + std_cumulative_regret,
                      alpha = 0.4, color = 'darkgrey')
    plt.plot(mean_cumulative_regret, color = 'black') 
    plt.xlabel('t (days)')
    plt.title('Cumulative regret with standard deviation - \n' + algorithm)
    plt.show()
    
    # mean reward versus optimal reward
    plt.figure
    plt.grid(True, color = 'whitesmoke', linestyle = '-', linewidth = 1)
    plt.axhline(optimal_reward, color = 'forestgreen')
    plt.axhline(optimal_reward_naive, color = 'forestgreen', linestyle = 'dashdot', alpha  = 0.5)
    plt.plot(mean_reward, color = 'black')
    plt.xlabel('t (days)')
    plt.ylabel('Expected Reward(t)')
    plt.title('Mean reward - ' + algorithm)
    plt.legend(['Optimal Reward (mean of disaggregated)', 'Optimal Reward (aggregated)', 'Mean Expected Reward'])
    plt.show()
    

def plot_results_CD(optimal_reward, reward_history, demand_curve_params, algorithm):
    
    regret                 = optimal_reward - np.array(reward_history)
    
    std_deviation_regret   = np.std(regret, axis = 0)
    
    cumulative_regret      = np.cumsum(regret, axis = 1)
    mean_cumulative_regret = np.mean(cumulative_regret, axis = 0)
    std_cumulative_regret  = np.std(cumulative_regret, axis = 0)
    
    mean_reward            = np.mean(reward_history, axis = 0)
    
    # changes
    changes_times  = list(demand_curve_params.keys())
    changes_number = len(changes_times)
    
    # cumulative regret
    plt.figure
    plt.grid(True, color = 'whitesmoke', linestyle = '-', linewidth = 1)
    plt.plot(mean_cumulative_regret, color = 'black')
    plt.vlines(changes_times, [0] * changes_number, [max(mean_cumulative_regret)] * changes_number, color = 'firebrick', label = 'changes time', linestyle = 'dashdot', linewidth = 1)
    plt.xlabel('t (days)')
    plt.title('Cumulative regret - ' + algorithm)
    plt.show()
    
    # standard deviation
    plt.figure
    plt.grid(True, color = 'whitesmoke', linestyle = '-', linewidth = 1)
    plt.plot(std_deviation_regret, color = 'darkgrey')
    plt.vlines(changes_times, [0] * changes_number, [max(std_deviation_regret)] * changes_number, color = 'firebrick', label = 'changes time', linestyle = 'dashdot', linewidth = 1)
    plt.xlabel('t (days)')
    plt.title('Standard deviation - ' + algorithm)
    plt.show()
    
    # cumulative regret and standard deviation
    plt.figure
    plt.grid(True, color = 'whitesmoke', linestyle = '-', linewidth = 1)
    plt.fill_between(range(np.shape(reward_history)[1]),
                      mean_cumulative_regret - std_cumulative_regret,
                      mean_cumulative_regret + std_cumulative_regret,
                      alpha = 0.4, color = 'darkgrey')
    plt.plot(mean_cumulative_regret, color = 'black')
    plt.vlines(changes_times, [0] * changes_number, [max(mean_cumulative_regret + std_cumulative_regret)] * changes_number, color = 'firebrick', label = 'changes time', linestyle = 'dashdot', linewidth = 1)
    plt.xlabel('t (days)')
    plt.title('Cumulative regret with standard deviation - \n' + algorithm)
    plt.show()
    
    # mean reward versus optimal reward
    plt.figure
    plt.grid(True, color = 'whitesmoke', linestyle = '-', linewidth = 1)
    plt.plot(np.array(optimal_reward), color = 'forestgreen')
    plt.plot(mean_reward, color = 'black')
    plt.xlabel('t (days)')
    plt.ylabel('Expected Reward(t)')
    plt.title('Mean reward - ' + algorithm)
    plt.legend(['Optimal Reward', 'Mean Expected Reward'])
    plt.show()

 