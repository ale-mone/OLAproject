import numpy as np
from scipy.stats import norm
import copy


class User:

    """ Class used to define the structure of a user
    """
    
    def __init__(self, name, res_prices_params : dict, alphas, n_prods_param, Q):
        self.name              = name                                   # name
        
        self.res_prices_params = res_prices_params                      # reservation price parameters
        self.res_prices        = [0 for x in range(len(n_prods_param))] # reservation prices
        self.sample_res_prices()
        
        self.alphas            = alphas                                 # probability to start from a product (Dirichelet parameters)
        self.sampled_alphas    = [x/sum(alphas) for x in alphas]        # probability to start from a product (Dirichelet parameters)
        
        self.n_prods_param     = np.array(n_prods_param)                # average number of products bought (Poisson parameter)
        
        self.trans_probs       = Q                                      # transition probabilities
        
        self.visited_products  = []                                     # visited products
        
    
    def sample_res_prices(self):
        # sample reservation prices according to the parameters
        self.res_prices = np.random.normal(self.res_prices_params['mu'], self.res_prices_params['sigma'])
            
    def sample_alphas(self):
        # sample alphas according to a Dirichelet distribution
        self.sampled_alphas = np.random.dirichlet(self.alphas, 1)
        
    def sample_n_prods(self, prod_idx):
        # sample how many products are bought
        return np.random.poisson(self.n_prods_param[prod_idx]) + 1

    def starting_product(self):
        # select a starting product according to the alphas
        return np.random.choice(list(range(len(self.alphas))), p = self.sampled_alphas.reshape(-1))
    
    def get_buy_prob(self, price, prod_idx):
        # get probability of buying
        mu    = self.res_prices_params['mu'][prod_idx]
        sigma = self.res_prices_params['sigma'][prod_idx]
        return 1 - norm.cdf(price, loc = mu, scale = sigma)

    def buy(self, price, prod_idx):
        # sample whether the product is bought or not
        return np.random.binomial(1, self.get_buy_prob(price, prod_idx))

    def empty_visited_products(self):
        # empty the set of visited products
        self.visited_products = []

        
