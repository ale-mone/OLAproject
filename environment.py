import numpy as np
import copy


class Environment:
    
    """ Class used to define the structure of the environment
    """

    def __init__(self, users, products, secondary_prods, cat_matrix, feature_prob):
        self.users              = users                                    # list of users types
        
        self.products           = products                                 # list of products types
        self.secondary_prods    = secondary_prods                          # secondary products associated to each product
        
        self.user_prob          = find_user_prob(cat_matrix, feature_prob) # relative frequency of the user type
        self.cat_prob           = find_cat_prob(feature_prob)              # relative frequency of each user category
        self.cat_matrix         = cat_matrix                               # matrix specifying the type to which the different categories of users belong
        
        # User parameters defined as list:
        # conversion_rates = probability to buy a product given a price (for a specific user)
        # alphas           = probability to start from a product for a user (Dirichelet parameters)
        # n_prods_param    = average number of products bought by a user (Poisson parameters)
        # graph_probs      = transition probabilities of a specific user
        self.conversion_rates   = []
        self.set_conversion_rates()
        self.alphas             = [[x/sum(users[i].alphas) for x in users[i].alphas] for i in range(len(users))]
        self.n_prods_param      = [[x+1 for x in users[i].n_prods_param] for i in range(len(users))]
        self.graph_probs        = [users[i].trans_probs for i in range(len(users))]
        self.theoretical_values = {'conversion_rates' : copy.deepcopy(self.conversion_rates),
                                   'alphas'           : copy.deepcopy(self.alphas),
                                   'n_prods_param'    : copy.deepcopy(self.n_prods_param),
                                   'graph_probs'      : copy.deepcopy(self.graph_probs)}
            
    def set_conversion_rates(self):
        
        self.conversion_rates = []
        cr_user = []
        cr_prod = []
        
        for user in self.users:
            for product in self.products:
                for price in product.prices:
                    prod_idx = product.index
                    cr_prod.append(user.get_buy_prob(price, prod_idx))
                cr_user.append(cr_prod.copy())
                cr_prod = []
            cr_user = np.array(cr_user)
            self.conversion_rates.append(cr_user.copy())
            cr_user = []
            
    def update_conversion_rates(self, group_list):
        
        self.conversion_rates = []
        
        for feat_idxs in group_list:
            i, j = feat_idxs
            user_idx = self.cat_matrix[i, j]
            self.conversion_rates.append(copy.deepcopy(self.theoretical_values['conversion_rates'][user_idx]))
    
    def update_alphas(self, group_list):
        
        self.alphas = []
        
        for feat_idxs in group_list:
            i, j = feat_idxs
            user_idx = self.cat_matrix[i, j]
            self.alphas.append(copy.deepcopy(self.theoretical_values['alphas'][user_idx]))
        
    def update_n_prods_param(self, group_list):
        
        self.n_prods_param = []
        
        for feat_idxs in group_list:
            i, j = feat_idxs
            user_idx = self.cat_matrix[i, j]
            self.n_prods_param.append(copy.deepcopy(self.theoretical_values['n_prods_param'][user_idx]))
        
    def update_graph_probs(self, group_list):
        
        self.graph_probs = []
        
        for feat_idxs in group_list:
            i, j = feat_idxs
            user_idx = self.cat_matrix[i, j]
            self.graph_probs.append(copy.deepcopy(self.theoretical_values['graph_probs'][user_idx]))
        

def find_user_prob(cat_matrix, feature_prob):
    
    cat_prob = find_cat_prob(feature_prob)
    user_prob = [0. for i in range(cat_matrix.max()+1)]
    rows, cols = cat_matrix.shape
    
    for i in range(rows):
        for j in range(cols):
            user_prob[cat_matrix[i][j]] += cat_prob[i][j]
            
    return user_prob
    
    
def find_cat_prob(feature_prob):
    
    result = np.array([[feature_prob[0]], [1-feature_prob[0]]]) * np.array([feature_prob[1], 1-feature_prob[1]])
    
    return result.transpose()

  
