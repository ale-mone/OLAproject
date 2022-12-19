from environment import *


class Network:

    """ Network class used to represent the website
    """

    def __init__(self, info_dict = None):

        """ Initialize object of class Network
        """
        
        # initialize path with basic informations
        if info_dict is None:
            self.products_seen    = []
            self.path_probability = 1
            self.path_margin      = 0
            self.link             = [-1]
            self.link_prob        = [0]

        # copy the given informations
        else:
            self.products_seen    = info_dict['products_seen']
            self.path_probability = info_dict['path_probability']
            self.path_margin      = info_dict['path_margin']
            self.link             = info_dict['link']
            self.link_prob        = info_dict['link_prob']

    def copy_info(self):
        
        """ Deep copy information from an object of type path into a dictionary
             -> this is important for containers of type list
        """

        info_dict = {'products_seen'    : self.products_seen.copy(),
                     'path_probability' : self.path_probability,
                     'path_margin'      : self.path_margin,
                     'link'             : self.link.copy(),
                     'link_prob'        : self.link_prob.copy()}

        return info_dict

    def expected_return(self):
        return self.path_probability * self.path_margin


def explore_path(env, paths_list, path, primary_idx, prices_idxs, user_idx):
        
    # initialize the path at the beginning with the initial values
    if path is None:
        path = Network()

    # check if the path has jumped to an unexplored secondary product
    if primary_idx == -1:
        
        # define a copy of the current path
        path0 = Network(path.copy_info())

        # check if the secondary (last linked node to the path) has already been visited
        if path0.link[-1] in path0.products_seen:

            # if it has been visited remove it from the secondary products
            path0.link.pop()      # remove linking
            path0.link_prob.pop() # remove probability associated to the link

            # check if other nodes are connected to the current path
            if path0.link[-1] == -1: # no nodes are connected to the path...
                # ...add the current path to the list of paths and stop
                paths_list.append(path0)
            else:                    # some nodes are connected to the path...
                # ...explore the next node
                explore_path(env, paths_list, path0, -1, prices_idxs, user_idx)
            return

        # if the secondary product (last linked node to the path) has not already been visited we can click it...
        # (Case 1) click on the secondary (last linked node to the path)
        # (Case 2) not click on it

        # initialize the possible paths we can choose copying the current path
        path1 = Network(path.copy_info())
        path2 = Network(path.copy_info())

        # CASE 1 =================================================== #
        # (Case 1) CLICK => explore a new path with the secondary product as primary

        # 1.a. take out the seconary node from the list of links and consider it as the new primary
        new_primary = path1.link.pop()

        # 1.b. update the probability of the path by taking out the probability of clicking on the secondary
        #       product from the list of links probabilities and updating the probability of following the current path
        path1.path_probability *= path1.link_prob.pop()

        # 1.c. explore the new path
        explore_path(env, paths_list, path1, new_primary, prices_idxs, user_idx)

        # CASE 2 =================================================== #
        # (Case 2) NO CLICK...
        # (Case 2.1) stop the exploration since there is not another secondary node to explore
        # (Case 2.2) continue to explore since there is another secondary node to explore

        # 2.a. take out the seconary node from the list of links
        path2.link.pop()

        # 2.b. update the probability of the path by taking out the probability of clicking on the secondary
        #       product from the list of links probabilities and updating the probability of following
        #       the current path (with 1 - P(clicking))
        path2.path_probability *= 1 - path2.link_prob.pop()
        
        # CASE 2.1 | CASE 2.2    -*- -*- -*- -*- -*- -*- -*- -*- -*- #
        # choose between (Case 2.1) and (Case 2.2) by checking if there is another secondary product to explore
        if path2.link[-1] == -1: # (Case 2.1) there is no other secondary product to explore...
            # ...add the current node to the list of paths
            paths_list.append(path2)
        else:                    # (Case 2.2) there is another secondary product to explore...
            # ...explore the next node
            explore_path(env, paths_list, path2, -1, prices_idxs, user_idx)
        return

    # add the current primary to the list of the seen products of the current path
    path.products_seen.append(primary_idx)

    # retrieve the indexes of the secondary products
    primary_name    = env.products[primary_idx].name
    first_scnd_idx  = env.secondary_prods[primary_name][0]
    second_scnd_idx = env.secondary_prods[primary_name][1]

    # compute the probability to buy the considered primary product
    p_buy_primary = env.conversion_rates[user_idx][primary_idx, prices_idxs[primary_idx]]

    # compute the expected margin obtained if the primary is bought as:
    #  E[margin] = margin * number of items bought
    margin     = env.products[primary_idx].get_margins(prices_idxs[primary_idx])
    exp_margin = margin * ( env.n_prods_param[user_idx][primary_idx] )
    
    # compute the probabilities of clicking on the secondary product given that the primary is bought
    q_first_scnd  = env.graph_probs[user_idx][primary_idx, first_scnd_idx]
    q_second_scnd = env.graph_probs[user_idx][primary_idx, second_scnd_idx]

    # create a path according to the choice of the user of buying or not the primary product...
    # (Case 1) primary product is not bought
    # (Case 2) primary product is bought

    # CASE 1 =================================================== #
    # (Case 1) NOT BOUGHT...
    # (Case 1.1) stop the exploration since there is not a secondary node to explore
    # (Case 1.2) continue to explore since there is a secondary node to explore

    # 1.a. initialize the possible paths we can choose copying the current path
    path1 = Network(path.copy_info())

    # 1.b. update the path probability according to the event of not buying the product
    path1.path_probability *= 1 - p_buy_primary

    # CASE 1.1 | CASE 1.2    -*- -*- -*- -*- -*- -*- -*- -*- -*- #
    # choose between (Case 1.1) and (Case 1.2) by checking if there is a secondary product to explore
    if path1.link[-1] == -1: # (Case 1.1) there is not a secondary product to explore...
        # ...add the current node to the list of paths
        paths_list.append(path1)
    else:                    # (Case 1.2) there is a secondary product to explore...
        # ...explore the next node
        explore_path(env, paths_list, path1, -1, prices_idxs, user_idx)

    # CASE 2  =================================================== #
    # (Case 2) BOUGHT...
    # (Case 2.1) there is no secondary products to explore
    # (Case 2.2) the first secondary product cannot be explored
    # (Case 2.3) the second secondary product cannot be explored
    # (Case 2.4) both secondary products can be explored

    # 2.a. initialize the possible paths we can choose copying the current path
    path2 = Network(path.copy_info())

    # 2.b. update the path probability according to the event of buying the product
    path2.path_probability *= p_buy_primary

    # 2.c. update the path expected margin according to the event of buying the product
    path2.path_margin += exp_margin

    # 2.d. check if there is some secondaries to explore
    first_scnd_seen  = first_scnd_idx  in path.products_seen
    second_scnd_seen = second_scnd_idx in path.products_seen

    # CASE 2.1 -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- #
    # (Case 2.1) NO SECONDARIES TO EXPLORE...
    if first_scnd_seen and second_scnd_seen:
        # check if other nodes are connected to the current path
        if path2.link[-1] == -1: # no nodes are connected to the path...
            # ...add the current path to the list of paths and stop
            paths_list.append(path2)
        else:                    # some nodes are connected to the path...
            # ...explore the next node
            explore_path(env, paths_list, path2, -1, prices_idxs, user_idx)
        return

    # CASE 2.2 -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- #
    # (Case 2.2) EXPLORE SECOND SECONDARY...
    # (Case 2.2.1) click on the second secondary and explore the related path
    # (Case 2.2.2) do not click on the second secondary
    if first_scnd_seen and not second_scnd_seen:

        # CASE 2.2.1. -- - -- - -- - -- - -- - -- - -- - -- - -- - #
        # (Case 2.2.1) CLICK

        # 2.2.1.a. initialize the possible path we can choose copying the current path
        path3 = Network(path2.copy_info())
        
        # 2.2.1.b. update the path probability according to the event of clicking the product
        path3.path_probability *= q_second_scnd

        # 2.2.1.c. explore the created path
        explore_path(env, paths_list, path3, second_scnd_idx, prices_idxs, user_idx)

        # CASE 2.2.2. -- - -- - -- - -- - -- - -- - -- - -- - -- - #
        # (Case 2.2.2) NO CLICK...
        # (Case 2.2.2.1) no secondaries are present so we stop exploring
        # (Case 2.2.2.2) secondaries are present so we continue exploring

        # 2.2.2.a. initialize the possible path we can choose copying the current path
        path4 = Network(path2.copy_info())

        # 2.2.2.b. update the path probability according to the event of not clicking the product
        path4.path_probability *= 1 - q_second_scnd

        # CASE 2.2.2.1 | CASE 2.2.2.2  - _ - _ - _ - _ - _ - _ - _ #
        # choose between (Case 2.2.2.1) and (Case 2.2.2.2) by checking if there is a secondary product to explore
        if path4.link[-1] == -1: # no nodes are connected to the path...
            # ...add the current path to the list of paths and stop
            paths_list.append(path4)
        else:                    # some nodes are connected to the path...
            # ...explore the next node
            explore_path(env, paths_list, path4, -1, prices_idxs, user_idx)
        return

    # CASE 2.3 -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- #
    # (Case 2.3) EXPLORE FIRST SECONDARY...
    # (Case 2.3.1) click on the first secondary and explore the related path
    # (Case 2.3.2) do not click on the first secondary
    if second_scnd_seen and not first_scnd_seen:

        # CASE 2.3.1. -- - -- - -- - -- - -- - -- - -- - -- - -- - #
        # (Case 2.3.1) CLICK

        # 2.3.1.a. initialize the possible path we can choose copying the current path
        path3 = Network(path2.copy_info())
        
        # 2.3.1.b. update the path probability according to the event of clicking the product
        path3.path_probability *= q_first_scnd

        # 2.3.1.c. explore the created path
        explore_path(env, paths_list, path3, first_scnd_idx, prices_idxs, user_idx)

        # CASE 2.3.2. -- - -- - -- - -- - -- - -- - -- - -- - -- - #
        # (Case 2.3.2) NO CLICK...
        # (Case 2.3.2.1) no secondaries are present so we stop exploring
        # (Case 2.3.2.2) secondaries are present so we continue exploring

        # 2.3.2.a. initialize the possible path we can choose copying the current path
        path4 = Network(path2.copy_info())

        # 2.3.2.b. update the path probability according to the event of not clicking the product
        path4.path_probability *= 1 - q_first_scnd

        # CASE 2.3.2.1 | CASE 2.3.2.2  - _ - _ - _ - _ - _ - _ - _ #
        # choose between (Case 2.3.2.1) and (Case 2.3.2.2) by checking if there is a secondary product to explore
        if path4.link[-1] == -1: # no nodes are connected to the path...
            # ...add the current path to the list of paths and stop
            paths_list.append(path4)
        else:                    # some nodes are connected to the path...
            # ...explore the next node
            explore_path(env, paths_list, path4, -1, prices_idxs, user_idx)
        return

    # CASE 2.4 -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- #
    # (Case 2.4) EXPLORE FIRST OR SECOND SECONDARY...
    # (Case 2.4.1) click on the first secondary (append the second secondary to the list of possible links)
    # (Case 2.4.2) do not click on the first secondary but click on the second secondary
    # (Case 2.4.3) do not click on any secondary

    # CASE 2.4.1 -- - -- - -- - -- - -- - -- - -- - -- - -- - -- - #
    # (Case 2.4.1) CLICK FIRST SECONDARY and APPEND SECOND SECONDARY

    # 2.4.1.a. initialize the possible path we can choose copying the current path
    path3 = Network(path2.copy_info())

    # 2.4.1.b. update the path probability according to the event of clicking the first secondary product
    path3.path_probability *= q_first_scnd

    # 2.4.1.c. append the second secondary to the list of links
    path3.link.append(second_scnd_idx)
    path3.link_prob.append(q_second_scnd)

    # 2.4.1.d. continue exploring the path
    explore_path(env, paths_list, path3, first_scnd_idx, prices_idxs, user_idx)
    
    # CASE 2.4.2 -- - -- - -- - -- - -- - -- - -- - -- - -- - -- - #
    # (Case 2.4.2) NO CLICK FIRST SECONDARY but CLICK SECOND SECONDARY

    # 2.4.2.a. initialize the possible path we can choose copying the current path
    path4 = Network(path2.copy_info())

    # 2.4.2.b. update the path probability according to the event of not clicking the
    #           first secondary product but clicking the second secondary
    path4.path_probability *= ( 1 - q_first_scnd ) * q_second_scnd

    # 2.4.2.c. continue exploring the path
    explore_path(env, paths_list, path4, second_scnd_idx, prices_idxs, user_idx)
    
    # CASE 2.4.3 -- - -- - -- - -- - -- - -- - -- - -- - -- - -- - #
    # (Case 2.4.3) NO CLICKS...
    # (Case 2.4.3.1) no secondaries are present so we stop exploring
    # (Case 2.3.3.2) secondaries are present so we continue exploring

    # 2.4.3.a. initialize the possible path we can choose copying the current path
    path5 = Network(path2.copy_info())

    # 2.4.3.b. update the path probability according to the event of not clicking any secondary product
    path5.path_probability *= ( 1 - q_first_scnd ) * ( 1 - q_second_scnd )

    # CASE 2.4.3.1 | CASE 2.4.3.1  - _ - _ - _ - _ - _ - _ - _ - _ #
    # choose between (Case 2.4.3.1) and (Case 2.4.3.2) by checking if there is a secondary product to explore
    if path5.link[-1] == -1: # no nodes are connected to the path...
        # ...add the current path to the list of paths and stop
        paths_list.append(path5)
    else:                    # some nodes are connected to the path...
        # ...explore the next node
        explore_path(env, paths_list, path5, -1, prices_idxs, user_idx)
    return
  
