import numpy as np


class Product:

    """ Class used to define the structure of a product
    """

    def __init__(self, name, prices, margins, index):
        self.name    = name    # name
        self.prices  = prices  # prices
        self.margins = margins # margins
        self.index   = index   # position in list

    def get_name(self):
        return self.name
    
    def get_prices(self, index):
        return self.prices[index]

    def get_margins(self, index):
        return self.margins[index]

