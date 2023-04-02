
import pandas as pd
import re
import matplotlib.pyplot as plt
import math
import numpy as np
import random
import pandas as pd
from sklearn.cluster import KMeans
import re


class Solution:
    def __init__(self, routes):
        self.routes = routes
        self.fitness_value = None

    def prevent_duplicated_customers(self):
        # this function to remove duplicated customers in other routes
        # check if customer is another route
        # if customers in another route remove the customer in the first route
        unique_customers = []
        unique_customers_idx = []
        duplicated_customers = []
        for route_idx, route in enumerate(self.routes):
            for customer in route.get_customers_ids():
                if customer not in unique_customers:
                    unique_customers.append(customer)
                    unique_customers_idx.append(route_idx)
                else:
                    duplicated_customers.append(route_idx)

        for route_idx in duplicated_customers:
            route = self.routes[route_idx]
            route_customers_array = route.customers_array
            rows_to_delete = []
            for i, row in enumerate(route_customers_array.copy()):
                if row[0] in unique_customers:
                    unique_customers.remove(row[0])
                else:
                    rows_to_delete.append(i)

            route_customers_array = np.delete(route_customers_array, rows_to_delete, axis=0)
            self.routes[route_idx].customers_array = route_customers_array
            self.routes[route_idx].update_route_variables()

        self.update_variables()

                    

    def get_total_solution_distance(self):
        self.total_distance = 0
        for route in self.routes:
            self.total_distance +=route.get_route_distance()

        return self.total_distance 

    def check_feasibility(self):
        self.time_constrain = True
        self.capacity_constrain = True
        for route in self.routes:
            if not route.check_capacity_constrain():
                self.capacity_constrain = False
            if not route.check_time_constrain():
                self.time_constrain = False

        self.feasibility = self.capacity_constrain * self.time_constrain

        return self.feasibility


    def get_total_customers_served(self):
        self.customers_served = []
        for route in self.routes:
            self.customers_served =np.concatenate((self.customers_served, route.get_customers_ids()))

        self.total_customers_served = self.customers_served.shape[0]
        return self.total_customers_served 

    def get_number_of_routes(self):
        self.no_of_routes = len(self.routes)
        return self.no_of_routes

    def fitness_func(self):
        if self.fitness_value is not None:
            return self.fitness_value

        self.update_variables()
        self.fitness_value = self.check_feasibility() * (self.total_customers_served*1000 - self.total_distance)
        return self.fitness_value



    def update_variables(self):
        self.get_total_solution_distance()
        self.check_feasibility()
        self.get_total_customers_served()


