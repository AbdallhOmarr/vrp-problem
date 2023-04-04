
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
        duplicated_customers_route = []
        for route_idx, route in enumerate(self.routes):
            # print(f"customers:{route.get_customers_ids()}")
            for customer in route.get_customers_ids():

                if customer not in unique_customers:
                    unique_customers.append(customer)
                    unique_customers_idx.append(route_idx)
                else:
                    # print(f"customer:{customer} duplicated!")
                    duplicated_customers.append(customer)
                    duplicated_customers_route.append(route_idx)

        for customer, route_idx in zip(duplicated_customers, duplicated_customers_route):
            route = self.routes[route_idx]
            route_customers_array = route.customers_array
            rows_to_delete = []
            deleted = False  # flag variable to track whether a customer has been deleted from this route yet
            for i, row in enumerate(route_customers_array.copy()):
                if int(row[0]) == int(customer) and not deleted:
                    # print(f"customer:{customer}, row:{row[0]}")
                    rows_to_delete.append(i)
                    deleted = True  # set flag to True after first deletion
            route_customers_array = np.delete(route_customers_array, rows_to_delete, axis=0)
            self.routes[route_idx].customers_array = route_customers_array

        # print("new customers:")

        routes_to_delete = []
        for route in self.routes:
            # print(f"new customers: {route.get_customers_ids()}")
            if len(route.get_customers_ids()) == 0:
                routes_to_delete.append(route)

        for route in routes_to_delete:
            self.routes.remove(route)  # remove first occurrence using pop()

        # for route in self.routes:
        #     print(f"new customers after deleting empty rows:{route.get_customers_ids()}")

        for route in self.routes:
            route.get_customers_ids()
            route.update_route_variables()



        # self.update_variables()

                    

    def get_total_solution_distance(self):

        self.distances = []
        self.total_distance = 0
        for route in self.routes:
            self.total_distance +=route.get_route_distance()
            self.distances.append(route.get_route_distance())
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
        self.length_of_customers_array = []
        for route in self.routes:
            self.customers_served =np.concatenate((self.customers_served, route.get_customers_ids()))
            self.length_of_customers_array.append(len(route.customers_ids))
        self.total_customers_served = self.customers_served.shape[0]
        return self.total_customers_served 

    def get_number_of_routes(self):
        self.no_of_routes = len(self.routes)
        return self.no_of_routes

    def fitness_func(self):

        # if self.fitness_value is not None:
        #     return self.fitness_value
        self.update_variables()

        self.routes_fitness_value = 0
        for route in self.routes:
            self.routes_fitness_value+= route.route_fitness()

        
        feasibility = self.check_feasibility()
        self.distances  = np.array(self.distances)
        self.length_of_customers_array = np.array(self.length_of_customers_array)
        # - self.total_distance/np.mean(self.distances) 
        # /np.mean(self.length_of_customers_array)  + feasibility + self.routes_fitness_value 
        self.fitness_value = self.total_customers_served - self.total_distance/100
        return self.fitness_value



    def update_variables(self):
        self.prevent_duplicated_customers()
        self.get_total_solution_distance()
        self.check_feasibility()
        self.get_total_customers_served()


