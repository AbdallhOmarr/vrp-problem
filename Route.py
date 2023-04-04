import math
import random
import pandas as pd
import numpy as np 
import copy
def get_distance_between_two_points(x1,y1,x2,y2):
    x_diff = x2-x1
    y_diff = y2-y1
    dist = math.sqrt(math.pow(x_diff, 2) + math.pow(y_diff, 2))
    return dist


class Route:
    def __init__(self,customers_array,depot):
        self.customers_array = customers_array
        self.depot = depot
        self.current_time = 0

    def get_customers_ids(self):
        self.customers_ids = self.customers_array[:,0]
        return self.customers_ids

    def get_load(self):
        self.load = np.sum(self.customers_array[:,3])
        return self.load

    def route_fitness(self):
        
        self.route_fitness_value = 0
        if self.time_constrain and self.capacity_constrain:
            self.route_fitness_value = 1 
        
        else:
            self.route_fitness_value = - 1
        return self.route_fitness_value

    def drop_customers(self,row):
        self.customers_array = np.delete(self.customers_array,row,axis=0)

    def repair_operator(self):
        # removing unfulfilled customers from route to make it feasible 

        customers_array = copy.deepcopy(self.customers_array)

        cluster_changed = np.concatenate(([False], customers_array[1:, 8] != customers_array[:-1, 8]))

        print(customers_array.shape)
        customers_array = np.column_stack((customers_array, cluster_changed))

        # Initialize current_time array with zeros
        current_time = np.zeros(len(customers_array))

        # Set the first element of current_time to the READY_TIME of the first row
        current_time[0] = customers_array[0, 4]

        # Loop over the remaining rows in customers_array
        for i in range(1, len(customers_array)):
            if cluster_changed[i]:
                # If the cluster has changed, set current_time to the READY_TIME of the current row
                current_time[i] = customers_array[i, 4]
            else:
                # If the cluster has not changed, calculate the current_time based on the previous row
                previous_time = current_time[i-1]
                previous_service_time = customers_array[i-1, 6]
                ready_time = customers_array[i, 4]
                current_time[i] = max(ready_time, previous_time + previous_service_time)

        # Add the current_time array as a new column in customers_array
        customers_array = np.column_stack((customers_array, current_time))

        # and current_time is a NumPy array containing the current_time values
        time_variation = customers_array[:, 5] - current_time
        time_constrain = current_time <= customers_array[:, 5]

        # Convert the boolean values in time_constrain to 1's and 0's
        time_constrain = time_constrain.astype(int)

        # Add the time_variation and time_constrain arrays as new columns in customers_array
        customers_array = np.column_stack((customers_array, time_variation, time_constrain))

        return customers_array

    def check_time_constrain(self):

        #current time is between ready time and due date
        # ready_time <= current time <= due date
        #initial current time = first customer readytime
        #ready time row[4]
        #due date row[5]
        # print(f"route customers:{self.customers_ids}")
        self.time_constrain = True          # initialize time_constrain to True
        self.current_time = self.customers_array[0,4]   # set the current_time to the ready time of the first customer
        for row in self.customers_array:   # loop through all customers
            if self.current_time > row[5]:  # if the current_time is after the due date of the customer
                self.time_constrain = False   # set time_constrain to False
                return self.time_constrain   # return False
            else:
                self.current_time = max(self.current_time, row[4]) + row[6]   # update the current_time to be the maximum of the ready time of the customer and the current time, plus the service time of the customer
        return self.time_constrain   # return True if all customers have valid time constraints

    def check_capacity_constrain(self):
        self.load = self.get_load()
        if self.load>200:
            self.capacity_constrain = False
        else:
            self.capacity_constrain = True
            
        return self.capacity_constrain



    def get_route_distance(self):
        if self.customers_array.shape[0] == 0:
            print("customers array is empty?!")
            return 0
            
        #i will get the distance between each customer and the next customer in the array
        #distance row[7]
        self.total_distance = 0 
        for i in range(self.customers_array.shape[0]):
            if i == self.customers_array.shape[0]-1:
                break

            distance = get_distance_between_two_points(self.customers_array[i,1],self.customers_array[i,2],self.customers_array[i+1,1],self.customers_array[i+1,2])
            self.total_distance+=distance

        #calculate distance betwen first customer and depot and last customer and depot 
        distance_1 = get_distance_between_two_points(self.customers_array[0,1],self.customers_array[0,2],self.depot[1],self.depot[2])
        distance_2 = get_distance_between_two_points(self.customers_array[-1,1],self.customers_array[-1,2],self.depot[1],self.depot[2])

        self.total_distance += distance_1
        self.total_distance += distance_2 
        return self.total_distance
    def update_route_variables(self):
        self.get_load()
        self.get_customers_ids()
        self.get_route_distance()
        self.check_capacity_constrain()
        self.check_time_constrain()

    def mutate(self,mutation_rate):
        #get randomly two indices in the customers array and exchange them 
        if self.customers_array.shape[0]<=1:
            return False
        
        if random.random() > mutation_rate:
            return False

        idx_1 = random.randint(0, self.customers_array.shape[0] - 1)
        idx_2 = random.randint(0, self.customers_array.shape[0] - 1)

        first_row = self.customers_array[idx_1,:]
        second_row = self.customers_array[idx_2,:]

        self.customers_array[idx_1,:] = second_row
        self.customers_array[idx_2,:] = first_row 
        self.update_route_variables()
        return True


