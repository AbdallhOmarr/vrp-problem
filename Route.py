import math
import random
import pandas as pd
import numpy as np 
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

    def check_time_constrain(self):

        #current time is between ready time and due date
        # ready_time <= current time <= due date
        #initial current time = first customer readytime
        #ready time row[4]
        #due date row[5]

        self.time_constrain = True
        self.current_time = self.customers_array[0,4]
        for row in self.customers_array:
            if (not self.current_time <= row[4]) or (not self.current_time<=row[5]):
                self.time_constrain =False

                return self.time_constrain
            else:
                self.current_time+=row[6]
        return self.time_constrain

    def check_capacity_constrain(self):
        self.load = self.get_load()
        if self.load>200:
            self.capacity_constrain = False
        else:
            self.capacity_constrain = True
            
        return self.capacity_constrain



    def get_route_distance(self):
        
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


