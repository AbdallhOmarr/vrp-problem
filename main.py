import pandas as pd
import re
import matplotlib.pyplot as plt
import math
import numpy as np


def get_distance_between_two_points(x1, y1, x2, y2):
    """
    Calculates the Euclidean distance between two points.
    """
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


class Dispatcher:
    def __init__(self):
        """ This is responsible for creating routes, adding, delete, update customers to routes  """
        self.customers = []
        self.vehicles = []
        self.routes = []
        self.total_distance = 0

    def read_customer_data(self, txt_file):
        # Function to convert txt file into a pandas dataframe
        data = []
        with open(txt_file, "r") as file:
            lines = file.readlines()  # Read all lines in the file
            column_parsed = False
            column_names = []
            if column_parsed == False:
                # Extract column names from first line
                words = re.findall("\S+", lines[0])
                for word in words:
                    column_names.append(word)
                column_parsed = True

            for line in lines[1:]:  # Loop through lines, skipping the first line
                # Extract data points from line
                dataline = re.findall("\S+", line)
                line_dict = {}
                for i, point in enumerate(dataline):
                    # Convert data point to float and store in dictionary
                    line_dict[column_names[i]] = float(point)
                data.append(line_dict)  # Append dictionary to list of data

        # Create pandas dataframe from list of dictionaries
        self.customers_df = pd.DataFrame(data)

    def create_customers(self):
        try:
            for i, v in df.iterrows():
                if i == 0:
                    self.DEPOT = Customer(v["CUST_NO"], v["XCOORD"], v["YCOORD"],
                                          v["DEMAND"], v["READY_TIME"], v["DUE_DATE"], v["SERVICE_TIME"])
                else:
                    self.customers.append(Customer(
                        v["CUST_NO"], v["XCOORD"], v["YCOORD"], v["DEMAND"], v["READY_TIME"], v["DUE_DATE"], v["SERVICE_TIME"]))
            return True
        except:
            return False

    def create_routes(self, customer):
        # do i need to create initial routes for each customer?
        route = Route()
        route.add_customer(customer)

    def calculate_distance_matrix(self):
        n_customers = len(self.customers)
        dist_matrix = np.zeros((n_customers, n_customers))

        # Loop over each pair of customers
        for i in range(n_customers):
            for j in range(n_customers):
                # Calculate the Euclidean distance between the XCOORD and YCOORD values of the two customers
                x_diff = self.customers_df['XCOORD'][i] - \
                    self.customers_df['XCOORD'][j]
                y_diff = self.customers_df['YCOORD'][i] - \
                    self.customers_df['YCOORD'][j]
                dist = math.sqrt(math.pow(x_diff, 2) + math.pow(y_diff, 2))

                # Store the calculated distance in the corresponding position in the distance matrix
                dist_matrix[i][j] = dist

        return dist_matrix

    def calculate_total_distance(self):
        for route in self.routes:
            self.total_distance += route.distance
        return self.total_distance


class Route:
    def __init__(self,):
        """ this class to create routes, store data"""
        self.current_time = 0
        self.customers = []
        self.distance = 0
        self.CAPACITY = 200  # constant
        self.load = load

    def add_customer(self, customer):
        self.customers.append(customer)
        self.update_current_time()

        self.update_distance()

    def delete_customer(self, customer):
        self.customers = self.customers.pop(self.customers.index(customer))

    def update_current_time(self):
        for customer in self.customers:
            service_start_time = max(self.current_time, customer.ready_time)
            service_end_time = service_start_time + customer.service_time
            self.current_time = service_end_time

        return self.current_time

    def get_current_time(self, customer):
        service_start_time = max(self.current_time, customer.ready_time)
        service_end_time = service_start_time + customer.service_time
        current_time = service_end_time
        return current_time

    def check_time_window(self):
        current_time = 0
        for customer in self.customers:
            current_time = self.get_current_time(customer)
            if customer.ready_time > current_time:
                return False
            if customer.due_date < current_time:
                return False
        return True

    def update_distance(self):
        # distance between customers + distance between boundary customers and DEPOT
        n = len(self.customers)
        distances = []
        for i, customer in enumerate(self.customers):
            if i == n:
                break

            next_customer = self.customers[i+1]
            distance_between_cust_next_customer = get_distance_between_two_points(
                customer.xcoord, customer.ycoord, next_customer.xcoord, next_customer.ycoord)
            distances.append(distance_between_cust_next_customer)
        # distance between DEPOT and boundry customers
        b1 = self.customers[0]
        b2 = self.customers[1]

        distance_between_b1_DEPOT = get_distance_between_two_points(
            b1.xcoord, b1.ycoord, self.DEPOT.xcoord, self.DEPOT.ycoord)
        distance_between_b2_DEPOT = get_distance_between_two_points(
            b2.xcoord, b2.ycoord, self.DEPOT.xcoord, self.DEPOT.ycoord)
        distances.append(distance_between_b1_DEPOT)
        distances.append(distance_between_b2_DEPOT)

        self.distance = 0
        for distance in distances:
            self.distance += distance

        return self.distance

    def check_capacity(self):
        self.update_load()
        if self.load >= self.capacity:
            return False
        else:
            return True

    def update_load(self):
        self.load = 0
        for customer in self.customers:
            self.load += customer.demand
        return self.load


class Customer:
    def __init__(self, cust_no, xcoord, ycoord, demand, ready_time, due_date, service_time):
        """ This class to create customers, store its data"""
        self.cust_no = cust_no
        self.xcoord = xcoord
        self.ycoord = ycoord
        self.demand = demand
        self.ready_time = ready_time
        self.due_date = due_date
        self.service_time = service_time
