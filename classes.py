import pandas as pd
import re
import matplotlib.pyplot as plt
import math
import numpy as np


class Customer:
    def __init__(self, id, x, y, demand, ready_time, due_date, service_time):
        self.id = id
        self.x = x
        self.y = y
        self.demand = demand
        self.ready_time = ready_time
        self.due_date = due_date
        self.service_time = service_time
        self.in_route = False
        self.interior_customer = False

    def __eq__(self, other):
        # Check if other is a Customer object
        if isinstance(other, Customer):
            # Compare ids
            return self.id == other.id
        else:
            # Return NotImplemented for other types
            return NotImplemented

    def __str__(self):
        return str(self.id)


class Vehicle:
    def __init__(self, id, capacity):
        self.id = id
        self.capacity = capacity
        self.load = 0
        self.customers = []
        self.vehicle_time = 18
        self.route = Route()

    def add_customers(self, customers):
        for customer in customers:
            self.customers.append(customer)
            self.load += customer.demand
            self.vehicle_time += customer.service_time
            self.route.add_to_route(customer)
            customer.in_route = True
            self.update_customers()

    def check_if_customer_is_on_vehicle(self, customer):
        if customer in self.customers:
            return True
        else:
            return False

    def get_customer_data(self, customer):
        print(self.customers[self.customers.index(customer)])
        return self.customers[self.customers.index(customer)]

    def get_load(self):
        return self.load

    def get_vehicle_time(self):
        return self.vehicle_time

    def check_customer_load(self, customers):
        customers_load = customers[0].demand + customers[1].demand
        if customers_load + self.load > self.capacity:
            return False
        else:
            return True

    def check_customer_time_window(self, customers):
        current_time = self.vehicle_time

        if current_time >= customers[0].ready_time and self.vehicle_time <= customers[0].due_date:
            current_time += customers[0].service_time
            if current_time >= customers[1].ready_time and self.vehicle_time <= customers[1].due_date:
                current_time += customers[1].service_time
                return True
        elif current_time >= customers[1].ready_time and self.vehicle_time <= customers[1].due_date:
            current_time += customers[1].service_time
            if current_time >= customers[0].ready_time and self.vehicle_time <= customers[0].due_date:
                current_time += customers[0].service_time
                return True

        return False

    def update_vehicle_time(self, customers):
        self.current_vehicle_time += customers[0].service_time + \
            customers[1].service_time

    def update_vehicle_load(self, customers):
        self.load += customers[0].demand + customers[1].demand

    def update_customers(self):
        for customer in self.customers:
            customer.in_route = True
            if self.route.check_if_customer_is_interior(customer):
                customer.interior_customer = True
            else:
                customer.interior_customer = False

    def __eq__(self, other):
        # Check if other is a Customer object
        if isinstance(other, Vehicle):
            # Compare ids
            return self.id == other.id
        else:
            # Return NotImplemented for other types
            return NotImplemented

    def __str__(self):
        return str(self.id)


class Route:
    def __init__(self):
        print("route initialized")
        self.route = ""
        self.customers = []

    def get_interior_customers(self):
        interior_customers = self.customers[1:-1]
        return interior_customers

    def get_route(self):
        if len(self.customers) != 0:
            ids = [str(int(customer.id)) for customer in self.customers]
            self.route = "-".join(ids)  # Join ids with commas
            return self.route
        else:
            return self.route

    def add_to_route(self, customer):
        self.customers.append(customer)
        self.get_route()
        return True

    def check_if_customer_is_interior(self, customer):
        if customer in self.get_interior_customers():
            return True
        else:
            return False

    def check_if_customer_is_in_route(self, customer):
        if customer in self.customers:
            return True
        else:
            return False

    def __str__(self):
        return self.route
