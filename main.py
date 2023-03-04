import pandas as pd
import re
import matplotlib.pyplot as plt
import math
import numpy as np
import random


def get_distance_between_two_points(x1, y1, x2, y2):
    """
    Calculates the Euclidean distance between two points.
    """
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


class Dispatcher:
    def __init__(self):
        """ This is responsible for creating routes, adding, delete, update customers to routes  """
        self.customers = []
        self.routes = []
        self.total_distance = 0
        self.route_nums = []

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
            for i, v in self.customers_df.iterrows():
                if i == 0:
                    self.DEPOT = Customer(v["CUST_NO"], v["XCOORD"], v["YCOORD"],
                                          v["DEMAND"], v["READY_TIME"], v["DUE_DATE"], v["SERVICE_TIME"])
                else:
                    self.customers.append(Customer(
                        v["CUST_NO"], v["XCOORD"], v["YCOORD"], v["DEMAND"], v["READY_TIME"], v["DUE_DATE"], v["SERVICE_TIME"]))
            return True
        except:
            return False

    def create_initial_routes(self):
        for customer in self.customers:
            route_num = random.randint(1, 10000)
            while route_num in self.route_nums:
                route_num = random.randint(1, 10000)
            self.route_nums.append(route_num)
            route = Route(route_num, self.DEPOT)
            route.add_customer(customer)
            self.routes.append(route)
            self.update_total_solution_distance()

    def create_routes(self, customer):
        # do i need to create initial routes for each customer?
        route_num = random.randint(1, 10000)
        while route_num in self.route_nums:
            route_num = random.randint(1, 10000)
        self.route_nums.append(route_num)
        route = Route(route_num, self.DEPOT)
        route.add_customer(customer)
        self.routes.append(route)
        self.update_total_solution_distance()

    def create_route(self, customers):
        route_num = random.randint(1, 10000)
        while route_num in self.route_nums:
            route_num = random.randint(1, 10000)
        route = Route(route_num, self.DEPOT)
        for customer in customers:
            route.add_customer(customer)
        return route

    def merge_routes(self, route1, route2, customer1, customer2):
        lst_of_customers = []
        for customer in route1.customers:
            lst_of_customers.append(customer)

        for customer in route2.customers:
            lst_of_customers.append(customer)

        route_num = random.randint(1, 10000)
        while route_num in self.route_nums:
            route_num = random.randint(1, 10000)

        route = Route(route_num, self.DEPOT)
        for customer in lst_of_customers:
            route.add_customer(customer)

        return route

    def check_route_constrains(self, route):
        # i have capacity constrain and time constrain.
        if route.check_capacity() and route.check_time_window():
            return True
        else:
            return False

    def calculate_distance_matrix(self):
        n_customers = len(self.customers)
        self.dist_matrix = np.zeros((n_customers, n_customers))

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
                self.dist_matrix[i][j] = dist

        return self.dist_matrix

    def calculate_savings(self):
        """
        Calculates the savings for each pair of customers

        Args:
            customers: A pandas dataframe representing customer data with the columns:
                CUST_NO, XCOORD, YCOORD, DEMAND, READY_TIME, DUE_DATE, SERVICE_TIME
            dist_matrix: A 2D numpy array representing the distance matrix between customers

        Returns:
            A list of tuples representing the savings for each pair of customers, in descending order
        """
        # Calculate savings
        n_customers = len(self.customers)
        savings = []
        for i in range(2, n_customers):
            for j in range(i+1, n_customers):
                saving = self.dist_matrix[0][i] + \
                    self.dist_matrix[0][j] - self.dist_matrix[i][j]
                savings.append(((i, j), saving))

        # Sort savings in descending order
        self.sorted_savings = sorted(savings, key=lambda x: x[1], reverse=True)

        return self.sorted_savings

    def update_total_solution_distance(self):
        self.total_distance = 0
        for route in self.routes:
            route.update_distance()
            self.total_distance += route.distance
        return self.total_distance

    def get_percentage_of_customers_served(self):
        customers_unserved = []
        for r in self.routes:
            if len(r.customers) == 1:
                customers_unserved.append(r.customers)
        return (len(self.customers)-len(customers_unserved))/len(self.customers)*100

    def formulate_solution(self):
        route_num = 1
        self.update_total_solution_distance()
        for r in self.routes:
            r.update_load()
            r.update_distance()
            route_str = f"Route {route_num} (Load {r.load}): "
            route_str += "1-"
            route_str += " ->".join([f"{c.cust_no}" for c in r.customers])
            route_str += "-1"
            distance = r.distance

            route_str += f" (Distance {round(distance,2)})"
            print(route_str)
            route_num += 1

        # Print total distance and percentage of customers served
        print(f"Total distance: {round(self.total_distance,2)}")
        print(
            f"Percentage of customers served: {round(self.get_percentage_of_customers_served(),2)}%")

    def get_route_by_customer(self, customer_num):
        for route in self.routes:
            if customer_num in route.customer_nums:
                return route
        print("customer is not in sol !! ")
        print(f"customer:{customer_num}")
        return False

    def generate_initial_solution(self):
        new_routes = self.routes.copy()
        self.calculate_savings()
        for (i, j), s in self.sorted_savings:
            # Check if customers i and j are already in the same route
            capacity_constrain = False
            time_constrain = False

            i_route = self.get_route_by_customer(i)
            j_route = self.get_route_by_customer(j)
            if i_route == j_route:
                continue

            print(
                f"i:{i}, i_route:{i_route.customer_nums}, j:{j}, j_route:{j_route.customer_nums}")

            new_route = self.merge_routes(i_route, j_route, i, j)
            print(f"new route customers: {new_route.customer_nums}")
            new_route.customers.sort(
                key=lambda customer: customer.due_date)
            new_route.customer_nums = [
                customer.cust_no for customer in new_route.customers]
            print(f"new route customers sorted: {new_route.customer_nums}")

            # check capacity constrain
            if new_route.check_capacity():
                print("checking capacity")
                capacity_constrain = True

            if new_route.check_time_window():
                print("checking time constrain")
                time_constrain = True
            else:
                print("time constrain failed ")
                new_route = self.merge_routes(j_route, i_route, j, i)
                if new_route.check_capacity():
                    print("checking capacity after reversing route")
                    capacity_constrain = True
                else:
                    capacity_constrain = False

                if new_route.check_time_window():
                    print("checking time constrain")
                    time_constrain = True

            if time_constrain and capacity_constrain:
                merged = True
                print("constrains are not violated")
                print(f"i_route:{i_route.id}\nj_route:{j_route.id}")
                print(
                    f"new routes: before adding the merged route:{[route.id for route in self.routes]}")
                print(f"i:{i}-j:{j}")
                self.routes.append(new_route)
                if i_route in self.routes:
                    self.routes.remove(i_route)
                else:
                    print(f"i_route is not in new routes!")
                if j_route in self.routes:
                    self.routes.remove(j_route)
                else:
                    print(f"j route is not in routes")

                print(
                    f"new routes: after adding the merged route:{[route.id for route in self.routes]}")

        print("done loops")
        # self.routes = new_routes.copy()
        self.update_total_solution_distance()
        self.best_solution = Solution(self.routes)
        self.best_solution.update_distance()
        self.best_solution.update_fitness()

    def generate_initial_population(self, POPULATION_SIZE):
        # Step 1: Generate initial population
        print(f"routes:{self.routes}")
        self.population = []
        for i in range(POPULATION_SIZE):
            shuffled_routes = random.sample(self.routes, len(self.routes))
            print(f"shuffiled_routes={shuffled_routes}")
            solution = Solution(shuffled_routes)
            solution.update_fitness()

            self.population.append(solution)
        return self.population

    def generate_population(self, POPULATION_SIZE, routes):
        # BEST Solution routes to get the new population
        print(f"routes:{routes}")
        self.population = []
        for i in range(POPULATION_SIZE):
            shuffled_routes = random.sample(routes, len(routes))
            print(f"shuffiled_routes={shuffled_routes}")
            solution = Solution(shuffled_routes)
            solution.update_fitness()

            self.population.append(solution)
        return self.population

    def calculate_total_distance(self, solution):
        total_distance = 0
        for route in solution:
            distance = route.update_distance()
            total_distance += distance
        return total_distance

    def count_unserved_customers(self, solution):
        customers_unserved = []
        for r in solution:
            if len(r.customers) == 1:
                customers_unserved.append(r.customers)
        return len(customers_unserved)/len(self.customers)*100

    def fitness_func(self, solution):
        # it should return fitness_value
        # it should check capacity, time constrains for the whole solutions:
        total_distance = 0
        customer_served = []
        for route in solution:
            if len(r.customers) == 1:
                pass
            else:
                for customer in route.customers:
                    customer_served.append(customer)

            distance = route.update_distance()
            total_distance += distance
            if route.check_time_window() and route.check_capacity():
                pass
            else:
                return 0

        customer_served_percentage = (len(customers_served)/100)*100

        fitness_value = customer_served_percentage*1000-total_distance

        return fitness_value

    def rank_based_selection(self, num_parents):
        ranked_population = sorted(
            self.population, key=lambda solution: solution.fitness)
        for i, solution in enumerate(ranked_population):
            solution.rank = i+1

        rank_sum = sum(range(1, len(self.population)+1))

        for solution in ranked_population:
            solution.probability = solution.rank / rank_sum

        self.parents = []
        for i in range(num_parents):
            r = random.random()
            cumulative_prob = 0
            for solution in ranked_population:
                cumulative_prob += solution.probability
                if cumulative_prob >= r:
                    self.parents.append(solution)
                    break

        return self.parents

    def cross_over_operator(self):
        # each two parent will return a child with its next parent
        self.children = []
        for i, parent in enumerate(self.parents):
            print(f"i:{i}")
            # this to break if its the last element in the lst
            if parent == self.parents[-1]:
                break
            next_parent = self.parents[i+1]
            print(
                f"--------------\nparent:{parent}\nnext_parent:{next_parent}\n------------------")
            # now i have parent and next_parent
            # create a probability factor to choose randomly parts from parent 1 and parts from parent 2 in the new parent

            p_factor = random.randint(
                1, min(len(parent.routes), len(next_parent.routes)))
            routes = parent.routes[:p_factor] + next_parent.routes[p_factor:]
            print(f"routes:{routes}")
            print("------------------------")

            new_routes = []
            for ix, route in enumerate(routes):
                if route == routes[-1]:
                    break

                next_route = routes[ix+1]

                p_factor = random.randint(
                    1, min(len(route.customers), len(next_route.customers)))

                new_customers = route.customers[:p_factor] + \
                    next_route.customers[p_factor:]

                new_route = self.create_route(new_customers)
                new_routes.append(new_route)
            if len(new_routes) == 0:
                continue
            child = Solution(new_routes)
            child.update_fitness()
            self.children.append(child)
        return self.children

    def insertion_mutation(self):
        for solution in self.children:
            for route in solution.routes:
                print(f"route customers before:{route.customer_nums}")
                # randomly select two positions in the solution
                pos1 = random.randint(0, len(route.customers) - 1)
                pos2 = random.randint(0, len(route.customers) - 1)
                # insert the gene at pos1 to pos2
                gene = route.customers.pop(pos1)
                route.customers.insert(pos2, gene)
                gene = route.customer_nums.pop(pos1)
                route.customer_nums.insert(pos2, gene)

                print(f"route customers after:{route.customer_nums}")
                route.update_load()
                route.update_distance()

            solution.update_distance()
            solution.update_fitness()

    def apply_genetic_algorithm(self, GENERATION_NUM, POPULATION_SIZE, PARENTS_SIZE):
        # create initial population
        self.generate_initial_population(POPULATION_SIZE)

        best_solutions = []
        for i in range(GENERATION_NUM):
            best_solution = self.best_solution
            best_solution_fitness = self.best_solution.fitness
            for j in range(POPULATION_SIZE):
                self.rank_based_selection(PARENTS_SIZE)
                self.cross_over_operator()
                self.insertion_mutation()
                print(f"children after finished {self.children}")
                for sol in self.children:
                    if sol.fitness > best_solution_fitness:
                        best_solution = sol
                        best_solution_fitness = sol.fitness

            self.generate_population(POPULATION_SIZE, best_solution.routes)
            best_solutions.append(best_solution)

        ranked_solutions = sorted(
            best_solutions, key=lambda solution: solution.fitness)
        print(f"ranked solution 0:{ranked_solutions[0].fitness}")
        print(f"ranked solution -1:{ranked_solutions[-1].fitness}")

        self.best_solution_ever = ranked_solutions[-1]
        return self.best_solution_ever


class Solution:
    def __init__(self, routes):
        self.routes = routes
        self.fitness = 0
        self.rank = 0
        self.probability = 0
        self.distances = []
        self.customers = []
        self.total_distance = 0
        for route in self.routes:
            self.customers.append(route.customers)
        self.total_distance = self.update_distance()
        self.update_distance()

        self.fitness = self.update_fitness()
        self.customers_served_percentage = self.get_percentage_of_customers_served()

    def update_distance(self):
        self.total_distance = 0
        for route in self.routes:
            route.update_load()
            route.update_distance()
            self.total_distance += route.distance

        return self.total_distance

    def update_fitness(self):
        self.update_distance()
        # it should return fitness_value
        # it should check capacity, time constrains for the whole solutions:
        for route in self.routes:
            if route.check_time_window() and route.check_capacity():
                print("route is feasible")
                pass
            else:
                print("route is not feasible")
                self.fitness = 0
                return self.fitness
        self.customers_served_percentage = self.get_percentage_of_customers_served()

        self.fitness = self.customers_served_percentage*1000-self.total_distance

        return self.fitness

    def formulate_solution(self):
        route_num = 1
        for r in self.routes:
            r.update_load()
            r.update_distance()
            route_str = f"Route {route_num} (Load {r.load}): "
            route_str += "1-"
            route_str += " ->".join([f"{c.cust_no}" for c in r.customers])
            route_str += "-1"
            distance = r.distance

            route_str += f" (Distance {round(distance,2)})"
            print(route_str)
            route_num += 1
        self.update_distance()
        # Print total distance and percentage of customers served
        print(f"Total distance: {round(self.total_distance,2)}")
        print(
            f"Percentage of customers served: {round(self.get_percentage_of_customers_served(),2)}%")

    def get_percentage_of_customers_served(self):

        customers_served = []
        for r in self.routes:
            if len(r.customers) == 1:

                pass
            else:
                for customer in r.customers:
                    customers_served.append(customer)
        print(
            f"len of customer served: {len(customers_served)}")
        return (len(customers_served)/100)*100


class Route:
    def __init__(self, id, DEPOT):
        """ this class to create routes, store data"""
        self.id = id
        self.current_time = 0
        self.customers = []
        self.customer_nums = []
        self.distance = 0
        self.CAPACITY = 200  # constant
        self.load = 0
        self.DEPOT = DEPOT

    def __eq__(self, other):
        if isinstance(other, Route):
            return (self.id == other.id)
        return False

    def add_customer(self, customer):
        self.customers.append(customer)
        self.customer_nums.append(customer.cust_no)
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
            service_start_time = max(current_time, customer.ready_time)

            if service_start_time < customer.ready_time:
                return False
            if service_start_time > customer.due_date:
                return False
            service_end_time = service_start_time + customer.service_time
            current_time = service_end_time

        return True

    def update_distance(self):
        # distance between customers + distance between boundary customers and DEPOT
        n = len(self.customers)
        distances = []
        for i, customer in enumerate(self.customers):

            if i+1 >= n:
                continue

            next_customer = self.customers[i+1]
            distance_between_cust_next_customer = get_distance_between_two_points(
                customer.xcoord, customer.ycoord, next_customer.xcoord, next_customer.ycoord)
            distances.append(distance_between_cust_next_customer)

        # distance between DEPOT and boundry customers
        b1 = self.customers[0]
        distance_between_b1_DEPOT = get_distance_between_two_points(
            b1.xcoord, b1.ycoord, self.DEPOT.xcoord, self.DEPOT.ycoord)
        distances.append(distance_between_b1_DEPOT)

        b2 = self.customers[-1]
        distance_between_b2_DEPOT = get_distance_between_two_points(
            b2.xcoord, b2.ycoord, self.DEPOT.xcoord, self.DEPOT.ycoord)
        distances.append(distance_between_b2_DEPOT)

        self.distance = 0
        for distance in distances:
            self.distance += distance

        return self.distance

    def check_capacity(self):
        self.update_load()
        if self.load >= self.CAPACITY:
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
        self.cust_no = int(cust_no)
        self.xcoord = xcoord
        self.ycoord = ycoord
        self.demand = demand
        self.ready_time = ready_time
        self.due_date = due_date
        self.service_time = service_time
