import pandas as pd
import re
import matplotlib.pyplot as plt
import math
import numpy as np
import random
import pandas as pd
from sklearn.cluster import KMeans
import re
from Route import *
from Solution import * 
from sklearn.cluster import KMeans
import copy
from scipy.cluster.hierarchy import linkage, fcluster
import numpy as np
# def load_data():
#     # lets load data
#     txt_file = "vrp data.txt"
#     data = []
#     with open(txt_file, "r") as file:
#         lines = file.readlines()  # Read all lines in the file
#         column_parsed = False
#         column_names = []
#         if column_parsed == False:
#             # Extract column names from first line
#             words = re.findall("\S+", lines[0])
#             for word in words:
#                 column_names.append(word)
#             column_parsed = True

#         for line in lines[1:]:  # Loop through lines, skipping the first line
#             # Extract data points from line
#             dataline = re.findall("\S+", line)
#             line_dict = {}
#             for i, point in enumerate(dataline):
#                 # Convert data point to float and store in dictionary
#                 line_dict[column_names[i]] = float(point)
#             data.append(line_dict)  # Append dictionary to list of data
       


#     #converting list of dict to numpy array
#     data_array = np.array([[d[col] for col in column_names] for d in data])

#     #extract first row into depot array
#     DEPOT = data_array[0, :]

#     # extract rest of the rows into customers_array
#     customers_array = data_array[1:, :]
#     customers_array = add_distance_feature(customers_array,DEPOT)

#     return customers_array,DEPOT


# #getting distance from depot for all points 
# def get_distance_between_two_points(x1,y1,x2,y2):
#     x_diff = x2-x1
#     y_diff = y2-y1
#     dist = math.sqrt(math.pow(x_diff, 2) + math.pow(y_diff, 2))
#     return dist

# def add_distance_feature(customers_array, depot):
#     # extract XCOORD and YCOORD columns from customers array
#     XCOORD = customers_array[:, 1]
#     YCOORD = customers_array[:, 2]
    
#     # calculate distances from depot to customers
#     distances = np.sqrt((XCOORD - depot[1])**2 + (YCOORD - depot[2])**2)
    
#     # add distances as new column to customers array
#     customers_array = np.column_stack((customers_array, distances))
    
#     return customers_array

# def cluster_array(customers_array, n_clusters):
   
#     # # extract columns to cluster from customers array
#     # columns_to_cluster = [1,2,4]  # indices of READY_TIME, DUE_DATE, XCOORD, YCOORD, and distance_FROM_DEPOT columns
    
#     # # normalize the data
#     # data_norm = (customers_array[:, columns_to_cluster] - customers_array[:, columns_to_cluster].mean(axis=0)) / customers_array[:, columns_to_cluster].std(axis=0)
    
#     # # cluster the data using KMeans
#     # kmeans = KMeans(n_clusters=n_clusters).fit(data_norm)
    
#     # # add cluster labels to customers array
#     # customers_array = np.column_stack((customers_array, kmeans.labels_))
    
#     # # sort points by cluster and ready time
#     # sorted_indices = np.lexsort((customers_array[:, 5], customers_array[:, 8]))
#     # customers_array = customers_array[sorted_indices, :]
    
#     # # # sort points by distance from depot within each cluster
#     # # for i in range(n_clusters):
#     # #     cluster_indices = np.where(customers_array[:, -1] == i)[0]
#     # #     sorted_indices = np.argsort(customers_array[cluster_indices, 7])
#     # #     customers_array[cluster_indices] = customers_array[cluster_indices][sorted_indices]
        
        
#     # Extract columns to cluster from customers array
#     columns_to_cluster = [1, 2, 7]  # indices of XCOORD, YCOORD, and distance_FROM_DEPOT columns

#     # Normalize the data
#     data_norm = (customers_array[:, columns_to_cluster] - customers_array[:, columns_to_cluster].mean(axis=0)) / customers_array[:, columns_to_cluster].std(axis=0)

#     # Use HAC to cluster the data
#     Z = linkage(data_norm, method='ward')

#     # Determine cluster labels for each data point
#     max_distance = 2.5  # maximum distance to consider when clustering
#     clusters = fcluster(Z, t=max_distance, criterion='distance')

#     # Add cluster labels to customers array
#     customers_array = np.column_stack((customers_array, clusters))

#     # Sort points by cluster and ready time
#     sorted_indices = np.lexsort((customers_array[:, 4], customers_array[:, 8]))
#     customers_array = customers_array[sorted_indices, :]

#     return customers_array


import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
import math
import re

def load_data():
    # lets load data
    txt_file = "vrp data.txt"
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
       


    #converting list of dict to numpy array
    data_array = np.array([[d[col] for col in column_names] for d in data])

    #extract first row into depot array
    DEPOT = data_array[0, :]

    # extract rest of the rows into customers_array
    customers_array = data_array[1:, :]
    customers_array = add_distance_feature(customers_array,DEPOT)

    return customers_array,DEPOT


def get_distance_between_two_points(x1, y1, x2, y2):
    x_diff = x2 - x1
    y_diff = y2 - y1
    dist = math.sqrt(math.pow(x_diff, 2) + math.pow(y_diff, 2))
    return dist


def add_distance_feature(customers_array, depot):
    # Extract XCOORD and YCOORD columns from customers array
    XCOORD = customers_array[:, 1]
    YCOORD = customers_array[:, 2]
    
    # Calculate distances from depot to customers
    distances = np.sqrt((XCOORD - depot[1])**2 + (YCOORD - depot[2])**2)
    
    # Add distances as new column to customers array
    customers_array = np.column_stack((customers_array, distances))
    
    return customers_array


def add_time_window_feature(customers_array):
    # Calculate the amount of time available to service each customer
    time_window = customers_array[:, 5] - customers_array[:, 4]
    customers_array = np.column_stack((customers_array, time_window))

    return customers_array

def cluster_array(customers_array, n_clusters):
    
    # extract columns to cluster from customers array
    columns_to_cluster = [1,2,4]  # indices of READY_TIME, DUE_DATE, XCOORD, YCOORD, and distance_FROM_DEPOT columns
    
    # normalize the data
    data_norm = (customers_array[:, columns_to_cluster] - customers_array[:, columns_to_cluster].mean(axis=0)) / customers_array[:, columns_to_cluster].std(axis=0)
    
    # cluster the data using KMeans
    kmeans = KMeans(n_clusters=n_clusters).fit(data_norm)
    
    # add cluster labels to customers array
    customers_array = np.column_stack((customers_array, kmeans.labels_))
    
    # sort points by cluster and ready time
    sorted_indices = np.lexsort((customers_array[:, 5], customers_array[:, 8]))
    customers_array = customers_array[sorted_indices, :]
    
    # # sort points by distance from depot within each cluster
    # for i in range(n_clusters):
    #     cluster_indices = np.where(customers_array[:, -1] == i)[0]
    #     sorted_indices = np.argsort(customers_array[cluster_indices, 7])
    #     customers_array[cluster_indices] = customers_array[cluster_indices][sorted_indices]
    
    return customers_array,n_clusters



# def cluster_array(customers_array, n_clusters):
#     customers_array = add_time_window_feature(customers_array)
#     # Extract columns to cluster from customers array
#     columns_to_cluster = [1, 2, 4, 5, 6, 7]  # indices of XCOORD, YCOORD, READY_TIME, DUE_DATE, distance_FROM_DEPOT, and time window columns

#     # Normalize the data
#     eps = 1e-6
#     data_norm = (customers_array[:, columns_to_cluster] - customers_array[:, columns_to_cluster].mean(axis=0)) / (customers_array[:, columns_to_cluster].std(axis=0) + eps)
#     # Check for NaN values
#     # print(np.isnan(data_norm).any())

#     # Check for infinite values
#     # print(np.isfinite(data_norm).all())

#     # Use HAC to cluster the data
#     Z = linkage(data_norm, method='ward')

#     # Determine cluster labels for each data point
#     max_distance = 3  # maximum distance to consider when clustering
#     clusters = fcluster(Z, t=max_distance, criterion='distance')
#     n_clusters = len(np.unique(clusters))
    
#     # Add cluster labels to customers array
#     customers_array = np.column_stack((customers_array, clusters))

#     # Sort points by cluster, ready time, and due date
#     sorted_indices = np.lexsort((customers_array[:, 9], customers_array[:, 4], customers_array[:, 5]))
#     customers_array = customers_array[sorted_indices, :]

#     # Sort points by distance from depot within each cluster
#     for i in range(n_clusters):
#         cluster_indices = np.where(customers_array[:, -1] == i)[0]
#         sorted_indices = np.argsort(customers_array[cluster_indices, 8])
#         customers_array[cluster_indices] = customers_array[cluster_indices][sorted_indices]

#     return customers_array,n_clusters









def get_routes(customers_array,n_clusters,DEPOT):
    routes =[]
    for n in range(1,n_clusters):
        route_customers = customers_array[customers_array[:, 8] == n]
        if route_customers.shape[0]==0:
            continue

        route = Route(route_customers,DEPOT)
        routes.append(route)

    return routes

def generate_initial_population(customers_array,POPULATION_SIZE,n_cluster,DEPOT):

    population = []
    for i in range(POPULATION_SIZE):
        clustered_customers_array,n_cluster = cluster_array(customers_array,n_cluster)
        routes = get_routes(clustered_customers_array,n_cluster,DEPOT)

        #error area
        sol = Solution(routes)
        # sol.prevent_duplicated_customers()
        sol.update_variables()
        population.append(sol)

    return population


def rank_based_selection(population, k=2):
    fitness_list = [sol.fitness_func() for sol in population]
    ranked_list = sorted(range(len(fitness_list)), key=lambda k: fitness_list[k])
    ranks = [len(fitness_list) - i for i in ranked_list]
    probs = [rank / sum(ranks) for rank in ranks]
    selected_indices = random.choices(range(len(population)), weights=probs, k=k)
    return [population[i] for i in selected_indices]

def create_offspring(parent1,parent2,crossover_rate):
    #getting routes of the first parent and second parent
    #create crossoverpoint
    #exchange routes between the two routes
    #create new sol 
    #run prevent duplication method
    if random.random() > crossover_rate:
        # if not, return a clone of parent1 or parent2 at random
        if random.random() < 0.5:
            return copy.deepcopy(parent1)
        else:
            return copy.deepcopy(parent2)
    
    p1_routes = parent1.routes
    p2_routes = parent2.routes
    crossover_point = random.randint(1,len(p1_routes)-1)

    offspring_routes = p1_routes[:crossover_point] + p2_routes[crossover_point:]
    offspring_sol = Solution(offspring_routes)
    # offspring_sol.prevent_duplicated_customers()
    offspring_sol.update_variables()
    return offspring_sol


def main():
    customers_array,DEPOT = load_data()
    
    #customers_df = cluster_df(customers_df,20)
    n_generation = 50
    n_population = 100 
    co_rate = 0.5
    m_rate = 0.5
    n_clusters = 20
    pop = generate_initial_population(customers_array, n_population,n_clusters,DEPOT)
    pop_fitness = [sol.fitness_func() for sol in pop]

    best_idx = pop_fitness.index(max(pop_fitness))
    best_sol_ever = pop[best_idx]
    best_sol_ever_fitness = best_sol_ever.fitness_func()
    for g in range(n_generation):
        print(f"Genration:{g}")
        # pop = generate_initial_population(customers_array, n_population,n_clusters,DEPOT)
        print("Population created")
        parent1,parent2 = rank_based_selection(pop,2)
        parent1.prevent_duplicated_customers()
        parent2.prevent_duplicated_customers()
        print("Two parents created")
        offspring = create_offspring(parent1,parent2,co_rate)
        offspring.prevent_duplicated_customers()
        print("an offspring created")
        # Mutate offspring
        for route in offspring.routes:
            route.mutate(m_rate)
        print(f"Route mutated")

        offspring.prevent_duplicated_customers()
        offspring.update_variables()
        # Calculate fitness for offspring
        offspring_fitness = offspring.fitness_func()
        print("offspring solution:")
        print(len(offspring.routes),offspring.total_distance,offspring.get_total_customers_served(),offspring.check_feasibility(),offspring.fitness_func())
        print("-"*50)

        if offspring.get_total_customers_served()>100:
            break

        # Replace worst solution in population with offspring
        pop_fitness = [sol.fitness_func() for sol in pop]

        worst_idx = pop_fitness.index(min(pop_fitness))
        print(f"offspring fitness:{offspring_fitness} > worest solution fitness:{pop_fitness[worst_idx]}?")

        if offspring_fitness > pop_fitness[worst_idx]:
            print("offspring fitness > worst sol fitness")
            pop[worst_idx] = offspring
            pop_fitness[worst_idx] = offspring_fitness

        
        best_idx = pop_fitness.index(max(pop_fitness))
        best_sol = pop[best_idx]
        best_sol_fitness = pop_fitness[best_idx]

        if best_sol_fitness> best_sol_ever_fitness:
            best_sol_ever = best_sol
            best_sol_ever_fitness = best_sol_fitness
            
        print("best solution till now:")
        print(len(best_sol_ever.routes),best_sol_ever.get_total_solution_distance(),best_sol_ever.get_total_customers_served(),best_sol_ever.check_feasibility(),best_sol_ever.fitness_func())
        print("-"*50)


