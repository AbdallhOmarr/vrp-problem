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


#getting distance from depot for all points 
def get_distance_between_two_points(x1,y1,x2,y2):
    x_diff = x2-x1
    y_diff = y2-y1
    dist = math.sqrt(math.pow(x_diff, 2) + math.pow(y_diff, 2))
    return dist

def add_distance_feature(customers_array, depot):
    # extract XCOORD and YCOORD columns from customers array
    XCOORD = customers_array[:, 1]
    YCOORD = customers_array[:, 2]
    
    # calculate distances from depot to customers
    distances = np.sqrt((XCOORD - depot[1])**2 + (YCOORD - depot[2])**2)
    
    # add distances as new column to customers array
    customers_array = np.column_stack((customers_array, distances))
    
    return customers_array


def cluster_array(customers_array,  n_clusters):
    
    # extract columns to cluster from customers array
    columns_to_cluster = [4, 1, 2, 7]  # indices of READY_TIME, XCOORD, YCOORD, and distance_FROM_DEPOT columns
    
    # normalize the data
    data_norm = (customers_array[:, columns_to_cluster] - customers_array[:, columns_to_cluster].mean(axis=0)) / customers_array[:, columns_to_cluster].std(axis=0)
    
    # cluster the data using KMeans
    kmeans = KMeans(n_clusters=n_clusters).fit(data_norm)
    
    # add cluster labels to customers array
    customers_array = np.column_stack((customers_array, kmeans.labels_))
    
    # sort points by distance from depot within each cluster
    sorted_indices = np.lexsort((customers_array[:, 7], customers_array[:, 5], customers_array[:, 8]))
    customers_array = customers_array[sorted_indices, :]
    
    return customers_array




        









def get_routes(customers_array,n_clusters,DEPOT):

    routes =[]
    for n in range(n_clusters):
        route_customers = customers_array[customers_array[:, 8] == n]
  
        if route_customers.shape[0]==0:
            print(f"n:{n}")
            print("-"*50)
            print(route_customers)
            print("-"*50)
            continue

        route = Route(route_customers,DEPOT)
        routes.append(route)

    return routes

def generate_initial_population(customers_array,POPULATION_SIZE,n_cluster,DEPOT):

    population = []
    for i in range(POPULATION_SIZE):
        clustered_customers_array = cluster_array(customers_array,n_cluster)
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
            return parent1
        else:
            return parent2
        
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
    n_generation = 300
    n_population = 100 
    co_rate = 0.8
    m_rate = 0.3
    n_clusters = 20
    pop = generate_initial_population(customers_array, n_population,n_clusters,DEPOT)
    pop_fitness = [sol.fitness_func() for sol in pop]

    best_idx = pop_fitness.index(max(pop_fitness))
    best_sol_ever = pop[best_idx]

    for g in range(n_generation):
        print(f"Genration:{g}")
        pop = generate_initial_population(customers_array, n_population,n_clusters,DEPOT)
        print("Population created")
        parent1,parent2 = rank_based_selection(pop,2)
        print("Two parents created")
        offspring = create_offspring(parent1,parent2,co_rate)
        print("an offspring created")
        # Mutate offspring
        for route in offspring.routes:
            route.mutate(m_rate)
        print(f"Route mutated")


        # Calculate fitness for offspring
        offspring_fitness = offspring.fitness_func()
        print(f"offspring fitness:{offspring_fitness}")
        print("offspring solution:")
        print(len(offspring.routes),offspring.total_distance,offspring.get_total_customers_served(),offspring.check_feasiblity(),offspring.fitness_func())
        print("-"*50)

        if offspring.get_total_customers_served()>100:
            break

        # Replace worst solution in population with offspring
        pop_fitness = [sol.fitness_func() for sol in pop]

        worst_idx = pop_fitness.index(min(pop_fitness))
        print(f"worest solution fitness:{pop_fitness[worst_idx]}")
    

        if offspring_fitness > pop_fitness[worst_idx]:
            pop[worst_idx] = offspring

    
        best_idx = pop_fitness.index(max(pop_fitness))
        best_sol = pop[best_idx]

        if best_sol.fitness_func() > best_sol_ever.fitness_func():
            best_sol_ever = best_sol
        print("best solution till now:")
        print(len(best_sol_ever.routes),best_sol_ever.get_total_solution_distance(),best_sol_ever.get_total_customers_served(),best_sol_ever.check_feasiblity(),best_sol_ever.fitness_func())
        print("-"*50)

