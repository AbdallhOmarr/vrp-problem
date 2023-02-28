import pandas as pd
import re
import matplotlib.pyplot as plt
import math
import numpy as np

def get_data_points(txt_file):
    ## Function to convert txt file into a pandas dataframe
    data = []
    with open(txt_file, "r") as file:
        lines = file.readlines()  # Read all lines in the file
        column_parsed = False
        column_names = []
        if column_parsed == False:
            words =  re.findall("\S+", lines[0])  # Extract column names from first line
            for word in words:
                column_names.append(word)   
            column_parsed=True                 
                
        for line in lines[1:]:  # Loop through lines, skipping the first line
            dataline = re.findall("\S+", line)  # Extract data points from line
            line_dict={}
            for i,point in enumerate(dataline):
                line_dict[column_names[i]]= float(point)  # Convert data point to float and store in dictionary
            data.append(line_dict)  # Append dictionary to list of data
    
    df = pd.DataFrame(data)   # Create pandas dataframe from list of dictionaries
    return df  # Return the dataframe




# i tried to use nearest neighbour instead of clark wright saving to see if it will improve the initial solution
def generate_initial_population(data, pop_size):
    """
    Generates an initial population of pop_size solutions using the nearest neighbor heuristic and respecting time windows.
    
    Args:
        data (pandas.DataFrame): A DataFrame containing the customer data, including demand, coordinates, time windows,
                                 and service times.
        pop_size (int): The desired population size.
    
    Returns:
        list: A list of pop_size solutions, where each solution is a list of route sequences that represent a valid solution.
    """
    def distance(x1, y1, x2, y2):
        """
        Calculates the Euclidean distance between two points.
        """
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    speed = 50  # km/h

    # Sort customers by their x-coordinate
    sorted_data = data.sort_values('XCOORD')

    # Initialize the population
    population = []

    # Generate a solution for each vehicle in the fleet
    for _ in range(pop_size):
        # Initialize a new route sequence
        route = []
        remaining_cap = 200

        # Choose a random customer to start the route
        cust_no = np.random.choice(sorted_data['CUST_NO'])
        route.append(cust_no)

        # Repeat until all customers have been added to a route
        while len(route) < len(sorted_data):
            # Find the nearest unvisited customer
            curr_x, curr_y = sorted_data.loc[sorted_data['CUST_NO'] == cust_no, ['XCOORD', 'YCOORD']].values[0]
            unvisited_data = sorted_data[~sorted_data['CUST_NO'].isin(route)]
            distances = [distance(curr_x, curr_y, row['XCOORD'], row['YCOORD']) for _, row in unvisited_data.iterrows()]
            nearest_idx = np.argmin(distances)
            nearest_cust_no = unvisited_data.iloc[nearest_idx]['CUST_NO']

            # Check if the nearest customer can be added to the route while respecting time windows
            nearest_x, nearest_y = unvisited_data.iloc[nearest_idx][['XCOORD', 'YCOORD']]
            nearest_due_date = unvisited_data.iloc[nearest_idx]['DUE_DATE']
            nearest_service_time = unvisited_data.iloc[nearest_idx]['SERVICE_TIME']
            dist = distances[nearest_idx]
            prev_cust_no = route[-1]
            prev_due_date = sorted_data.loc[sorted_data['CUST_NO'] == prev_cust_no, 'DUE_DATE'].values[0]
            prev_service_time = sorted_data.loc[sorted_data['CUST_NO'] == prev_cust_no, 'SERVICE_TIME'].values[0]
            if remaining_cap - unvisited_data.iloc[nearest_idx]['DEMAND'] >= 0 and \
                    prev_due_date + prev_service_time + dist/speed + nearest_service_time + dist/speed <= nearest_due_date:
                # Add the nearest customer to the route
                route.append(nearest_cust_no)
                remaining_cap -= unvisited_data.iloc[nearest_idx]['DEMAND']
                cust_no = nearest_cust_no
            else:
                # If the nearest customer cannot be added, start a new route
                population.append(route)
                route = []
                remaining_cap = 200
                cust_no = np.random.choice(sorted_data['CUST_NO'])
                route.append(cust_no)

        # Add the final route to the population
        population.append(route)

    return population



data = get_data_points("vrp data.txt")

pop = generate_initial_population(data,1)

print(pop)

