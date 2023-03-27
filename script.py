import pandas as pd
from sklearn.cluster import KMeans
import re
# Load the data into a Pandas DataFrame


def read_customer_data(txt_file):
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
    customers_df = pd.DataFrame(data)
    return customers_df


data = read_customer_data("vrp data.txt")


# Select the columns to use for clustering
columns_to_cluster = ['XCOORD', 'YCOORD', 'DEMAND',
                      'READY_TIME', 'DUE_DATE', 'SERVICE_TIME']

# Normalize the data
data_norm = (data[columns_to_cluster] -
             data[columns_to_cluster].mean()) / data[columns_to_cluster].std()

# Apply k-means clustering with k=2
kmeans = KMeans(n_clusters=20, random_state=0).fit(data_norm)

# Add the cluster labels to the DataFrame
data['cluster'] = kmeans.labels_

# Print the resulting clusters
print(data[['CUST_NO', 'cluster']])

data.to_excel("Clustered data.xlsx")
