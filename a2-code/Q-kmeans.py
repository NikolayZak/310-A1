
# 0. Adjust parameters
NUM_CLUSTERS = 2       # Number of clusters for K-Means (Experiment with 2, 3, 4)
MAX_ITER = 5           # Maximum number of iterations for the algorithm (Experiment with 5, 10, 20)
FEATURE_X_INDEX = 2    # Index of the feature for the x-axis (0 to 3 for Iris)
FEATURE_Y_INDEX = 3    # Index of the feature for the y-axis (0 to 3 for Iris)

# 1. Import any other required libraries (e.g., numpy, scikit-learn)

# 2. Load the Iris dataset using scikit-learn's load_iris() function

# 3. Implement K-Means Clustering
    # 3.1. Import KMeans from scikit-learn
    # 3.2. Create an instance of KMeans with the specified number of clusters and max_iter
    # 3.3. Fit the KMeans model to the data X
    # 3.4. Obtain the cluster labels

# 4. Visualize the Results
    # 4.1. Extract the features for visualization
    # 4.2. Create a scatter plot of x_feature vs y_feature, colored by the cluster labels
    # 4.3. Use different colors to represent different clusters

