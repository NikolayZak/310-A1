
# 0. Adjust parameters
NUM_CLUSTERS = 2       # Number of clusters for K-Means (Experiment with 2, 3, 4)
MAX_ITER = 5           # Maximum number of iterations for the algorithm (Experiment with 5, 10, 20)
FEATURE_X_INDEX = 2    # Index of the feature for the x-axis (0 to 3 for Iris)
FEATURE_Y_INDEX = 3    # Index of the feature for the y-axis (0 to 3 for Iris)

# 1. Import any other required libraries (e.g., numpy, scikit-learn)
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# 2. Load the Iris dataset using scikit-learn's load_iris() function
iris = load_iris()
X = iris.data
y = iris.target

# 3. Implement K-Means Clustering
    # 3.1. Import KMeans from scikit-learn
from sklearn.cluster import KMeans

    # 3.2. Create an instance of KMeans with the specified number of clusters and max_iter
kmeans = KMeans(n_clusters=NUM_CLUSTERS, max_iter=MAX_ITER, random_state=42)

    # 3.3. Fit the KMeans model to the data X
kmeans.fit(X)

    # 3.4. Obtain the cluster labels
cluster_labels = kmeans.labels_

# 4. Visualize the Results
    # 4.1. Extract the features for visualization
x_feature = X[:, FEATURE_X_INDEX]
y_feature = X[:, FEATURE_Y_INDEX]

    # 4.2. Create a scatter plot of x_feature vs y_feature, colored by the cluster labels
plt.figure(figsize=(7, 7))
plt.scatter(
    x = x_feature,
    y = y_feature,
    c = cluster_labels
)
    # 4.3. Use different colors to represent different clusters
plt.scatter(
    kmeans.cluster_centers_[:, FEATURE_X_INDEX],
    kmeans.cluster_centers_[:, FEATURE_Y_INDEX],
    color='black',
    marker='X',
    s=200,
    label='Centroids'
)

plt.xlabel(iris.feature_names[FEATURE_X_INDEX])
plt.ylabel(iris.feature_names[FEATURE_Y_INDEX])
plt.title(f'K-Means Clustering')
plt.show()