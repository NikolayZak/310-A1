
# 0. Adjust parameters
FEATURE_X_INDEX = 2    # Index of the feature for the x-axis (0 to 3 for Iris)
FEATURE_Y_INDEX = 3    # Index of the feature for the y-axis (0 to 3 for Iris).

NUM_CLUSTERS = 3       # Number of clusters for K-Means (Experiment with 2, 3, 4)
MAX_ITER = 10          # Maximum number of iterations for the algorithm (Experiment with 5, 10, 20)
learning_rate = 0.01    # lr: 0.1, 0.01
num_epochs = 10        # epoch: 5, 10, 50
hidden_layers = [128]  # layer: [128], [256, 256]

# 1. Import any other required libraries (e.g., numpy, scikit-learn)
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

np.random.seed(42)
tf.random.set_seed(42)
batch_size = 64

iris = load_iris()
X = iris.data
y = iris.target
num_classes = len(np.unique(y))
input_size = X.shape[1]

y_onehot = keras.utils.to_categorical(y, num_classes=num_classes)

x_train, x_test, y_train, y_test = train_test_split(
    X, y_onehot, test_size=0.1, random_state=42, shuffle=True
)

# 3. Implement K-Means Clustering
    # 3.1. Import KMeans from scikit-learn
from sklearn.cluster import KMeans

    # 3.2. Create an instance of KMeans with the specified number of clusters and max_iter
kmeans = KMeans(n_clusters=NUM_CLUSTERS, max_iter=MAX_ITER, random_state=42)

    # 3.3. Fit the KMeans model to the data X
kmeans.fit(X)

    # 3.4. Obtain the cluster labels
cluster_labels = kmeans.labels_



def create_mlp_model(input_size, hidden_layers, output_size):
    model = keras.Sequential()

    # Input layer is defined by specifying input_shape in the first layer
    for i, units in enumerate(hidden_layers):
        if i == 0:
            # First hidden layer with input shape specified
            model.add(layers.Dense(units, activation='relu', input_shape=(input_size,)))
        else:
            # Subsequent hidden layers
            model.add(layers.Dense(units, activation='relu'))

    # Output layer with softmax activation for multi-class classification
    model.add(layers.Dense(output_size, activation='softmax'))

    return model

model = create_mlp_model(input_size, hidden_layers, num_classes)

model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=learning_rate),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    x_train, y_train,
    epochs=num_epochs,
    batch_size=batch_size,
    validation_split=0.1,  # Use 10% of training data for validation
    verbose=1
)


test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')



plt.figure(figsize=(12, 5))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()

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