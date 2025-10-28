import numpy as np
import q3 as q3

#-------------------------------------------------------------------------------
# Auto Data
#-------------------------------------------------------------------------------

# load auto-mpg-regression.tsv, including  Keys are the column names, including mpg.
auto_data_all = q3.load_auto_data('auto-mpg-regression.tsv')

# The choice of feature processing for each feature, mpg is always raw and
# does not need to be specified.  Other choices are q3.standard and q3.one_hot.

features1 = [('cylinders', q3.standard),
            ('displacement', q3.standard),
            ('horsepower', q3.standard),
            ('weight', q3.standard),
            ('acceleration', q3.standard),
            ('origin', q3.one_hot)]

features2 = [('cylinders', q3.one_hot),
            ('displacement', q3.standard),
            ('horsepower', q3.standard),
            ('weight', q3.standard),
            ('acceleration', q3.standard),
            ('origin', q3.one_hot)]

# Construct the standard data and label arrays
#auto_data[0] has the features for choice features1
#auto_data[1] has the features for choice features2
#The labels for both are the same, and are in auto_values
auto_data = [0, 0]
auto_values = 0
auto_data[0], auto_values = q3.auto_data_and_values(auto_data_all, features1)
auto_data[1], _ = q3.auto_data_and_values(auto_data_all, features2)

#standardize the y-values
auto_values, mu, sigma = q3.std_y(auto_values)

#-------------------------------------------------------------------------------
# Analyze auto data
#-------------------------------------------------------------------------------     

#Your code for cross-validation goes here
import numpy as np
import q3 as q3

lambda_grid_order_1_2 = np.arange(0.0, 0.11, 0.01)
lambda_grid_order_3 = np.arange(0, 201, 20)
k = 10  # folds

best_rmse = float('inf')
best_config = None

_, n_samples = auto_data[0].shape

# Shuffle indices for cross-validation splits
indices = np.arange(n_samples)
np.random.seed(0)
np.random.shuffle(indices)
fold_sizes = np.full(k, n_samples // k)
fold_sizes[:n_samples % k] += 1
current = 0
fold_indices = []
for fold_size in fold_sizes:
    fold_indices.append(indices[current:current + fold_size])
    current += fold_size

for feature_set_index in [0, 1]:
    X_orig = auto_data[feature_set_index]
    y = auto_values

    for order in [1, 2, 3]:
        poly_fun = q3.make_polynomial_feature_fun(order)
        X_poly = poly_fun(X_orig)

        if order in [1, 2]:
            lambda_grid = lambda_grid_order_1_2
        else:
            lambda_grid = lambda_grid_order_3

        for lam in lambda_grid:
            rmses = []
            for i in range(k):
                # Prepare train and test splits based on fold indices
                test_idx = fold_indices[i]
                train_idx = np.hstack([fold_indices[j] for j in range(k) if j != i])

                X_train, y_train = X_poly[:, train_idx], y[:, train_idx]
                X_test, y_test = X_poly[:, test_idx], y[:, test_idx]

                # Train ridge regression model using ridge_min
                th, th0 = q3.ridge_min(X_train, y_train, lam)

                # Predict on test set
                y_pred = q3.lin_reg(X_test, th, th0)

                # Calculate RMSE for this fold
                rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
                rmses.append(rmse)

            avg_rmse = np.mean(rmses)
            if avg_rmse < best_rmse:
                best_rmse = avg_rmse
                best_config = {
                    'feature_set': feature_set_index,
                    'order': order,
                    'lambda': lam,
                    'rmse': avg_rmse
                }

# Report best hyperparameters and RMSE
print("Best config:")
print(f"Feature set: {best_config['feature_set']}")
print(f"Polynomial order: {best_config['order']}")
print(f"Lambda: {best_config['lambda']:.3f}")
print(f"Average RMSE (std units): {best_config['rmse']:.4f}")
