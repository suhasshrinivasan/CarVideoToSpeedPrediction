import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


def init_model(c):
    """
    Given a model configuration c, returns an initialized model of proper type and hyperparameters
    """
    # TODO: Add ridge, lasso, en
    if c['model_type'] == 'ridge':
        return Ridge(alpha=c['alpha'])
    elif c['model_type'] == 'lasso':
        return Lasso(alpha=c['alpha'])
    elif c['model_type'] == 'en':
        return ElasticNet(alpha=c['alpha'], l1_ratio=c['l1_ratio'])
    elif c['model_type'] == 'svr':
        return SVR(C=c['C'])
    elif c['model_type'] == 'gbr':
        return GradientBoostingRegressor(n_estimators=c['n_estimators'], max_depth=c['max_depth'])
    elif c['model_type'] == 'rfr':
        return RandomForestRegressor(n_estimators=c['n_estimators'], max_depth=c['max_depth'])
    elif c['model_type'] == 'dtr':
        return DecisionTreeRegressor(max_features=c['max_features'], max_depth=c['max_depth'])
    elif c['model_type'] == 'knr':
        return KNeighborsRegressor(weights=c['weights'], n_neighbors=c['n_neighbors'])
    else:
        raise Exception('Improper Model Config ' + str(config))
        return


def init_configs(model_type):
    """
    Returns a list of model configurations (type and hyperparameters).
    Facilitates model selection process.
    """
    model_type = str.lower(model_type)
    configs = []
    if model_type == 'ridge':  # Ridge Regression (L2 Penalty)
        for alpha in np.logspace(-6, 6, 30):
            configs.append({'model_type': model_type, 'alpha': alpha})
    elif model_type == 'lasso':  # Lasso Regression (L1 Penalty)
        for alpha in np.logspace(-6, 6, 30):
            configs.append({'model_type': model_type, 'alpha': alpha})
    elif model_type == 'en':  # Elastic Net (L1 and L2 Penalty)
        for alpha in np.logspace(-4, 2, 10):
            for l1_ratio in [0.0, 0.01, 0.02, 0.05, 0.1, 0.5, 0.9, 0.95, 0.98, 0.99, 1.0]:
                configs.append({'model_type': model_type, 'alpha': alpha, 'l1_ratio': l1_ratio})
    elif model_type == 'svr':  # Support Vector Regressor
        for C in np.logspace(-5, 4, 10):
            configs.append({'model_type': model_type, 'C': C})
    elif model_type == 'gbr':  # Gradient Boosting Regressor
        for n_estimators in [128, 512, 2048]:
            for max_depth in [1, 3, 6, 10]:
                configs.append({'model_type': model_type, 'n_estimators': n_estimators,
                    'max_depth': max_depth})
    elif model_type == 'rfr':  # Random Forest Regressor
        for n_estimators in [128, 512]:
            for max_features in [0.2, 0.5, 1.0]:
                for max_depth in [1, 3, 6]:
                    configs.append({'model_type': model_type, 'n_estimators': n_estimators,
                        'max_features': max_features, 'max_depth': max_depth})
    elif model_type == 'dtr':  # Decision Tree Regressor
        for max_features in [0.2, 0.4, 0.6, 0.8, 1.0]:
            for max_depth in [1, 2, 3, 4, 5]:
                configs.append({'model_type': model_type, 'max_features': max_features,
                    'max_depth': max_depth})
    elif model_type == 'knr':  # K Nearest Neighbors Regressor
        for weights in ['uniform', 'distance']:
            for n_neighbors in [1, 2, 3, 4, 8, 16, 24, 32, 40, 48, 56, 64]:
                configs.append({'model_type': model_type, 'weights': weights,
                    'n_neighbors': n_neighbors})
    return configs


def test_config(config, X_train, y_train, X_test, y_test):
    """
    Tests a model configuration on given sets of training and testing data.
    Uses Mean Squared Error as accuracy metric.
    Returns the MSE of using the model on the training data and on the validation data.
    """
    model = init_model(config)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    mse_train = mean_squared_error(y_pred_train, y_train)

    y_pred_test = model.predict(X_test)
    mse_test = mean_squared_error(y_test, y_pred_test)
    return np.array([mse_train, mse_test])


def hp_sweep(model_type, X_train, y_train, X_val, y_val):
    """
    Sweeps over the hyperparameter space specified in init_configs for a given model type,
    Training models on the given training data and validating on the given validation data
    Check init_configs for valid model_type strings.
    Returns the best configuration and its training accuracy and validation accuracy.
    """
    # Initialize Configs and Variables
    configs = init_configs(model_type)
    best_config = {}
    best_train_val_error = np.array([np.inf, np.inf])

    # Sweep Hyperparameter Space
    for config in configs:
        train_val_error = test_config(config, X_train, y_train, X_val, y_val)
        print train_val_error.round(3), config
        if train_val_error[1] < best_train_val_error[1]:
            best_train_val_error = train_val_error
            best_config = config.copy()

    return best_config, best_train_val_error
