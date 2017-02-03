import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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
        # for alpha in [1e2, 2e2, 4e2, 1e3, 2e3, 4e3, 1e4, 2e4, 4e4, 1e5, 2e5]:
        for alpha in [1, 3, 10, 30, 100, 300]:
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
        for n_estimators in [128]:
            for max_depth in [1, 3, 6]:
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


def test_config(config, X, y, k_fold, scale_data, pca_n_components):
    """
    Cross-Validates model configuration on given data, returning the average MSE.
    """
    # Cross Validate
    mse_train = 0.0
    mse_test = 0.0
    kf = KFold(n_splits=k_fold)
    for train, test in kf.split(X):
        # Set Data
        X_train = X[train].copy()
        X_test = X[test].copy()
        y_train = y[train].copy()
        y_test = y[test].copy()

        # Scale Data
        if scale_data:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # PCA on Data
        if pca_n_components != 0 or pca_n_components == X.shape[1]:
            pca = PCA(n_components=pca_n_components)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)

        # Train Model
        model = init_model(config)
        model.fit(X_train, y_train)

        # Test Model
        y_pred_train = model.predict(X_train)
        mse_train += mean_squared_error(y_pred_train, y_train)
        y_pred_test = model.predict(X_test)
        mse_test += mean_squared_error(y_test, y_pred_test)

    mse_train /= k_fold
    mse_test /= k_fold
    return np.array([mse_train, mse_test])


def hp_sweep(model_type, X, y, k_fold, scale_data, pca_n_components_list):
    """
    Sweeps over the hyperparameter space specified in init_configs for a given model type,
    Cross-Validating each model type on the data.
    Check init_configs for valid model_type strings.
    Returns the best configuration and its training accuracy and validation accuracy.
    """
    # Initialize Configs and Variables
    configs = init_configs(model_type)
    best_config = {}
    best_cv_error = np.array([np.inf, np.inf])

    # Sweep Hyperparameter Space
    for config in configs:
        for pca_n_components in pca_n_components_list:
            cv_error = test_config(
                config, X, y, k_fold,
                scale_data, pca_n_components
            )
            print cv_error.round(3), config, pca_n_components
            if cv_error[1] < best_cv_error[1]:
                best_cv_error = cv_error
                best_config = config.copy()

    return best_config, best_cv_error
