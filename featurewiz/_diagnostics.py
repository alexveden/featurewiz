import operator
import pandas as pd
import numpy as np
import tqdm
from sklearn import model_selection
import matplotlib.pyplot as plt


def plot_learning_curve(estimator, X, y, scoring, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5), title='Learning curve'):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    scoring: scoring : string, callable or None, optional, default: None

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = model_selection.learning_curve(estimator, X, y, cv=cv,
                                                                            n_jobs=n_jobs,
                                                                            train_sizes=train_sizes,
                                                                            scoring=scoring)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def plot_feature_learning_curve(estimator, X: pd.DataFrame, y: pd.Series, scoring, scoring_kwargs={}, base_estimator=None):
    """
    ML model features importance diagnostic plot.
    The algorithm outline:
    1. Calculate the estimator model on entire dataset
    2. Get features importances via .feature_importance_ or .coef_ (sort them)
    3. Calculate and plot 2 ML models versions (Red model - includes only worst to best feature, Green model - includes only best to worst features)
    4. Plot alternative naive and Base model (blue) for comparison

    :param estimator: ML model (must implement .feature_importance_ or .coef_ or to be XGBoost model)
    :param X: X features
    :param y: y-target
    :param scoring: scoring function f(y_true, y_predicted), see sklearn.metrics help
    :param scoring_kwargs: additional kwargs for scoring function (like sample_weights=), see sklearn.metrics help
    :param base_estimator: None or model class with .fit() / .predict()

    :return: pd.DataFrame for each model scores
    """
    #
    # Fit estimator first to get estimator.feature_importances_
    #
    estimator.fit(X, y)
    try:
        if hasattr(estimator, 'booster'):
            # XGBOOST!
            importance = estimator.get_booster().get_fscore()
            importance = sorted(importance.items(), key=operator.itemgetter(1))

            df = pd.DataFrame(importance, columns=['feature', 'fscore']).set_index('feature')
            feat_imp = (df['fscore'] / df['fscore'].sum()).sort_values()
        else:
            # * Forests
            feat_imp = pd.Series(estimator.feature_importances_, index=X.columns).sort_values()
    except AttributeError:
        # Other linear models
        feat_imp = pd.Series(estimator.coef_, index=X.columns).sort_values()

    # Sort dataset columns by importance [lowest -> highest]
    sorted_X = X[feat_imp.index]

    rfe_result = []
    for i in tqdm.tqdm_notebook(range(len(sorted_X.columns))):
        X_best = sorted_X[sorted_X.columns[i:]]
        X_worst = sorted_X[sorted_X.columns[:i + 1]]

        _X_train_b, _X_test_b, _y_train_b, _y_test_b = model_selection.train_test_split(X_best,
                                                                                        y,
                                                                                        test_size=0.25,
                                                                                        random_state=0,
                                                                                        )

        _X_train_w, _X_test_w, _y_train_w, _y_test_w = model_selection.train_test_split(X_worst,
                                                                                        y,
                                                                                        test_size=0.25,
                                                                                        random_state=0,
                                                                                        )
        # Fit and predict best features
        estimator.fit(_X_train_b, _y_train_b)
        y_predicted_best_test = estimator.predict(_X_test_b)
        y_predicted_best_train = estimator.predict(_X_train_b)

        # Fit and predict worst features
        estimator.fit(_X_train_w, _y_train_w)
        y_predicted_worst_test = estimator.predict(_X_test_w)
        y_predicted_worst_train = estimator.predict(_X_train_w)

        # Evaluate models
        score_best_test = scoring(_y_test_b, y_predicted_best_test, **scoring_kwargs)
        score_worst_test = scoring(_y_test_w, y_predicted_worst_test, **scoring_kwargs)
        score_best_train = scoring(_y_train_b, y_predicted_best_train, **scoring_kwargs)
        score_worst_train = scoring(_y_train_w, y_predicted_worst_train, **scoring_kwargs)

        rfe_result.append({
            'column': sorted_X.columns[i],
            'score_best_test': score_best_test,
            'score_worst_test': score_worst_test,
            'score_best_train': score_best_train,
            'score_worst_train': score_worst_train,

        })

    score_df = pd.DataFrame(rfe_result)

    #
    # NAIVE Models
    #
    _X_train_naive, _X_test_naive, _y_train_naive, _y_test_naive = model_selection.train_test_split(X, y,
                                                                                                    test_size=0.25,
                                                                                                    random_state=0,
                                                                                                    )
    #
    # Random Naive Model: randomly select values from _y_test_naive
    #
    N = 1000
    rnd_idx = np.random.randint(0, len(_y_test_naive), size=(N, len(_y_test_naive)))
    rnd_predictions = _y_test_naive.values[rnd_idx]
    rnd_errs = np.full(rnd_predictions.shape[0], np.nan)

    for i in range(rnd_predictions.shape[0]):
        rnd_errs[i] = scoring(_y_test_naive.values, rnd_predictions[i], **scoring_kwargs)
    score_df['naive_random'] = pd.Series(rnd_errs).dropna().mean()
    #
    # Global Mean Model: always use _y_test_naive.mean() as prediction
    #
    try:
        score_df['naive_mean'] = scoring(_y_test_naive, np.full(len(_y_test_naive), _y_test_naive.mean()), **scoring_kwargs)
    except ValueError:
        # In case if scoring is classification (skip naive average!)
        score_df['naive_mean'] = np.nan


    plt.figure()
    plt.title('Feature importance validation curve')

    plt.xlabel("Features by importance [worst -> best]")
    plt.ylabel(f'Score: {scoring.__name__}')
    plt.grid()

    plt.plot(score_df.index, score_df['score_best_train'], '--', color="g",
             label="Best training score")
    plt.plot(score_df.index, score_df['score_best_test'], '-', color="g",
             label="Best test score")

    plt.plot(score_df.index, score_df['score_worst_train'], '--', color="r",
             label="Worst training score")
    plt.plot(score_df.index, score_df['score_worst_test'], '-', color="r",
             label="Worst test score")

    # NAIVE
    plt.plot(score_df.index, score_df['naive_random'], '-', color="black",
             label="Naive Random test score")
    plt.plot(score_df.index, score_df['naive_mean'], '--', color="black",
             label="Naive Mean test score")

    #
    # Comparing to base estimator (ML model or dummy ML model MUST implement fit() and predict()!)
    #
    if base_estimator is not None:
        base_estimator.fit(_X_train_naive, _y_train_naive)
        _y_base_test = base_estimator.predict(_X_test_naive)
        _y_base_train = base_estimator.predict(_X_train_naive)

        score_df['score_base_test'] = scoring(_y_test_naive, _y_base_test, **scoring_kwargs)
        score_df['score_base_train'] = scoring(_y_train_naive, _y_base_train, **scoring_kwargs)

        plt.plot(score_df.index, score_df['score_base_test'], '-', color="blue",
                 label="Base test score")
        plt.plot(score_df.index, score_df['score_base_train'], '--', color="blue",
                 label="Base train score")

    plt.xticks(score_df.index, score_df['column'], rotation='vertical');
    plt.legend(loc="best");

    return score_df