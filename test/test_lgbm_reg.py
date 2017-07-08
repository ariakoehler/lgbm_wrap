import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

import autosklearn.regression


def main():
    X, y = sklearn.datasets.load_boston(return_X_y=True)
    feature_types = (['numerical'] * 3) + ['categorical'] + (['numerical'] * 9)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=1)

    automl = autosklearn.regression.AutoSklearnRegressor(
        time_left_for_this_task=120, per_run_time_limit=30,
        tmp_folder='/tmp/autosklearn_lgbm_test',
        output_folder='/tmp/autosklearn_lgbm_out',
        delete_tmp_folder_after_terminate=False,
        include_estimators=['lightgb_machine'])
    automl.fit(X_train, y_train, dataset_name='boston',
               feat_type=feature_types)

    print(automl.show_models())
    predictions = automl.predict(X_test)
    print('R2 score:', sklearn.metrics.r2_score(y_test, predictions))


if __name__=='__main__':
    main()
