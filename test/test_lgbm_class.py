import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

import autosklearn.classification

def main():
    X, y = sklearn.datasets.load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, random_state=1)

    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=120, per_run_time_limit=30,
        tmp_folder='/tmp/autosklearn_lgbm_class_test',
        output_folder='/tmp/autosklearn_lgbm_class_out',
        delete_tmp_folder_after_terminate=False,
        include_estimators=['lightgb_machine']
    )

    automl.fit(X_train.copy(), y_train.copy(), dataset_name='digits')
    
    print(automl.show_models())
    predictions = automl.predict(X_test)
    print("Accuracy score", sklearn.metrics.accuracy_score(y_test, predictions))

if __name__=='__main__':
    main()
