import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold


def _strat_kfold_confusion_mat(model, X, y, n_splits=10, n_repeats=10, random_state = 42):
    """
	Gives confusion matrices obtained from stratified k-fold cross validation. 

	Inputs:
	-------
	model: 
		sklearn estimator. it must have a fit and predict method
	
	X: ndarray, shape (number_of_samples, number_of_features)
		training data features
	
	y: ndarry, shape (number_of_samples,)
		training data labels

	Returns
	-------
	training_confusion_matrix: ndarray, shape = (n_splits, n_labels, n_labels) 
		n_repeats confusion matrices of model's prediction on training data

	test_confusion_matrix: ndarray, shape = (n_splits, n_labels, n_labels)
		n_repeats confusion matrices of model's prediction on training data

    """
    labels = np.unique(y)
    training_confusion_matrix_list = []
    validation_confusion_matrix_list = []

    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )
    splits = rskf.split(X, y)

    for train_index, validation_index in splits:

        X_train, X_validation = X[train_index], X[validation_index]
        y_train, y_validation = y[train_index], y[validation_index]

        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_validation_pred = model.predict(X_test)

        training_confusion_matrix = confusion_matrix(
            y_train, y_train_pred, labels=labels, normalize="all"
        )[:, :, None]
        validation_confusion_matrix = confusion_matrix(
            y_validation, y_validation_pred, labels=labels, normalize="all"
        )[:, :, None]

        training_confusion_matrix_list.append(training_confusion_matrix)
        validation_confusion_matrix_list.append(validation_confusion_matrix)
	# permute entries so that n_repeats is the 0-th dimension
    training_confusion_matrices = np.concatenate(training_confusion_matrix_list, axis=2)
    validation_confusion_matrices = np.concatenate(validation_confusion_matrix_list, axis=2)

	training_confusion_matrices = np.transpose(training_confusion_matrices)
	validation_confusion_matrices =  np.transpose(validation_confusion_matrices)

    return (training_confusion_matrices, test_confusion_matrices)

def _strat_kfold_confusion_matrix_summary_statistics(confusion_matrices):
	pass


def _strat_kfold_confusion_matrix_confidence_intervals(confusion_matrices, alpha = .05):
	
	pass 

def strat_kfold_confusion_matrix_summary(model, X, y, n_splits=10, n_repeats=10, random_state=42):

	pass

def plot_strat_kfold_confusion_matrix_summary():
	pass