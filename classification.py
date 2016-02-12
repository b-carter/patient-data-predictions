# Classification of data using various models
# Model type and settings configured in main

import data_parsing
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import schema
import pydot
from sklearn import cross_validation, ensemble, linear_model, svm, tree
from sklearn.externals.six import StringIO
from patient_objects import Patient, PatientFeatureVectors
from matplotlib import gridspec

def train_model(training_patients, patient_feature_vectors, model):
	X_train = np.array([patient_feature_vectors.get_patient_feature_vector(patient) for patient in training_patients])
	y_train = np.array([patient.transfused for patient in training_patients])
	model.fit(X_train, y_train)
	return model

def predict_transfused(test_patients, model, patient_feature_vectors):
	X_test = np.array([patient_feature_vectors.get_patient_feature_vector(patient) for patient in test_patients])
	predictions = model.predict(X_test)
	return predictions

def evaluate_prediction_accuracy(patients, predictions):
	num_correct = 0
	for i in xrange(len(patients)):
		patient = patients[i]
		correct_value = patient.transfused
		prediction = predictions[i]
		if prediction == correct_value:
			num_correct += 1
	percent_correct = float(num_correct) / len(patients) * 100.0
	return percent_correct

def eval_cross_validation(patients, patient_feature_vectors, clf):
	X = np.array([patient_feature_vectors.get_patient_feature_vector(patient) for patient in patients])
	y = np.array([patient.transfused for patient in patients])
	scores = cross_validation.cross_val_score(clf, X, y, cv=5)
	print scores
	print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)

def get_feature_names(run_type):
	if run_type == 'STATIC_0':
		feature_names = schema.STATIC_0_FEATURES
	elif run_type == 'STATIC_24':
		feature_names = schema.STATIC_24_FEATURES
	elif run_type == 'TIME_0':
		feature_names = schema.STATIC_0_FEATURES + schema.TIMESERIES_FEATURES
	elif run_type == 'TIME_24':
		feature_names = schema.STATIC_24_FEATURES + schema.TIMESERIES_FEATURES + schema.TIMESERIES_STATS_FEATURES
	elif run_type == 'TIME_ALL':
		feature_names = schema.STATIC_24_FEATURES + schema.TIMESERIES_FEATURES + schema.TIMESERIES_STATS_FEATURES
	else:
		raise ValueError('Trying to assemble feature names for invalid run_type: ' + str(run_type))
	return feature_names

def make_tree_pdf(model, filename):
	dot_data = StringIO()
	tree.export_graphviz(model, out_file=dot_data, feature_names=get_feature_names(run_type),
							class_names=['False', 'True'], filled=True, rounded=True, special_characters=True)
	graph = pydot.graph_from_dot_data(dot_data.getvalue())
	graph.write_pdf(filename)

def compute_important_features(model):
	importances = model.feature_importances_
	feature_names = get_feature_names(run_type)
	important_features = [(feature_names[i], importances[i]) for i in xrange(len(importances)) if importances[i] != 0.0]
	return important_features

# Takes in a list of lists representative of each cluster
def graph_clusters(cluster_representatives, attribute):
	n = len(cluster_representatives)
	if n <= 10:
		rows = 2
	if n > 10 and n <= 30:
		rows = 5
	if n > 30:
		rows = 10
	columns = int(math.ceil(len(cluster_representatives) / float(rows)))
	gs = gridspec.GridSpec(rows, columns)
	fig = plt.figure()
	cluster_number = 0
	for values in cluster_representatives:
		y_val = np.array(values)
		x_val = np.array(range(len(values)))
		ax = fig.add_subplot(gs[cluster_number])
		ax.plot(x_val,y_val, 'ko')
		fig.suptitle(attribute, fontsize=14)
		ax.set_title("P"+str(cluster_number+1), fontsize=10)
		xticks, xticklabels = plt.xticks()
		xmin = (3*xticks[0] - xticks[1])/2.
		xmax = (3*xticks[-1] - xticks[-2])/2.
		plt.xlim(xmin, xmax)
		yticks, yticklabels = plt.yticks()
		ymin = (3*yticks[0] - yticks[1])/2.
		ymax = (3*yticks[-1] - yticks[-2])/2.
		plt.ylim(ymin, ymax)
		z = np.polyfit(x_val, y_val, 2)
		p = np.poly1d(z)
		plt.plot(x_val, p(x_val), linewidth=1, c='r')
		plt.tick_params(axis = 'both', which='major', labelsize = 6)
		plt.tick_params(axis = 'both', which='minor', labelsize = 6)
		cluster_number += 1
	plt.show()

def graph_numeric_time_attr(attr, num_samples):
	sample = random.sample(all_patients, num_samples)
	cluster_data = []
	for p in sample:
		values = getattr(p, attr)
		if None in values:
			continue
		cluster_data.append(getattr(p, attr))
	return graph_clusters(cluster_data, attr)


if __name__ == '__main__':
	# parse the patients
	all_patients = data_parsing.parse_file('data.tsv')
	# only use patients that have > 1 measurements in first 24 hours
	all_patients = [p for p in all_patients if p.get_num_timesteps_in_minutes(1440) > 1]
	num_training_patients = int(len(all_patients) * 0.90) # use 90% of patients for training
	training_patients = all_patients[:num_training_patients]
	test_patients = all_patients[num_training_patients:]

	# set up the conditions for the feature vectors
	#run_type = 'STATIC_0' # uses static features based on initial data
	#run_type = 'STATIC_24' # uses static features from first 24 hours
	#run_type = 'TIME_0' # uses initial static features and first timestep data
	run_type = 'TIME_24' # uses static 24 features, first timestep data, and timeseries stats for all 'ATTRIBUTE' time points from first 24 hours
	#run_type = 'TIME_ALL' # uses static 24 features, first timestep data, and timeseries stats for 'ATTRIBUTE' across all time points

	# choose a classifier
	#model = linear_model.Perceptron()
	#model = ensemble.RandomForestClassifier()
	#model = ensemble.AdaBoostClassifier()
	#model = svm.SVC(C=1.0, kernel='rbf')
	model = tree.DecisionTreeClassifier(random_state=0, max_depth=3, max_features=None, criterion='entropy')

	# train a classifier model
	patient_feature_vectors = PatientFeatureVectors(training_patients, run_type, scale_feature_vectors=True)
	train_model(training_patients, patient_feature_vectors, model)

	# make predictions and evaluate
	training_predictions = predict_transfused(training_patients, model, patient_feature_vectors)
	test_predictions = predict_transfused(test_patients, model, patient_feature_vectors)
	training_accuracy = evaluate_prediction_accuracy(training_patients, training_predictions)
	test_accuracy = evaluate_prediction_accuracy(test_patients, test_predictions)
	print 'Training accuracy:  ' + str(training_accuracy) + '% correct'
	print 'Test accuracy:      ' + str(test_accuracy) + '% correct'
	print ''
	print 'Important features:\n' + str(compute_important_features(model))
	#make_tree_pdf(model, 'dtree.pdf')
	print ''
	eval_cross_validation(all_patients, patient_feature_vectors, model)
	#graph_numeric_time_attr('ATTRIBUTE', 20)
	print ''
