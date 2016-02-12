# Defines the Patient and PatientFeatureVectors objects

from scipy import stats
from sklearn.preprocessing import StandardScaler
import data_parsing
import math
import numpy as np
import operator
import schema


class Patient:
	def __init__(self, attribute_data):
		for key in attribute_data:
			setattr(self, key, attribute_data[key])

	def get_num_timesteps(self):
		return len(getattr(self, 'ATTRIBUTE'))

	def get_num_timesteps_in_minutes(self, minutes):
		times = getattr(self, 'ATTRIBUTE')
		total_time = 0
		for timestep in xrange(len(times)-1):
			new_time = total_time + times[timestep]
			if new_time <= minutes:
				total_time = new_time
			else:
				return timestep + 1
		return len(times)

	def get_min_for_timeseries(self, attribute, num_timesteps=None):
		values = getattr(self, attribute)
		return min(values[:num_timesteps])

	def get_max_for_timeseries(self, attribute, num_timesteps=None):
		values = getattr(self, attribute)
		return max(values[:num_timesteps])

	def get_mean_for_timeseries(self, attribute, num_timesteps=None):
		values = getattr(self, attribute)
		return np.mean(values[:num_timesteps])

	def get_median_for_timeseries(self, attribute, num_timesteps=None):
		values = getattr(self, attribute)
		return np.median(values[:num_timesteps])

	def get_std_for_timeseries(self, attribute, num_timesteps=None):
		values = getattr(self, attribute)
		return np.std(values[:num_timesteps])

	def get_linreg_for_timeseries(self, attribute, num_timesteps=None):
		values = getattr(self, attribute)
		values = values[:num_timesteps]
		time = range(len(values))
		slope, yint, r_value, p_value, std_err = stats.linregress(time, values)
		return (yint, slope)

	def get_expreg_for_timeseries(self, attribute, num_timesteps=None):
		values = getattr(self, attribute)[:num_timesteps]
		log_values = [math.log(v) for v in values]
		time = range(len(values))
		# fitting log(y) = log(A) + log(B)*x
		log_B, log_A, r_value, p_value, std_err = stats.linregress(time, log_values)
		A = math.e**log_A
		B = math.e**log_B
		return (A, B)

	def get_polyreg_for_timeseries(self, attribute, poly_order=2, num_timesteps=None):
		values = getattr(self, attribute)[:num_timesteps]
		time = range(len(values))
		coeffs = np.polyfit(time, values, poly_order)
		return coeffs


class PatientFeatureVectors:
	def __init__(self, training_patients, run_type, scale_feature_vectors=True):
		self.run_type = run_type
		if run_type == 'STATIC_0':
			self.static_features = schema.STATIC_0_FEATURES
			self.first_timestep_features = False
			self.time_stats_features_24 = False
			self.time_stats_features_all = False
		elif run_type == 'STATIC_24':
			self.static_features = schema.STATIC_24_FEATURES
			self.first_timestep_features = False
			self.time_stats_features_24 = False
			self.time_stats_features_all = False
		elif run_type == 'TIME_0':
			self.static_features = schema.STATIC_0_FEATURES
			self.first_timestep_features = True
			self.time_stats_features_24 = False
			self.time_stats_features_all = False
		elif run_type == 'TIME_24':
			self.static_features = schema.STATIC_24_FEATURES
			self.first_timestep_features = True
			self.time_stats_features_24 = True
			self.time_stats_features_all = False
		elif run_type == 'TIME_ALL':
			self.static_features = schema.STATIC_24_FEATURES
			self.first_timestep_features = True
			self.time_stats_features_24 = False
			self.time_stats_features_all = True
		else:
			raise ValueError('Invalid run_type: ' + str(run_type))
		# compute medians and modes from training_patients for missing value replacement
		self.feature_to_median_static = {} # mean for static training_patients 'int' and 'float' features
		self.feature_to_mode_static = {} # mode for static training_patients 'str' and 'bool'
		self.feature_to_median_time = {} # mean for time training_patients 'int' and 'float' features
		self.feature_to_mode_time = {} # mode for time training_patients 'str' and 'bool'
		for (feature, t) in schema.SCHEMA:
			feature_all_patients = [getattr(patient, feature) for patient in training_patients]
			feature_all_patients = [v for v in feature_all_patients if v is not None]
			if len(feature_all_patients) == 0:
				continue
			if t == 'int' or t =='float':
				median = np.median(feature_all_patients)
				self.feature_to_median_static[feature] = median
			if t == 'str' or t == 'bool':
				value_to_count = {}
				for value in feature_all_patients:
					if value not in value_to_count:
						value_to_count[value] = 0
					value_to_count[value] += 1
				mode = max(value_to_count.iteritems(), key=operator.itemgetter(1))[0]
				self.feature_to_mode_static[feature] = mode
		for (feature, t) in schema.TIME_SCHEMA:
			feature_all_patients = [getattr(patient, feature) for patient in training_patients]
			feature_all_patients = [item for sublist in feature_all_patients for item in sublist]
			feature_all_patients = [v for v in feature_all_patients if v is not None]
			if len(feature_all_patients) == 0:
				continue
			if t == 'int' or t =='float':
				median = np.median(feature_all_patients)
				self.feature_to_median_time[feature] = median
			if t == 'str' or t == 'bool':
				value_to_count = {}
				for value in feature_all_patients:
					if value not in value_to_count:
						value_to_count[value] = 0
					value_to_count[value] += 1
				mode = max(value_to_count.iteritems(), key=operator.itemgetter(1))[0]
				self.feature_to_mode_time[feature] = mode
		# initialize and fit a StandardScaler if scaling feature vectors
		self.scaler = None
		if scale_feature_vectors:
			scaler = StandardScaler()
			X = [self.get_patient_feature_vector(p) for p in training_patients]
			scaler.fit(X)
			self.scaler = scaler

	def get_patient_feature_vector(self, patient):
		static_feature_vector = self.get_patient_static_feature_vector(patient)
		all_feature_vectors = [static_feature_vector]
		if self.first_timestep_features:
			timestep_feature_vector = self.get_timestep_feature_vector(patient, 0)
			all_feature_vectors.append(timestep_feature_vector)
		if self.time_stats_features_24:
			num_timesteps = patient.get_num_timesteps_in_minutes(1440)
			timeseries_stats_feature_vector = self.get_timeseries_stats_feature_vector(patient, 'ATTRIBUTE', num_timesteps=num_timesteps)
			all_feature_vectors.append(timeseries_stats_feature_vector)
		if self.time_stats_features_all:
			timeseries_stats_feature_vector = self.get_timeseries_stats_feature_vector(patient, 'ATTRIBUTE')
			all_feature_vectors.append(timeseries_stats_feature_vector)
		full_vector = np.concatenate(all_feature_vectors)
		# scale the feature vector if self.scaler is not None
		# if this is during initialization of PatientFeatureVectors, will be None during scaler fitting
		if not self.scaler is None:
			full_vector = self.scaler.transform([full_vector])[0]
		return full_vector

	# returns a feature vector corresponding to data points from static features
	def get_patient_static_feature_vector(self, patient):
		feature_vector = np.zeros(len(self.static_features))
		for i in xrange(len(feature_vector)):
			feature = self.static_features[i]
			t = schema.STATIC_FEATURE_TO_TYPE[feature]
			value = getattr(patient, feature)
			if value is None:
				if t == 'int' or t == 'float' and feature in self.feature_to_median_static:
					value = self.feature_to_median_static[feature]
				elif t == 'bool' or t == 'str' and feature in self.feature_to_mode_static:
					value = self.feature_to_mode_static[feature]
				else: # still missing, cannot replace with median or mode
					value = 0.0
			feature_vector[i] = float(value) # float maps true to 1, false to 0
		return feature_vector

	# returns a feature vector corresponding to data points from a particular timestep
	# timestep corresponding to 0 represents first timestep
	def get_timestep_feature_vector(self, patient, timestep):
		feature_vector = np.zeros(len(schema.TIMESERIES_FEATURES))
		for i in xrange(len(feature_vector)):
			feature = schema.TIMESERIES_FEATURES[i]
			t = schema.TIME_FEATURE_TO_TYPE[feature]
			value = getattr(patient, feature)[timestep]
			if value is None:
				if t == 'int' or t == 'float' and feature in self.feature_to_median_time:
					value = self.feature_to_median_time[feature]
				elif t == 'bool' or t == 'str' and feature in self.feature_to_mode_time:
					value = self.feature_to_mode_time[feature]
				else: # still missing, cannot replace with median or mode
					value = 0.0
			feature_vector[i] = float(value)
		return feature_vector

	# returns a feature vector corresponding to timeseries stats for a patient over the specified number of timesteps
	def get_timeseries_stats_feature_vector(self, patient, attribute, num_timesteps=None):
		feature_vector = np.zeros(len(schema.TIMESERIES_STATS_FEATURES))
		feature_vector[0] = patient.get_min_for_timeseries(attribute, num_timesteps) # min
		feature_vector[1] = patient.get_max_for_timeseries(attribute, num_timesteps) # max
		feature_vector[2] = patient.get_mean_for_timeseries(attribute, num_timesteps) # mean
		feature_vector[3] = patient.get_median_for_timeseries(attribute, num_timesteps) # median
		feature_vector[4] = patient.get_std_for_timeseries(attribute, num_timesteps) # stdev
		(expreg_A, expreg_B) = patient.get_expreg_for_timeseries(attribute, num_timesteps) # exp reg
		feature_vector[5] = expreg_A
		feature_vector[6] = expreg_B
		#(poly1, poly2, poly3) = patient.get_polyreg_for_timeseries(attribute, num_timesteps=num_timesteps, poly_order=1)
		(linreg_yint, linreg_slope) = patient.get_linreg_for_timeseries(attribute, num_timesteps) # lin reg
		feature_vector[7] = linreg_yint
		feature_vector[8] = linreg_slope
		if num_timesteps is None:
			num_timesteps_print = patient.get_num_timesteps()
		else:
			num_timesteps_print = num_timesteps
		feature_vector[9] = num_timesteps_print
		return feature_vector
