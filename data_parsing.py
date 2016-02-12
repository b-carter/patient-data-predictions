# Various methods for file reading and data/string parsing

import ast
import numpy as np
import schema
from patient_objects import Patient

# Given a split string from the data file, parses and returns a Patient object
def parse_line(data, use_time_series=True):
	patient_values = {}
	i = 0
	for (column, t) in schema.SCHEMA:
		try:
			if data[i] == '': # a missing value
				patient_values[column] = None
				continue
			if t == 'int':
				patient_values[column] = int(data[i])
			if t == 'bool':
				patient_values[column] = ast.literal_eval(data[i])
			if t == 'str':
				patient_values[column] = data[i]
			if t == 'float':
				patient_values[column] = float(data[i])
		except (ValueError, SyntaxError) as e:
			patient_values[column] = None
		finally:
			i += 1
	if use_time_series:
		time_data = data[len(schema.SCHEMA):]
		time_step = 0
		while time_step_exists(data, time_step):
			j = len(schema.TIME_SCHEMA)*time_step
			for (time_column, time_t) in schema.TIME_SCHEMA:
				if time_column not in patient_values:
					patient_values[time_column] = []
				try:
					if time_step == 0 and time_column == 'SPECIAL_ATTRIBUTE':
						patient_values[time_column].append(0.0)
					elif time_data[j] == '': # a missing value
						patient_values[time_column].append(None)
					elif time_t == 'int':
						patient_values[time_column].append(int(time_data[j]))
					elif time_t == 'bool':
						patient_values[time_column].append(ast.literal_eval(time_data[j]))
					elif time_t == 'str':
						patient_values[time_column].append(time_data[j])
					elif time_t == 'float':
						patient_values[time_column].append(float(time_data[j]))
				except (ValueError, SyntaxError) as e:
					patient_values[time_column].append(None)
				finally:
					j += 1
			time_step += 1
	return Patient(patient_values)

# Determine if a patient had any information recorded during a time step 
def time_step_exists(data, time_step):
	index = len(schema.SCHEMA) + len(schema.TIME_SCHEMA)*time_step
	time_step_data = data[index:index+len(schema.TIME_SCHEMA)]
	for value in time_step_data:
		if value != '':
			return True
	return False

# Replace r'^Yes$' and r'^No$' with 'True' and 'False' respectively
def clean_line(data):
	new_data = []
	for d in data:
		if d == 'Yes':
			new_data.append('True')
		elif d == 'No':
			new_data.append('False')
		else:
			new_data.append(d)
	return new_data

# Given a file with relative location fileloc, returns a list of parsed Patient objects
# Ignores the first row if ignore_header_row is True
def parse_file(fileloc, ignore_header_row=True):
	patients = []
	with open(fileloc, 'rU') as f:
		if ignore_header_row:
			next(f)
		for line in f:
			data = clean_line(line.replace('\n', '').split('\t'))
			patient = parse_line(data)
			patients.append(patient)
	return patients

# Prints some stats on all non-string patient attributes across all patients
# patients is a list of Patient objects
def print_stats(patients):
	for (column, t) in schema.SCHEMA:
		if t != 'int' and t != 'float' and t != 'bool':
			continue
		col_values = [getattr(p, column) for p in patients if not getattr(p, column) is None]
		num_missing_values = len(patients) - len(col_values)
		num_missing_percent = num_missing_values / float(len(patients)) * 100.0
		# bool stats
		if t == 'bool':
			num_yes = col_values.count(True)
			num_no = col_values.count(False)
			print column.upper() + ' STATS:'
			print '\tnum yes: ' + str(num_yes)
			print '\tnum no:  ' + str(num_no)
			print '\tmissing: ' + str(num_missing_values) + ' (' + str(num_missing_percent) + '%)'
			print ''
			continue
		# int and float stats
		val_max = max(col_values)
		val_min = min(col_values)
		val_mean = np.mean(col_values)
		val_median = np.median(col_values)
		val_stdev = np.std(col_values)
		print column.upper() + ' STATS:'
		print '\tmin:     ' + str(val_min)
		print '\tmax:     ' + str(val_max)
		print '\tmean:    ' + str(val_mean)
		print '\tmedian:  ' + str(val_median)
		print '\tstdev:   ' + str(val_stdev)
		print '\tmissing: ' + str(num_missing_values) + ' (' + str(num_missing_percent) + '%)'
		print ''

def output_patients_time24_data(patients, filename):
	f = open(filename, 'w')
	f.write('transfused'+'\t')
	f.write('\t'.join(schema.STATIC_24_FEATURES)+'\t')
	f.write('\t'.join(schema.TIMESERIES_FEATURES)+'\t')
	f.write('\t'.join(schema.TIMESERIES_STATS_FEATURES))
	f.write('\n')
	for patient in patients:
		line = ''
		line += str(patient.transfused) + '\t'
		for feature in schema.STATIC_24_FEATURES:
			value = getattr(patient, feature)
			if value is None:
				line += '\t'
			else:
				line += str(value) + '\t'
		for feature in schema.TIMESERIES_FEATURES:
			value = getattr(patient, feature)[0]
			if value is None:
				line += '\t'
			else:
				line += str(value) + '\t'
		num_timesteps = patient.get_num_timesteps_in_minutes(1440)
		line += str(patient.get_min_for_timeseries('ATTRIBUTE', num_timesteps)) + '\t'
		line += str(patient.get_max_for_timeseries('ATTRIBUTE', num_timesteps)) + '\t'
		line += str(patient.get_mean_for_timeseries('ATTRIBUTE', num_timesteps)) + '\t'
		line += str(patient.get_median_for_timeseries('ATTRIBUTE', num_timesteps)) + '\t'
		line += str(patient.get_std_for_timeseries('ATTRIBUTE', num_timesteps)) + '\t'
		(expreg_A, expreg_B) = patient.get_expreg_for_timeseries('ATTRIBUTE', num_timesteps)
		line += str(expreg_A) + '\t'
		line += str(expreg_B) + '\t'
		(linreg_yint, linreg_slope) = patient.get_linreg_for_timeseries('ATTRIBUTE', num_timesteps)
		line += str(linreg_yint) + '\t'
		line += str(linreg_slope)
		f.write(line+'\n')
	f.close()
