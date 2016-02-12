# Database Schema and Feature Vector Templates

# Valid schema attribute types include: 'bool', 'float', 'int', 'str'

SCHEMA = [
	('attribute_name', 'type')
]

TIME_SCHEMA = [
	('time_attribute_name', 'type')
]

# Mappings from features to types
STATIC_FEATURE_TO_TYPE = dict(SCHEMA)
TIME_FEATURE_TO_TYPE = dict(TIME_SCHEMA)

# Stats to consider immediately upon admission
STATIC_0_FEATURES = [
	'attribute_name'
]

# Static features to consider after 24 hours following admission
STATIC_24_FEATURES = STATIC_0_FEATURES + [
	'some_other_attribute'
]

# Stats for a particular timestep for the timeseries values
TIMESERIES_FEATURES = [
	'timeseries_attribute'
]

# Stats over the patient's entire time series for a feature
TIMESERIES_STATS_FEATURES = [
	'min',
	'max',
	'mean',
	'median',
	'stdev',
	'expreg_A',
	'expreg_B',
	'linreg_yint',
	'linreg_slope',
	'num_timesteps'
]
