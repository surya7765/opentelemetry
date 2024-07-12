from prometheus_client import Counter, Summary, Gauge

REQUEST_COUNT = Counter('request_count', 'Total number of requests')
REQUEST_LATENCY = Summary('request_latency', 'Request latency in seconds')
ERROR_COUNT = Counter('error_count', 'Total number of errors')
API_CALLS = Counter('api_calls', 'Number of API calls')
UNIQUE_USERS = Counter('unique_users', 'Number of unique users')
PEAK_USAGE = Gauge('peak_usage', 'Peak usage time')
ENDPOINT_POPULARITY = Counter('endpoint_popularity', 'Endpoint popularity')
PREDICTION_ACCURACY = Gauge('prediction_accuracy', 'Prediction accuracy of the model')
MODEL_VERSION = Gauge('model_version', 'Version of the model')
