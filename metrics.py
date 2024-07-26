from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider, Counter, Histogram, ObservableGauge
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.semconv.resource import ResourceAttributes
import psutil
import random

# Set up resources
resource = Resource(attributes={
    ResourceAttributes.SERVICE_NAME: "my-fastapi-service"
})

# Metrics setup
otlp_exporter = OTLPMetricExporter(endpoint="http://localhost:4317", insecure=True)
metric_reader = PeriodicExportingMetricReader(otlp_exporter)
meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
metrics.set_meter_provider(meter_provider)
meter = metrics.get_meter(__name__)

# Define metrics
response_time_histogram = meter.create_histogram("http.server.duration", unit="ms", description="Response time of HTTP requests")
api_calls_counter = meter.create_counter("api.calls", unit="1", description="Number of API calls")
unique_users_counter = meter.create_counter("unique.users", unit="1", description="Number of unique users")

# Define observable gauges
def record_resource_utilization(callback_options):
    cpu_usage = psutil.cpu_percent()
    memory_usage = psutil.virtual_memory().used
    # Yield values as a single tuple
    yield (cpu_usage,)
    yield (memory_usage,)

def record_prediction_accuracy(callback_options):
    accuracy = random.random()  # Simulate accuracy
    # Yield values as a single tuple
    yield (accuracy,)

# Register observable gauges with callbacks
meter.create_observable_gauge("resource.cpu_usage", callbacks=[record_resource_utilization])
meter.create_observable_gauge("resource.memory_consumption", callbacks=[record_resource_utilization])
meter.create_observable_gauge("model.prediction_accuracy", callbacks=[record_prediction_accuracy])

def record_metrics(request, result):
    response_time = random.randint(50, 200)  # Example: Simulate response time
    response_time_histogram.record(response_time)
    api_calls_counter.add(1)

def setup_metrics():
    meter_provider.start()
