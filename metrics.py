from opentelemetry import metrics
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics import MeterProvider as SDKMeterProvider, ObservableGauge
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
import psutil
import time

# Set up resources
resource = Resource(attributes={
    "service.name": "mlass",
    "service.version": "1.0.0",
    "service.instance.id": "instance-1"
})

# Configure OpenTelemetry Metrics
metric_exporter = OTLPMetricExporter(endpoint="http://localhost:4318", insecure=True)
metric_reader = PeriodicExportingMetricReader(metric_exporter)
metrics_provider = SDKMeterProvider(resource=resource, metric_readers=[metric_reader])
metrics.set_meter_provider(metrics_provider)
meter = metrics.get_meter_provider().get_meter(__name__)

# Observable gauges for CPU and Memory utilization
def cpu_usage_observable(options: metrics.CallbackOptions):
    return [metrics.Observation(value=psutil.cpu_percent(interval=None))]

def memory_usage_observable(options: metrics.CallbackOptions):
    return [metrics.Observation(value=psutil.virtual_memory().percent)]

# Create observable gauges
meter.create_observable_gauge(
    "cpu_usage",
    callbacks=[cpu_usage_observable],
    description="CPU Usage",
    unit="percent"
)

meter.create_observable_gauge(
    "memory_usage",
    callbacks=[memory_usage_observable],
    description="Memory Usage",
    unit="percent"
)

# Track API Calls, Latency, and Errors
performance_counter = meter.create_counter("api_calls", description="Count of API calls")
latency_histogram = meter.create_histogram("api_latency", description="Latency of API calls")
error_counter = meter.create_counter("api_errors", description="Count of API errors")

def record_api_call():
    performance_counter.add(1)

def track_latency(start_time):
    end_time = time.time()
    latency = end_time - start_time
    latency_histogram.record(latency)

def record_error():
    error_counter.add(1)

def collect_metrics():
    metrics_data = {
        "cpu_usage": psutil.cpu_percent(interval=None),
        "memory_usage": psutil.virtual_memory().percent,
        "api_calls": performance_counter,
        "api_latency": latency_histogram,
        "api_errors": error_counter
    }
    return metrics_data
