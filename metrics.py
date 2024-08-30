from datetime import datetime
import re
from urllib.parse import urlparse
from opentelemetry import metrics, trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.metrics import MeterProvider, ObservableGauge
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from fastapi import FastAPI, Request
import psutil
import time
import logging
import json

# Global variables for service information
SERVICE_NAME = "mlass"
SERVICE_VERSION = "1.0.0"
DEFAULT_INSTANCE_ID = "default-instance-mlass"
instance_id = DEFAULT_INSTANCE_ID

# Flag to ensure metrics, tracing, and logging are initialized only once
metrics_initialized = False
tracing_initialized = False
logging_initialized = False


# Custom log formatter
class CustomLogFormatter(logging.Formatter):
    def __init__(self, instance_id_func, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.instance_id_func = instance_id_func

    def format(self, record):
        instance_id = self.instance_id_func()  # Dynamically get the current instanceId
        custom_log = {
            "instanceId": instance_id,
            "loglevel": record.levelname.lower(),
            "message": record.getMessage()
        }
        return json.dumps(custom_log)

# Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to get the current instanceId
def get_current_instance_id():
    global instance_id
    return instance_id

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

# Replace default handler with custom handler
handler = logger.handlers[0]
handler.setFormatter(CustomLogFormatter(get_current_instance_id))

# Tracer initialization
tracer = trace.get_tracer(__name__)

# Initialize OpenTelemetry logging
def initialize_logging(resource: Resource):
    global logging_initialized
    if logging_initialized:
        return
    log_exporter = OTLPLogExporter(endpoint="http://otel-collector:4317", insecure=True)
    log_provider = LoggerProvider(resource=resource)
    log_provider.add_log_record_processor(BatchLogRecordProcessor(log_exporter))
    otlp_handler = LoggingHandler(level=logging.INFO, logger_provider=log_provider)
    logger.addHandler(otlp_handler)
    logging_initialized = True

# Global variables for metrics
API_CALLS = None
API_LATENCY = None
ERROR_COUNT = None
REQUEST_COUNT = None
ENDPOINT_POPULARITY = None
REQUEST_TIME = None
PEAK_USAGE_TIME = None
REQUEST_LATENCY = None
MODEL_VERSION = None
PREDICTION_ACCURACY = None

def initialize_metrics(resource: Resource):
    global API_CALLS, API_LATENCY, ERROR_COUNT, REQUEST_COUNT, ENDPOINT_POPULARITY, REQUEST_TIME, PEAK_USAGE_TIME, REQUEST_LATENCY, MODEL_VERSION, PREDICTION_ACCURACY, metrics_initialized

    if metrics_initialized:
        return  # Prevent re-initialization

    metric_exporter = OTLPMetricExporter(endpoint="http://otel-collector:4317", insecure=True)
    metric_reader = PeriodicExportingMetricReader(metric_exporter)
    metrics_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(metrics_provider)
    meter = metrics.get_meter(__name__)

    # Initialize metrics
    API_CALLS = meter.create_counter("api_calls", unit="1", description="Count of API calls")
    API_LATENCY = meter.create_histogram("api_latency", description="Latency of API calls")
    ERROR_COUNT = meter.create_counter("ERROR_COUNT", unit="1", description="Count of API errors")
    REQUEST_COUNT = meter.create_counter("request_count", unit="1", description="Count of requests to each endpoint")
    ENDPOINT_POPULARITY = meter.create_counter("endpoint_popularity", description="Popularity of each endpoint based on request count")
    REQUEST_TIME = meter.create_histogram("request_time", description="Time taken to process each request")
    PEAK_USAGE_TIME = meter.create_up_down_counter("peak_usage_time", description="Peak usage time of the API")
    REQUEST_LATENCY = meter.create_histogram("request_latency", description="Latency of requests")
    MODEL_VERSION = meter.create_observable_gauge("model_version", description="Version of the model used")
    PREDICTION_ACCURACY = meter.create_observable_gauge("prediction_accuracy", description="Accuracy of predictions made by the model")

    def cpu_usage_observable(options: metrics.CallbackOptions):
        return [metrics.Observation(value=psutil.cpu_percent(interval=None))]

    def memory_usage_observable(options: metrics.CallbackOptions):
        return [metrics.Observation(value=psutil.virtual_memory().percent)]

    meter.create_observable_gauge("cpu_usage", callbacks=[cpu_usage_observable], description="CPU Usage", unit="percent")
    meter.create_observable_gauge("memory_usage", callbacks=[memory_usage_observable], description="Memory Usage", unit="percent")

    metrics_initialized = True  # Set flag to true after initialization

def initialize_tracing(resource: Resource):
    global tracing_initialized
    if tracing_initialized:
        return
    trace.set_tracer_provider(TracerProvider(resource=resource))
    otlp_exporter = OTLPSpanExporter(endpoint="http://otel-collector:4317", insecure=True)
    span_processor = BatchSpanProcessor(otlp_exporter)
    trace.get_tracer_provider().add_span_processor(span_processor)
    tracing_initialized = True

def create_resource(MLaaS_ID: str, SERVICE_ID: str, APP_ID: str, CREATED_DATE: datetime) -> Resource:
    print(f"Creating resource with attributes {MLaaS_ID} {SERVICE_ID}")
    return Resource(attributes={
        "service.name": SERVICE_NAME,
        "service.version": SERVICE_VERSION,
        "service.mlass.id": MLaaS_ID,
        "service.id": SERVICE_ID,
        "app.id": APP_ID,
        "created.date": CREATED_DATE.isoformat()
    })

# Validate instance ID and service API key
def validate_instance_id_and_service_api_key(instance_id, service_api_key):
    print("validating instance ID and service API key")
    return {
        "status": True,
        "msg": "validation successful",
        "data": {
            "api_key": "yujh",
            "MLasS_id": instance_id,
            "appId": 123,
            "serviceId": "nvj123",
            "createdDate": datetime.now(),
        }
    }

def instrument_app(app: FastAPI):
    FastAPIInstrumentor.instrument_app(app)
    
    @app.middleware("http")
    async def metrics_and_traces_middleware(request: Request, call_next):
        global instance_id
        verification_info = {}
        try:
            url = str(request.url)
            parsed_url = urlparse(url)
            pattern = re.compile(r'/v[1-9]/([a-zA-Z0-9\-]+)/')
            match = pattern.search(parsed_url.path)
            instance_id_extract = match.group(1) if match else None
            path_params = {'MLaaS_ID': instance_id_extract}
            instance_id = path_params.get("MLaaS_ID", DEFAULT_INSTANCE_ID)
            serviceApiKey = request.headers.get("serviceApiKey", "Bearer default-api-key").split(" ")[-1]

            if instance_id and serviceApiKey:
                verification_info = validate_instance_id_and_service_api_key(instance_id, serviceApiKey)
                resource_data = verification_info.get("data", {})
                resource = create_resource(
                    MLaaS_ID=instance_id,
                    SERVICE_ID=resource_data.get("serviceId") or "unknown_service",
                    APP_ID=resource_data.get("appId") or "unknown_app",
                    CREATED_DATE=resource_data.get("createdDate")
                )
                initialize_metrics(resource)
                initialize_tracing(resource)
                initialize_logging(resource)
                print("Initialization of resources completed successfully 🎉🎉🎉")

            start_time = time.time()

            with trace.get_tracer(__name__).start_as_current_span(request.url.path) as span:
                span.set_attribute("instanceId", verification_info.get("instance_id", instance_id))
                span.set_attribute("serviceApiKey", verification_info.get("api_key", serviceApiKey))
                response = await call_next(request)
                elapsed_time = time.time() - start_time
                API_LATENCY.record(elapsed_time)
                API_CALLS.add(1)
                REQUEST_COUNT.add(1, attributes={
                    "MLaaS_ID": instance_id,
                    "path": request.url.path
                })
                ENDPOINT_POPULARITY.add(1, attributes={
                    "MLaaS_ID": instance_id,
                    "path": request.url.path
                })
                REQUEST_TIME.record(elapsed_time, attributes={
                    "MLaaS_ID": instance_id,
                    "path": request.url.path
                })
                PEAK_USAGE_TIME.add(elapsed_time)

            return response

        except Exception as e:
            ERROR_COUNT.add(1, attributes={
                "MLaaS_ID": instance_id,
                "path": request.url.path
            })
            logger.error(f"An error occurred: {str(e)}")
            raise e

    return app
