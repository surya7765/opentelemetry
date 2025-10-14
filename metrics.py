from prometheus_client import start_http_server, Gauge, Counter, Histogram
import psutil
import time

# Step 1: Define Prometheus Metrics
cpu_usage_gauge = Gauge("cpu_usage", "CPU Usage Percentage")
memory_usage_gauge = Gauge("memory_usage", "Memory Usage Percentage")
api_calls_counter = Counter("api_calls", "Total number of API calls made")
api_errors_counter = Counter("api_errors", "Total number of errors encountered")
api_latency_histogram = Histogram("api_latency", "Latency of API calls in seconds")

# Step 2: Define functions to update metrics
def update_system_metrics():
    """Updates the CPU and memory usage metrics."""
    cpu_usage = psutil.cpu_percent(interval=None)
    memory_usage = psutil.virtual_memory().percent

    cpu_usage_gauge.set(cpu_usage)
    memory_usage_gauge.set(memory_usage)

def record_api_call():
    """Increments the API calls counter."""
    api_calls_counter.inc()

def record_error():
    """Increments the API errors counter."""
    api_errors_counter.inc()

def track_latency(start_time):
    """Records the latency of an API call."""
    latency = time.time() - start_time
    api_latency_histogram.observe(latency)


# Step 3: Start Prometheus metrics server
if __name__ == "__main__":
    # Start the Prometheus metrics server on a different port, e.g., 8001
    start_http_server(8001)
    print("Prometheus metrics available at http://localhost:8001/metrics")
    # Continuously update system metrics
    try:
        while True:
            update_system_metrics()
            time.sleep(5)  # Update metrics every 5 seconds
    except KeyboardInterrupt:
        print("Shutting down...")
