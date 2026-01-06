# Monitoring & Logging Guide

## Overview
The Heart Disease Prediction API includes comprehensive monitoring and logging capabilities for tracking usage, performance, and errors.

## Features

### 1. Structured JSON Logging
All log entries are formatted as JSON for easy parsing and analysis.

**Log Format**:
```json
{
  "timestamp": "2024-01-06T12:00:00Z",
  "level": "INFO",
  "message": "Request completed",
  "module": "main",
  "function": "after_request",
  "event": "request_completed",
  "method": "POST",
  "path": "/predict",
  "status_code": 200,
  "duration_ms": 45.23
}
```

### 2. Request/Response Logging
Every API request is logged with:
- Request method and path
- Client IP address
- Response status code
- Request duration in milliseconds

### 3. Prediction Logging
Predictions are logged with detailed information:
- Prediction outcome (0 or 1)
- Probability/confidence score
- Risk level (Low/Medium/High)
- Model type used

### 4. Error Tracking
Errors are logged with:
- Error type
- Error message
- Affected endpoint
- Stack trace (in debug mode)

---

## Metrics Endpoint

### GET /metrics
Returns real-time metrics about API usage.

**Response**:
```json
{
  "total_requests": 150,
  "total_predictions": 120,
  "predictions_disease": 65,
  "predictions_no_disease": 55,
  "total_errors": 2,
  "average_duration_ms": 47.5,
  "uptime_seconds": 3600,
  "error_rate": 1.33,
  "model_loaded": true
}
```

**Usage**:
```bash
curl http://localhost:8000/metrics
```

---

## Log Analysis

### Using the Log Analysis Script

```bash
# Analyze logs
python scripts/analyze_logs.py /path/to/log/file.log

# Example output:
======================================================================
LOG ANALYSIS REPORT
======================================================================

REQUEST STATISTICS
Total Requests: 150
Total Predictions: 120
Total Errors: 2
Error Rate: 1.33%

RESPONSE TIME STATISTICS
Average: 47.50 ms
Min: 20.15 ms
Max: 125.30 ms
P50: 45.20 ms
P95: 95.10 ms
P99: 118.50 ms

ENDPOINT USAGE
/predict: 120 (80.0%)
/health: 25 (16.7%)
/: 5 (3.3%)

PREDICTION STATISTICS
Total Predictions: 120

Outcome Distribution:
  disease: 65 (54.2%)
  no_disease: 55 (45.8%)

Risk Level Distribution:
  High: 35 (29.2%)
  Medium: 50 (41.7%)
  Low: 35 (29.2%)
======================================================================
```

---

## Monitoring Best Practices

### 1. Log Collection
Redirect logs to a file for analysis:
```bash
python app/main.py > logs/api.log 2>&1
```

### 2. Regular Analysis
Run log analysis periodically:
```bash
# Daily analysis
python scripts/analyze_logs.py logs/api_$(date +%Y%m%d).log

# Weekly summary
cat logs/api_*.log | python scripts/analyze_logs.py -
```

### 3. Metrics Monitoring
Poll the `/metrics` endpoint regularly:
```bash
# Monitor every 60 seconds
watch -n 60 'curl -s http://localhost:8000/metrics | python -m json.tool'
```

### 4. Alert on Errors
Set up alerts when error rate exceeds threshold:
```bash
# Check error rate
ERROR_RATE=$(curl -s http://localhost:8000/metrics | jq '.error_rate')
if (( $(echo "$ERROR_RATE > 5" | bc -l) )); then
    echo "High error rate detected: $ERROR_RATE%"
    # Send alert
fi
```

---

## Integration with Monitoring Tools

### Prometheus
The `/metrics` endpoint can be scraped by Prometheus.

**prometheus.yml**:
```yaml
scrape_configs:
  - job_name: 'heart-disease-api'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

### Grafana
Create dashboards using Prometheus data source:
- Request rate graph
- Response time percentiles
- Error rate alert
- Prediction distribution pie chart

### ELK Stack
Forward JSON logs to Elasticsearch:
```bash
# Using Filebeat
filebeat -e -c filebeat.yml
```

**filebeat.yml**:
```yaml
filebeat.inputs:
  - type: log
    enabled: true
    paths:
      - /path/to/logs/*.log
    json.keys_under_root: true

output.elasticsearch:
  hosts: ["localhost:9200"]
```

---

## Troubleshooting

### No Logs Appearing
1. Check log level: `LOG_LEVEL=DEBUG`
2. Verify logging configuration
3. Check file permissions

### Metrics Not Updating
1. Verify API is receiving requests
2. Check `/metrics` endpoint is accessible
3. Restart API if metrics seem stuck

### High Error Rate
1. Check logs for error patterns
2. Run log analysis for error breakdown
3. Review recent code changes
4. Check model loading status

---

## Performance Monitoring

### Key Metrics to Track
- **Average response time**: Should be <100ms
- **P95 response time**: Should be <200ms
- **Error rate**: Should be <1%
- **Prediction accuracy**: Monitor distribution

### Setting Baselines
1. Run load test to establish normal performance
2. Set alert thresholds based on baselines
3. Monitor for deviations

---

## Production Recommendations

1. **Centralized Logging**: Use ELK, Splunk, or cloud logging
2. **Real-time Monitoring**: Set up Prometheus + Grafana
3. **Alerting**: Configure alerts for critical metrics
4. **Log Rotation**: Implement log rotation to manage disk space
5. **Audit Logging**: Log all prediction requests for compliance
6. **Performance Profiling**: Regular performance analysis
