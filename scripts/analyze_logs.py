#!/usr/bin/env python3
"""
Log Analysis Script for Heart Disease Prediction API

Analyzes JSON-formatted logs and generates statistics.
"""
import json
import sys
from collections import defaultdict


def analyze_logs(log_file):
    """Parse and analyze log file"""

    stats = {
        'total_requests': 0,
        'total_predictions': 0,
        'total_errors': 0,
        'predictions_by_risk': defaultdict(int),
        'predictions_by_outcome': defaultdict(int),
        'response_times': [],
        'error_types': defaultdict(int),
        'endpoints': defaultdict(int)
    }

    try:
        with open(log_file, 'r') as f:
            for line in f:
                try:
                    log = json.loads(line.strip())

                    # Count completed requests
                    if log.get('event') == 'request_completed':
                        stats['total_requests'] += 1
                        stats['endpoints'][log.get('path', 'unknown')] += 1

                        if 'duration_ms' in log:
                            stats['response_times'].append(log['duration_ms'])

                    # Count predictions
                    elif log.get('event') == 'prediction':
                        stats['total_predictions'] += 1

                        risk_level = log.get('risk_level', 'unknown')
                        stats['predictions_by_risk'][risk_level] += 1

                        prediction = log.get('prediction')
                        if prediction == 1:
                            stats['predictions_by_outcome']['disease'] += 1
                        elif prediction == 0:
                            stats['predictions_by_outcome']['no_disease'] += 1

                    # Count errors
                    elif log.get('event') == 'error':
                        stats['total_errors'] += 1
                        error_type = log.get('error_type', 'unknown')
                        stats['error_types'][error_type] += 1

                except json.JSONDecodeError:
                    continue

    except FileNotFoundError:
        print(f"Error: Log file '{log_file}' not found")
        sys.exit(1)

    return stats


def print_statistics(stats):
    """Print formatted statistics"""

    print("\\n" + "=" * 70)
    print("LOG ANALYSIS REPORT")
    print("=" * 70)

    # Request stats
    print("\nREQUEST STATISTICS")
    print(f"Total Requests: {stats['total_requests']}")
    print(f"Total Predictions: {stats['total_predictions']}")
    print(f"Total Errors: {stats['total_errors']}")

    if stats['total_requests'] > 0:
        error_rate = (stats['total_errors'] / stats['total_requests']) * 100
        print(f"Error Rate: {error_rate:.2f}%")

    # Response time stats
    if stats['response_times']:
        print("\nRESPONSE TIME STATISTICS")
        print(f"Average: {sum(stats['response_times']) / len(stats['response_times']):.2f} ms")
        print(f"Min: {min(stats['response_times']):.2f} ms")
        print(f"Max: {max(stats['response_times']):.2f} ms")

        # Calculate percentiles
        sorted_times = sorted(stats['response_times'])
        p50 = sorted_times[len(sorted_times) // 2]
        p95 = sorted_times[int(len(sorted_times) * 0.95)]
        p99 = sorted_times[int(len(sorted_times) * 0.99)]

        print(f"P50: {p50:.2f} ms")
        print(f"P95: {p95:.2f} ms")
        print(f"P99: {p99:.2f} ms")

    # Endpoint stats
    if stats['endpoints']:
        print("\nENDPOINT USAGE")
        for endpoint, count in sorted(stats['endpoints'].items(), key=lambda x: x[1], reverse=True):
            percentage = (count / stats['total_requests']) * 100
            print(f"{endpoint}: {count} ({percentage:.1f}%)")

    # Prediction stats
    if stats['total_predictions'] > 0:
        print("\nPREDICTION STATISTICS")
        print(f"Total Predictions: {stats['total_predictions']}")

        if stats['predictions_by_outcome']:
            print("\nOutcome Distribution:")
            for outcome, count in stats['predictions_by_outcome'].items():
                percentage = (count / stats['total_predictions']) * 100
                print(f"  {outcome}: {count} ({percentage:.1f}%)")

        if stats['predictions_by_risk']:
            print("\nRisk Level Distribution:")
            for risk, count in sorted(stats['predictions_by_risk'].items()):
                percentage = (count / stats['total_predictions']) * 100
                print(f"  {risk}: {count} ({percentage:.1f}%)")

    # Error stats
    if stats['total_errors'] > 0:
        print("\nERROR STATISTICS")
        print(f"Total Errors: {stats['total_errors']}")

        if stats['error_types']:
            print("\nError Types:")
            for error_type, count in sorted(stats['error_types'].items(), key=lambda x: x[1], reverse=True):
                percentage = (count / stats['total_errors']) * 100
                print(f"  {error_type}: {count} ({percentage:.1f}%)")

    print("\\n" + "=" * 70)


def main():
    """Main execution"""
    if len(sys.argv) < 2:
        print("Usage: python analyze_logs.py <log_file>")
        print("\\nExample: python analyze_logs.py /tmp/flask_api.log")
        sys.exit(1)

    log_file = sys.argv[1]

    print(f"Analyzing logs from: {log_file}")
    stats = analyze_logs(log_file)
    print_statistics(stats)


if __name__ == '__main__':
    main()
