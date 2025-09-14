#!/usr/bin/env python3
"""
Monitoring and telemetry system tests.
"""

import pytest
import time
import json
import threading
from unittest.mock import Mock, patch
from queue import Queue
import numpy as np


class TestMetricsCollection:
    """Test comprehensive metrics collection system."""
    
    @pytest.mark.monitoring
    def test_system_metrics_collection(self):
        """Test collection of system performance metrics."""
        class SystemMetricsCollector:
            def __init__(self):
                self.metrics_buffer = []
                self.collection_interval = 1.0
                
            def collect_system_metrics(self):
                # Mock system metrics collection
                metrics = {
                    "timestamp": time.time(),
                    "cpu": {
                        "usage_percent": np.random.uniform(30, 80),
                        "cores": 8,
                        "frequency": 2.4
                    },
                    "memory": {
                        "total_gb": 16,
                        "used_gb": np.random.uniform(4, 12),
                        "available_gb": np.random.uniform(4, 12)
                    },
                    "gpu": {
                        "usage_percent": np.random.uniform(20, 90),
                        "memory_used_mb": np.random.uniform(1000, 6000),
                        "memory_total_mb": 8192,
                        "temperature_c": np.random.uniform(45, 75)
                    },
                    "disk": {
                        "read_mb_s": np.random.uniform(50, 200),
                        "write_mb_s": np.random.uniform(30, 150),
                        "usage_percent": 65
                    }
                }
                
                self.metrics_buffer.append(metrics)
                return metrics
            
            def get_aggregated_metrics(self, window_minutes=5):
                # Get metrics from the last N minutes
                cutoff_time = time.time() - (window_minutes * 60)
                recent_metrics = [m for m in self.metrics_buffer if m["timestamp"] >= cutoff_time]
                
                if not recent_metrics:
                    return None
                
                # Calculate aggregated statistics
                cpu_usage = [m["cpu"]["usage_percent"] for m in recent_metrics]
                memory_usage = [m["memory"]["used_gb"] for m in recent_metrics]
                gpu_usage = [m["gpu"]["usage_percent"] for m in recent_metrics]
                
                return {
                    "cpu": {
                        "avg": np.mean(cpu_usage),
                        "max": np.max(cpu_usage),
                        "min": np.min(cpu_usage)
                    },
                    "memory": {
                        "avg": np.mean(memory_usage),
                        "max": np.max(memory_usage),
                        "min": np.min(memory_usage)
                    },
                    "gpu": {
                        "avg": np.mean(gpu_usage),
                        "max": np.max(gpu_usage),
                        "min": np.min(gpu_usage)
                    }
                }
        
        collector = SystemMetricsCollector()
        
        # Collect several metrics samples
        for _ in range(5):
            metrics = collector.collect_system_metrics()
            assert "timestamp" in metrics
            assert "cpu" in metrics
            assert "memory" in metrics
            assert "gpu" in metrics
            time.sleep(0.1)  # Small delay between collections
        
        # Test aggregated metrics
        aggregated = collector.get_aggregated_metrics(1)  # Last 1 minute
        assert aggregated is not None
        assert 0 <= aggregated["cpu"]["avg"] <= 100
        assert 0 <= aggregated["memory"]["avg"] <= 20
        assert 0 <= aggregated["gpu"]["avg"] <= 100
    
    @pytest.mark.monitoring
    def test_application_metrics_collection(self, performance_tracker):
        """Test collection of application-specific metrics."""
        class ApplicationMetricsCollector:
            def __init__(self):
                self.processing_metrics = []
                self.error_metrics = []
                
            def record_processing_event(self, event_data):
                metrics = {
                    "timestamp": time.time(),
                    "event_type": event_data.get("type", "unknown"),
                    "processing_time_ms": event_data.get("duration", 0) * 1000,
                    "input_size": event_data.get("input_size", 0),
                    "output_quality": event_data.get("quality", 0),
                    "memory_peak_mb": event_data.get("memory_peak", 0),
                    "gpu_utilization": event_data.get("gpu_util", 0)
                }
                
                self.processing_metrics.append(metrics)
                return metrics
            
            def record_error_event(self, error_data):
                error_metrics = {
                    "timestamp": time.time(),
                    "error_type": error_data.get("type", "unknown"),
                    "error_message": error_data.get("message", ""),
                    "severity": error_data.get("severity", "medium"),
                    "context": error_data.get("context", {}),
                    "recovery_attempted": error_data.get("recovery", False)
                }
                
                self.error_metrics.append(error_metrics)
                return error_metrics
            
            def get_processing_statistics(self):
                if not self.processing_metrics:
                    return {}
                
                processing_times = [m["processing_time_ms"] for m in self.processing_metrics]
                quality_scores = [m["output_quality"] for m in self.processing_metrics]
                
                return {
                    "total_events": len(self.processing_metrics),
                    "avg_processing_time_ms": np.mean(processing_times),
                    "p95_processing_time_ms": np.percentile(processing_times, 95),
                    "avg_quality": np.mean(quality_scores),
                    "error_rate": len(self.error_metrics) / len(self.processing_metrics)
                }
        
        collector = ApplicationMetricsCollector()
        
        # Record some processing events
        test_events = [
            {"type": "face_detection", "duration": 0.15, "quality": 0.92, "input_size": 1920*1080},
            {"type": "face_swap", "duration": 0.45, "quality": 0.88, "input_size": 1920*1080},
            {"type": "face_detection", "duration": 0.12, "quality": 0.95, "input_size": 1280*720},
            {"type": "face_swap", "duration": 0.38, "quality": 0.85, "input_size": 1280*720}
        ]
        
        for event in test_events:
            collector.record_processing_event(event)
        
        # Record an error
        collector.record_error_event({
            "type": "memory_error",
            "message": "CUDA out of memory",
            "severity": "high",
            "recovery": True
        })
        
        # Verify metrics
        stats = collector.get_processing_statistics()
        assert stats["total_events"] == 4
        assert 100 <= stats["avg_processing_time_ms"] <= 500
        assert 0.8 <= stats["avg_quality"] <= 1.0
        assert stats["error_rate"] == 0.25  # 1 error out of 4 events
    
    @pytest.mark.monitoring
    def test_real_time_metrics_streaming(self):
        """Test real-time metrics streaming capabilities."""
        class MetricsStreamer:
            def __init__(self):
                self.subscribers = []
                self.streaming = False
                self.stream_queue = Queue()
                
            def subscribe(self, callback):
                self.subscribers.append(callback)
                
            def start_streaming(self):
                self.streaming = True
                
                def stream_worker():
                    while self.streaming:
                        # Generate mock real-time metric
                        metric = {
                            "timestamp": time.time(),
                            "type": "real_time",
                            "value": np.random.uniform(0, 100),
                            "source": "face_processor"
                        }
                        
                        # Notify all subscribers
                        for callback in self.subscribers:
                            try:
                                callback(metric)
                            except Exception:
                                pass
                        
                        time.sleep(0.1)  # 10 Hz streaming
                
                self.stream_thread = threading.Thread(target=stream_worker)
                self.stream_thread.daemon = True
                self.stream_thread.start()
            
            def stop_streaming(self):
                self.streaming = False
                if hasattr(self, 'stream_thread'):
                    self.stream_thread.join(timeout=1.0)
        
        streamer = MetricsStreamer()
        received_metrics = []
        
        # Subscribe to metrics
        def metric_handler(metric):
            received_metrics.append(metric)
        
        streamer.subscribe(metric_handler)
        
        # Test streaming
        streamer.start_streaming()
        time.sleep(0.5)  # Let it stream for 500ms
        streamer.stop_streaming()
        
        assert len(received_metrics) >= 3  # Should receive several metrics
        assert all("timestamp" in m for m in received_metrics)
        assert all("value" in m for m in received_metrics)


class TestTelemetrySystem:
    """Test telemetry data collection and transmission."""
    
    @pytest.mark.monitoring
    def test_telemetry_data_aggregation(self):
        """Test aggregation of telemetry data."""
        class TelemetryAggregator:
            def __init__(self):
                self.raw_data = []
                self.aggregated_data = {}
                
            def add_telemetry_point(self, data_point):
                data_point["timestamp"] = time.time()
                self.raw_data.append(data_point)
                
            def aggregate_by_time_window(self, window_size_seconds=60):
                if not self.raw_data:
                    return {}
                
                # Group data by time windows
                current_time = time.time()
                windows = {}
                
                for point in self.raw_data:
                    window_start = int((point["timestamp"] - current_time + window_size_seconds) // window_size_seconds) * window_size_seconds
                    
                    if window_start not in windows:
                        windows[window_start] = []
                    windows[window_start].append(point)
                
                # Aggregate each window
                aggregated = {}
                for window_start, points in windows.items():
                    metrics = {}
                    
                    # Group by metric type
                    by_type = {}
                    for point in points:
                        metric_type = point.get("type", "unknown")
                        if metric_type not in by_type:
                            by_type[metric_type] = []
                        by_type[metric_type].append(point.get("value", 0))
                    
                    # Calculate statistics for each type
                    for metric_type, values in by_type.items():
                        metrics[metric_type] = {
                            "count": len(values),
                            "mean": np.mean(values),
                            "std": np.std(values),
                            "min": np.min(values),
                            "max": np.max(values)
                        }
                    
                    aggregated[window_start] = metrics
                
                return aggregated
            
            def get_anomalies(self, threshold_std=2.0):
                # Detect anomalies in telemetry data
                anomalies = []
                
                if len(self.raw_data) < 10:  # Need sufficient data for anomaly detection
                    return anomalies
                
                # Group by metric type
                by_type = {}
                for point in self.raw_data:
                    metric_type = point.get("type", "unknown")
                    if metric_type not in by_type:
                        by_type[metric_type] = []
                    by_type[metric_type].append({
                        "timestamp": point["timestamp"],
                        "value": point.get("value", 0)
                    })
                
                # Detect anomalies for each metric type
                for metric_type, points in by_type.items():
                    values = [p["value"] for p in points]
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    
                    for point in points:
                        z_score = abs(point["value"] - mean_val) / std_val if std_val > 0 else 0
                        
                        if z_score > threshold_std:
                            anomalies.append({
                                "timestamp": point["timestamp"],
                                "type": metric_type,
                                "value": point["value"],
                                "z_score": z_score,
                                "severity": "high" if z_score > 3.0 else "medium"
                            })
                
                return anomalies
        
        aggregator = TelemetryAggregator()
        
        # Add telemetry data points
        for i in range(20):
            aggregator.add_telemetry_point({
                "type": "cpu_usage",
                "value": np.random.normal(50, 10)  # Normal distribution around 50%
            })
            
            aggregator.add_telemetry_point({
                "type": "memory_usage", 
                "value": np.random.normal(60, 5)   # Normal distribution around 60%
            })
            
            time.sleep(0.01)  # Small delay
        
        # Add some anomalous data points
        aggregator.add_telemetry_point({"type": "cpu_usage", "value": 95})  # Anomaly
        aggregator.add_telemetry_point({"type": "memory_usage", "value": 5})  # Anomaly
        
        # Test aggregation
        aggregated = aggregator.aggregate_by_time_window(60)
        assert len(aggregated) >= 1
        
        # Test anomaly detection
        anomalies = aggregator.get_anomalies()
        assert len(anomalies) >= 1  # Should detect the anomalous values
        assert any(a["type"] == "cpu_usage" and a["value"] == 95 for a in anomalies)
    
    @pytest.mark.monitoring
    def test_telemetry_transmission(self):
        """Test telemetry data transmission and buffering."""
        class TelemetryTransmitter:
            def __init__(self):
                self.transmission_buffer = []
                self.transmission_history = []
                self.max_buffer_size = 100
                self.transmission_interval = 5.0
                
            def buffer_telemetry(self, telemetry_data):
                self.transmission_buffer.append({
                    "data": telemetry_data,
                    "buffered_at": time.time()
                })
                
                # Auto-transmit if buffer is full
                if len(self.transmission_buffer) >= self.max_buffer_size:
                    self.transmit_buffer()
            
            def transmit_buffer(self):
                if not self.transmission_buffer:
                    return {"status": "no_data", "transmitted": 0}
                
                # Mock transmission
                transmission_batch = {
                    "transmission_id": f"batch_{len(self.transmission_history)}",
                    "timestamp": time.time(),
                    "data_points": len(self.transmission_buffer),
                    "payload": self.transmission_buffer.copy(),
                    "status": "success"
                }
                
                self.transmission_history.append(transmission_batch)
                transmitted_count = len(self.transmission_buffer)
                self.transmission_buffer.clear()
                
                return {"status": "success", "transmitted": transmitted_count}
            
            def get_transmission_stats(self):
                if not self.transmission_history:
                    return {"total_transmissions": 0, "total_data_points": 0}
                
                total_transmissions = len(self.transmission_history)
                total_data_points = sum(t["data_points"] for t in self.transmission_history)
                
                return {
                    "total_transmissions": total_transmissions,
                    "total_data_points": total_data_points,
                    "last_transmission": self.transmission_history[-1]["timestamp"],
                    "success_rate": 1.0  # Mock 100% success rate
                }
        
        transmitter = TelemetryTransmitter()
        
        # Buffer telemetry data
        for i in range(15):
            transmitter.buffer_telemetry({
                "metric": f"test_metric_{i}",
                "value": i * 10,
                "timestamp": time.time()
            })
        
        # Manual transmission
        result = transmitter.transmit_buffer()
        assert result["status"] == "success"
        assert result["transmitted"] == 15
        
        # Check transmission stats
        stats = transmitter.get_transmission_stats()
        assert stats["total_transmissions"] == 1
        assert stats["total_data_points"] == 15


class TestAlertingSystem:
    """Test alerting and notification system."""
    
    @pytest.mark.monitoring
    def test_threshold_based_alerting(self):
        """Test threshold-based alerting system."""
        class ThresholdAlerter:
            def __init__(self):
                self.alert_rules = {}
                self.active_alerts = {}
                self.alert_history = []
                
            def add_alert_rule(self, rule_id, metric_type, threshold, condition="greater_than"):
                self.alert_rules[rule_id] = {
                    "metric_type": metric_type,
                    "threshold": threshold,
                    "condition": condition,
                    "enabled": True
                }
            
            def check_metric(self, metric_type, value):
                triggered_alerts = []
                
                for rule_id, rule in self.alert_rules.items():
                    if not rule["enabled"] or rule["metric_type"] != metric_type:
                        continue
                    
                    should_alert = False
                    
                    if rule["condition"] == "greater_than" and value > rule["threshold"]:
                        should_alert = True
                    elif rule["condition"] == "less_than" and value < rule["threshold"]:
                        should_alert = True
                    elif rule["condition"] == "equals" and value == rule["threshold"]:
                        should_alert = True
                    
                    if should_alert:
                        alert = self._create_alert(rule_id, rule, value)
                        triggered_alerts.append(alert)
                        
                        # Track active alerts
                        self.active_alerts[rule_id] = alert
                        self.alert_history.append(alert)
                    elif rule_id in self.active_alerts:
                        # Clear alert if condition no longer met
                        del self.active_alerts[rule_id]
                
                return triggered_alerts
            
            def _create_alert(self, rule_id, rule, current_value):
                return {
                    "alert_id": f"alert_{rule_id}_{int(time.time())}",
                    "rule_id": rule_id,
                    "metric_type": rule["metric_type"],
                    "threshold": rule["threshold"],
                    "current_value": current_value,
                    "condition": rule["condition"],
                    "timestamp": time.time(),
                    "severity": self._calculate_severity(rule, current_value)
                }
            
            def _calculate_severity(self, rule, current_value):
                # Calculate severity based on how much the threshold is exceeded
                threshold = rule["threshold"]
                
                if rule["condition"] == "greater_than":
                    excess = (current_value - threshold) / threshold
                elif rule["condition"] == "less_than":
                    excess = (threshold - current_value) / threshold
                else:
                    excess = 0
                
                if excess > 0.5:
                    return "critical"
                elif excess > 0.2:
                    return "high"
                elif excess > 0.1:
                    return "medium"
                else:
                    return "low"
        
        alerter = ThresholdAlerter()
        
        # Configure alert rules
        alerter.add_alert_rule("cpu_high", "cpu_usage", 80, "greater_than")
        alerter.add_alert_rule("memory_high", "memory_usage", 90, "greater_than")
        alerter.add_alert_rule("quality_low", "output_quality", 0.7, "less_than")
        
        # Test normal conditions (no alerts)
        alerts = alerter.check_metric("cpu_usage", 65)
        assert len(alerts) == 0
        
        # Test alert conditions
        cpu_alerts = alerter.check_metric("cpu_usage", 85)
        assert len(cpu_alerts) == 1
        assert cpu_alerts[0]["rule_id"] == "cpu_high"
        assert cpu_alerts[0]["severity"] in ["low", "medium", "high", "critical"]
        
        memory_alerts = alerter.check_metric("memory_usage", 95)
        assert len(memory_alerts) == 1
        assert memory_alerts[0]["rule_id"] == "memory_high"
        
        quality_alerts = alerter.check_metric("output_quality", 0.6)
        assert len(quality_alerts) == 1
        assert quality_alerts[0]["rule_id"] == "quality_low"
        
        # Check active alerts
        assert len(alerter.active_alerts) == 3
        assert len(alerter.alert_history) == 3
    
    @pytest.mark.monitoring
    def test_composite_alerting(self):
        """Test composite alerting based on multiple conditions."""
        class CompositeAlerter:
            def __init__(self):
                self.metric_buffer = {}
                self.composite_rules = {}
                self.alerts = []
                
            def add_metric(self, metric_type, value, timestamp=None):
                if timestamp is None:
                    timestamp = time.time()
                
                if metric_type not in self.metric_buffer:
                    self.metric_buffer[metric_type] = []
                
                self.metric_buffer[metric_type].append({
                    "value": value,
                    "timestamp": timestamp
                })
                
                # Keep only recent metrics (last 5 minutes)
                cutoff = timestamp - 300
                self.metric_buffer[metric_type] = [
                    m for m in self.metric_buffer[metric_type] 
                    if m["timestamp"] >= cutoff
                ]
            
            def add_composite_rule(self, rule_id, conditions, logic="AND"):
                self.composite_rules[rule_id] = {
                    "conditions": conditions,
                    "logic": logic,
                    "enabled": True
                }
            
            def evaluate_composite_rules(self):
                triggered_alerts = []
                
                for rule_id, rule in self.composite_rules.items():
                    if not rule["enabled"]:
                        continue
                    
                    condition_results = []
                    
                    for condition in rule["conditions"]:
                        result = self._evaluate_condition(condition)
                        condition_results.append(result)
                    
                    # Apply logic
                    if rule["logic"] == "AND":
                        rule_triggered = all(condition_results)
                    elif rule["logic"] == "OR":
                        rule_triggered = any(condition_results)
                    else:
                        rule_triggered = False
                    
                    if rule_triggered:
                        alert = {
                            "alert_id": f"composite_{rule_id}_{int(time.time())}",
                            "rule_id": rule_id,
                            "conditions_met": rule["conditions"],
                            "timestamp": time.time(),
                            "type": "composite"
                        }
                        
                        triggered_alerts.append(alert)
                        self.alerts.append(alert)
                
                return triggered_alerts
            
            def _evaluate_condition(self, condition):
                metric_type = condition["metric"]
                operator = condition["operator"]
                threshold = condition["threshold"]
                window = condition.get("window", 60)  # Default 60 seconds
                
                if metric_type not in self.metric_buffer:
                    return False
                
                # Get recent metrics within the window
                cutoff = time.time() - window
                recent_metrics = [
                    m for m in self.metric_buffer[metric_type] 
                    if m["timestamp"] >= cutoff
                ]
                
                if not recent_metrics:
                    return False
                
                # Calculate metric based on aggregation type
                agg_type = condition.get("aggregation", "avg")
                values = [m["value"] for m in recent_metrics]
                
                if agg_type == "avg":
                    metric_value = np.mean(values)
                elif agg_type == "max":
                    metric_value = np.max(values)
                elif agg_type == "min":
                    metric_value = np.min(values)
                else:
                    metric_value = values[-1]  # Latest value
                
                # Evaluate condition
                if operator == ">":
                    return metric_value > threshold
                elif operator == "<":
                    return metric_value < threshold
                elif operator == ">=":
                    return metric_value >= threshold
                elif operator == "<=":
                    return metric_value <= threshold
                elif operator == "==":
                    return metric_value == threshold
                
                return False
        
        alerter = CompositeAlerter()
        
        # Add composite rule: High CPU AND High Memory
        alerter.add_composite_rule("performance_degradation", [
            {"metric": "cpu_usage", "operator": ">", "threshold": 80, "aggregation": "avg"},
            {"metric": "memory_usage", "operator": ">", "threshold": 85, "aggregation": "avg"}
        ], "AND")
        
        # Add composite rule: Low quality OR High error rate
        alerter.add_composite_rule("quality_issues", [
            {"metric": "output_quality", "operator": "<", "threshold": 0.7, "aggregation": "avg"},
            {"metric": "error_rate", "operator": ">", "threshold": 0.1, "aggregation": "max"}
        ], "OR")
        
        # Test conditions that should NOT trigger composite alert
        alerter.add_metric("cpu_usage", 75)  # Below threshold
        alerter.add_metric("memory_usage", 90)  # Above threshold
        
        alerts = alerter.evaluate_composite_rules()
        assert len(alerts) == 0  # AND condition not fully met
        
        # Test conditions that SHOULD trigger composite alert
        alerter.add_metric("cpu_usage", 85)  # Above threshold
        alerter.add_metric("memory_usage", 90)  # Above threshold
        
        alerts = alerter.evaluate_composite_rules()
        assert len(alerts) >= 1  # AND condition met
        assert any(a["rule_id"] == "performance_degradation" for a in alerts)
        
        # Test OR condition
        alerter.add_metric("output_quality", 0.6)  # Below threshold
        
        alerts = alerter.evaluate_composite_rules()
        quality_alerts = [a for a in alerts if a["rule_id"] == "quality_issues"]
        assert len(quality_alerts) >= 1  # OR condition met


if __name__ == "__main__":
    # Run monitoring tests
    pytest.main([
        __file__,
        "-v",
        "--tb=short"
    ])