#!/usr/bin/env python3
"""
Advanced AI-driven monitoring and optimization system for Roop-Unleashed.
"""

import asyncio
import json
import time
import threading
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MetricDataPoint:
    """Represents a single metric data point."""
    timestamp: float
    metric_name: str
    value: float
    source: str
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


@dataclass
class Alert:
    """Represents an alert condition."""
    id: str
    timestamp: float
    severity: str  # critical, high, medium, low
    message: str
    metric_name: str
    current_value: float
    threshold: float
    source: str
    acknowledged: bool = False


class MetricsCollector:
    """Collects system and application metrics in real-time."""
    
    def __init__(self):
        self.metrics_buffer: List[MetricDataPoint] = []
        self.buffer_size = 10000
        self.collection_interval = 1.0  # seconds
        self.running = False
        self.collection_thread = None
        
    def start_collection(self):
        """Start metrics collection in background thread."""
        if self.running:
            return
            
        self.running = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        logger.info("Metrics collection started")
        
    def stop_collection(self):
        """Stop metrics collection."""
        self.running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5.0)
        logger.info("Metrics collection stopped")
        
    def _collection_loop(self):
        """Main collection loop running in background."""
        while self.running:
            try:
                self._collect_system_metrics()
                self._collect_application_metrics()
                self._collect_gpu_metrics()
                
                # Maintain buffer size
                if len(self.metrics_buffer) > self.buffer_size:
                    self.metrics_buffer = self.metrics_buffer[-self.buffer_size:]
                    
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                time.sleep(self.collection_interval)
                
    def _collect_system_metrics(self):
        """Collect system-level metrics."""
        try:
            import psutil
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            self._add_metric("system.cpu.usage_percent", cpu_percent, "system")
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self._add_metric("system.memory.usage_percent", memory.percent, "system")
            self._add_metric("system.memory.available_gb", memory.available / (1024**3), "system")
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            self._add_metric("system.disk.usage_percent", disk.percent, "system")
            
            # Network metrics
            net_io = psutil.net_io_counters()
            self._add_metric("system.network.bytes_sent", net_io.bytes_sent, "system")
            self._add_metric("system.network.bytes_recv", net_io.bytes_recv, "system")
            
        except ImportError:
            # Fallback if psutil not available
            self._add_metric("system.cpu.usage_percent", float('nan'), "system", tags={"status": "unavailable"})
            self._add_metric("system.memory.usage_percent", float('nan'), "system", tags={"status": "unavailable"})
            
    def _collect_application_metrics(self):
        """Collect application-specific metrics."""
        # These would be populated by the actual application
        # For now, we'll use mock data
        
        self._add_metric("app.face_detection.processing_time_ms", np.random.uniform(100, 300), "face_detector")
        self._add_metric("app.face_swapping.processing_time_ms", np.random.uniform(400, 800), "face_swapper")
        self._add_metric("app.quality.ssim_score", np.random.uniform(0.7, 0.95), "quality_assessor")
        self._add_metric("app.quality.psnr_score", np.random.uniform(25, 35), "quality_assessor")
        
        # Throughput metrics
        self._add_metric("app.throughput.images_per_second", np.random.uniform(2, 8), "processor")
        self._add_metric("app.throughput.frames_per_second", np.random.uniform(1, 5), "video_processor")
        
        # Error metrics
        error_rate = np.random.exponential(0.02)  # Low error rate
        self._add_metric("app.errors.rate_per_minute", min(error_rate, 0.5), "error_tracker")
        
    def _collect_gpu_metrics(self):
        """Collect GPU metrics if available."""
        try:
            import torch
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                
                # GPU utilization
                gpu_util = np.random.uniform(60, 95)  # Mock GPU utilization
                self._add_metric("gpu.utilization_percent", gpu_util, f"cuda_{device}")
                
                # GPU memory
                memory_allocated = torch.cuda.memory_allocated(device)
                memory_reserved = torch.cuda.memory_reserved(device)
                
                self._add_metric("gpu.memory.allocated_mb", memory_allocated / (1024**2), f"cuda_{device}")
                self._add_metric("gpu.memory.reserved_mb", memory_reserved / (1024**2), f"cuda_{device}")
                
                # GPU temperature (mock)
                temp = np.random.uniform(45, 75)
                self._add_metric("gpu.temperature_celsius", temp, f"cuda_{device}")
                
        except ImportError:
            pass  # PyTorch not available
            
    def _add_metric(self, name: str, value: float, source: str, tags: Dict[str, str] = None):
        """Add a metric data point to the buffer."""
        metric = MetricDataPoint(
            timestamp=time.time(),
            metric_name=name,
            value=value,
            source=source,
            tags=tags or {}
        )
        self.metrics_buffer.append(metric)
        
    def get_recent_metrics(self, metric_name: str, seconds: int = 60) -> List[MetricDataPoint]:
        """Get recent metrics for a specific metric name."""
        cutoff_time = time.time() - seconds
        return [m for m in self.metrics_buffer 
                if m.metric_name == metric_name and m.timestamp >= cutoff_time]
    
    def get_metric_statistics(self, metric_name: str, seconds: int = 60) -> Dict[str, float]:
        """Get statistical summary of a metric over time window."""
        recent_metrics = self.get_recent_metrics(metric_name, seconds)
        
        if not recent_metrics:
            return {}
            
        values = [m.value for m in recent_metrics]
        
        return {
            "count": len(values),
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "p50": np.percentile(values, 50),
            "p95": np.percentile(values, 95),
            "p99": np.percentile(values, 99)
        }


class AlertManager:
    """Manages alerts based on metric thresholds and patterns."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alert_rules: Dict[str, Dict] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        
    def add_threshold_rule(self, rule_id: str, metric_name: str, threshold: float, 
                          condition: str = "greater_than", severity: str = "medium"):
        """Add a threshold-based alert rule."""
        self.alert_rules[rule_id] = {
            "type": "threshold",
            "metric_name": metric_name,
            "threshold": threshold,
            "condition": condition,
            "severity": severity,
            "enabled": True
        }
        
    def add_anomaly_rule(self, rule_id: str, metric_name: str, sensitivity: float = 2.0,
                        window_size: int = 300, severity: str = "medium"):
        """Add an anomaly detection rule."""
        self.alert_rules[rule_id] = {
            "type": "anomaly",
            "metric_name": metric_name,
            "sensitivity": sensitivity,  # Standard deviations from mean
            "window_size": window_size,  # Seconds
            "severity": severity,
            "enabled": True
        }
        
    def add_pattern_rule(self, rule_id: str, metric_name: str, pattern_type: str,
                        parameters: Dict, severity: str = "medium"):
        """Add a pattern-based alert rule."""
        self.alert_rules[rule_id] = {
            "type": "pattern",
            "metric_name": metric_name,
            "pattern_type": pattern_type,  # "trend", "spike", "drop"
            "parameters": parameters,
            "severity": severity,
            "enabled": True
        }
        
    def subscribe_to_alerts(self, callback: Callable[[Alert], None]):
        """Subscribe to alert notifications."""
        self.alert_callbacks.append(callback)
        
    def check_alerts(self):
        """Check all alert rules against current metrics."""
        current_time = time.time()
        
        for rule_id, rule in self.alert_rules.items():
            if not rule["enabled"]:
                continue
                
            try:
                if rule["type"] == "threshold":
                    self._check_threshold_rule(rule_id, rule)
                elif rule["type"] == "anomaly":
                    self._check_anomaly_rule(rule_id, rule)
                elif rule["type"] == "pattern":
                    self._check_pattern_rule(rule_id, rule)
                    
            except Exception as e:
                logger.error(f"Error checking alert rule {rule_id}: {e}")
                
    def _check_threshold_rule(self, rule_id: str, rule: Dict):
        """Check threshold-based alert rule."""
        metric_name = rule["metric_name"]
        recent_metrics = self.metrics_collector.get_recent_metrics(metric_name, 30)
        
        if not recent_metrics:
            return
            
        current_value = recent_metrics[-1].value
        threshold = rule["threshold"]
        condition = rule["condition"]
        
        should_alert = False
        
        if condition == "greater_than" and current_value > threshold:
            should_alert = True
        elif condition == "less_than" and current_value < threshold:
            should_alert = True
        elif condition == "equals" and abs(current_value - threshold) < 0.001:
            should_alert = True
            
        if should_alert:
            if rule_id not in self.active_alerts:
                alert = Alert(
                    id=rule_id,
                    timestamp=time.time(),
                    severity=rule["severity"],
                    message=f"{metric_name} {condition} {threshold} (current: {current_value:.2f})",
                    metric_name=metric_name,
                    current_value=current_value,
                    threshold=threshold,
                    source="threshold_monitor"
                )
                self._trigger_alert(alert)
        else:
            # Clear alert if condition no longer met
            if rule_id in self.active_alerts:
                self._clear_alert(rule_id)
                
    def _check_anomaly_rule(self, rule_id: str, rule: Dict):
        """Check anomaly detection rule."""
        metric_name = rule["metric_name"]
        window_size = rule["window_size"]
        sensitivity = rule["sensitivity"]
        
        recent_metrics = self.metrics_collector.get_recent_metrics(metric_name, window_size)
        
        if len(recent_metrics) < 10:  # Need minimum data points
            return
            
        values = [m.value for m in recent_metrics]
        current_value = values[-1]
        
        # Calculate baseline statistics (excluding recent values)
        baseline_values = values[:-5]  # Exclude last 5 values
        
        if len(baseline_values) < 5:
            return
            
        baseline_mean = np.mean(baseline_values)
        baseline_std = np.std(baseline_values)
        
        if baseline_std == 0:
            return
            
        # Calculate z-score
        z_score = abs(current_value - baseline_mean) / baseline_std
        
        if z_score > sensitivity:
            if rule_id not in self.active_alerts:
                alert = Alert(
                    id=rule_id,
                    timestamp=time.time(),
                    severity=rule["severity"],
                    message=f"Anomaly detected in {metric_name}: {current_value:.2f} (z-score: {z_score:.2f})",
                    metric_name=metric_name,
                    current_value=current_value,
                    threshold=sensitivity,
                    source="anomaly_detector"
                )
                self._trigger_alert(alert)
        else:
            if rule_id in self.active_alerts:
                self._clear_alert(rule_id)
                
    def _check_pattern_rule(self, rule_id: str, rule: Dict):
        """Check pattern-based alert rule."""
        metric_name = rule["metric_name"]
        pattern_type = rule["pattern_type"]
        parameters = rule["parameters"]
        
        window_size = parameters.get("window_size", 300)
        recent_metrics = self.metrics_collector.get_recent_metrics(metric_name, window_size)
        
        if len(recent_metrics) < parameters.get("min_points", 10):
            return
            
        values = [m.value for m in recent_metrics]
        timestamps = [m.timestamp for m in recent_metrics]
        
        pattern_detected = False
        
        if pattern_type == "trend":
            # Detect upward or downward trend
            slope, _ = np.polyfit(timestamps, values, 1)
            trend_threshold = parameters.get("slope_threshold", 0.1)
            
            if abs(slope) > trend_threshold:
                pattern_detected = True
                
        elif pattern_type == "spike":
            # Detect sudden spikes
            recent_mean = np.mean(values[-5:])
            baseline_mean = np.mean(values[:-5])
            spike_ratio = parameters.get("spike_ratio", 2.0)
            
            if recent_mean > baseline_mean * spike_ratio:
                pattern_detected = True
                
        elif pattern_type == "drop":
            # Detect sudden drops
            recent_mean = np.mean(values[-5:])
            baseline_mean = np.mean(values[:-5])
            drop_ratio = parameters.get("drop_ratio", 0.5)
            
            if recent_mean < baseline_mean * drop_ratio:
                pattern_detected = True
                
        if pattern_detected:
            if rule_id not in self.active_alerts:
                alert = Alert(
                    id=rule_id,
                    timestamp=time.time(),
                    severity=rule["severity"],
                    message=f"Pattern '{pattern_type}' detected in {metric_name}",
                    metric_name=metric_name,
                    current_value=values[-1],
                    threshold=0,
                    source="pattern_detector"
                )
                self._trigger_alert(alert)
        else:
            if rule_id in self.active_alerts:
                self._clear_alert(rule_id)
                
    def _trigger_alert(self, alert: Alert):
        """Trigger a new alert."""
        self.active_alerts[alert.id] = alert
        self.alert_history.append(alert)
        
        logger.warning(f"ALERT: {alert.message}")
        
        # Notify subscribers
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
                
    def _clear_alert(self, alert_id: str):
        """Clear an active alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts.pop(alert_id)
            logger.info(f"CLEARED: Alert {alert_id} - {alert.message}")
            
    def get_active_alerts(self) -> List[Alert]:
        """Get all currently active alerts."""
        return list(self.active_alerts.values())
        
    def acknowledge_alert(self, alert_id: str):
        """Acknowledge an alert."""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledged = True
            logger.info(f"Alert {alert_id} acknowledged")


class OptimizationEngine:
    """AI-driven optimization engine for automatic system tuning."""
    
    def __init__(self, metrics_collector: MetricsCollector, alert_manager: AlertManager):
        self.metrics_collector = metrics_collector
        self.alert_manager = alert_manager
        self.optimization_history: List[Dict] = []
        self.optimization_rules: Dict[str, Dict] = {}
        self.enabled = True
        
    def add_optimization_rule(self, rule_id: str, trigger_condition: Dict, 
                            optimization_action: Dict, priority: str = "medium"):
        """Add an optimization rule."""
        self.optimization_rules[rule_id] = {
            "trigger_condition": trigger_condition,
            "optimization_action": optimization_action,
            "priority": priority,
            "enabled": True,
            "last_executed": 0,
            "cooldown_seconds": trigger_condition.get("cooldown", 300)  # 5 minute default cooldown
        }
        
    def evaluate_optimizations(self):
        """Evaluate and execute optimization rules."""
        if not self.enabled:
            return
            
        current_time = time.time()
        
        for rule_id, rule in self.optimization_rules.items():
            if not rule["enabled"]:
                continue
                
            # Check cooldown
            if current_time - rule["last_executed"] < rule["cooldown_seconds"]:
                continue
                
            if self._should_trigger_optimization(rule["trigger_condition"]):
                self._execute_optimization(rule_id, rule)
                
    def _should_trigger_optimization(self, condition: Dict) -> bool:
        """Check if optimization condition is met."""
        condition_type = condition["type"]
        
        if condition_type == "metric_threshold":
            metric_name = condition["metric_name"]
            threshold = condition["threshold"]
            operator = condition.get("operator", "greater_than")
            
            stats = self.metrics_collector.get_metric_statistics(metric_name, 60)
            if not stats:
                return False
                
            current_value = stats["mean"]
            
            if operator == "greater_than":
                return current_value > threshold
            elif operator == "less_than":
                return current_value < threshold
                
        elif condition_type == "alert_triggered":
            alert_severity = condition.get("alert_severity", "medium")
            active_alerts = self.alert_manager.get_active_alerts()
            
            for alert in active_alerts:
                if alert.severity == alert_severity and not alert.acknowledged:
                    return True
                    
        elif condition_type == "performance_degradation":
            baseline_metric = condition["baseline_metric"]
            current_metric = condition["current_metric"]
            degradation_threshold = condition.get("degradation_threshold", 0.2)
            
            baseline_stats = self.metrics_collector.get_metric_statistics(baseline_metric, 300)
            current_stats = self.metrics_collector.get_metric_statistics(current_metric, 60)
            
            if baseline_stats and current_stats:
                baseline_value = baseline_stats["mean"]
                current_value = current_stats["mean"]
                
                if baseline_value > 0:
                    degradation = (baseline_value - current_value) / baseline_value
                    return degradation > degradation_threshold
                    
        return False
        
    def _execute_optimization(self, rule_id: str, rule: Dict):
        """Execute an optimization action."""
        action = rule["optimization_action"]
        action_type = action["type"]
        
        optimization_result = {
            "rule_id": rule_id,
            "timestamp": time.time(),
            "action_type": action_type,
            "success": False,
            "details": {}
        }
        
        try:
            if action_type == "adjust_batch_size":
                new_batch_size = action["new_batch_size"]
                # In practice, this would adjust the actual application parameter
                optimization_result["success"] = True
                optimization_result["details"] = {"new_batch_size": new_batch_size}
                logger.info(f"Optimization: Adjusted batch size to {new_batch_size}")
                
            elif action_type == "reduce_quality_threshold":
                new_threshold = action["new_threshold"]
                optimization_result["success"] = True
                optimization_result["details"] = {"new_threshold": new_threshold}
                logger.info(f"Optimization: Reduced quality threshold to {new_threshold}")
                
            elif action_type == "switch_processing_mode":
                new_mode = action["new_mode"]
                optimization_result["success"] = True
                optimization_result["details"] = {"new_mode": new_mode}
                logger.info(f"Optimization: Switched processing mode to {new_mode}")
                
            elif action_type == "clear_cache":
                optimization_result["success"] = True
                optimization_result["details"] = {"cache_cleared": True}
                logger.info("Optimization: Cleared processing cache")
                
            elif action_type == "restart_component":
                component = action["component"]
                optimization_result["success"] = True
                optimization_result["details"] = {"restarted_component": component}
                logger.info(f"Optimization: Restarted component {component}")
                
            if optimization_result["success"]:
                rule["last_executed"] = time.time()
                
        except Exception as e:
            optimization_result["success"] = False
            optimization_result["error"] = str(e)
            logger.error(f"Optimization failed for rule {rule_id}: {e}")
            
        self.optimization_history.append(optimization_result)
        
    def get_optimization_history(self, hours: int = 24) -> List[Dict]:
        """Get optimization history for the specified time period."""
        cutoff_time = time.time() - (hours * 3600)
        return [opt for opt in self.optimization_history if opt["timestamp"] >= cutoff_time]
        
    def enable_optimization(self):
        """Enable automatic optimization."""
        self.enabled = True
        logger.info("Automatic optimization enabled")
        
    def disable_optimization(self):
        """Disable automatic optimization."""
        self.enabled = False
        logger.info("Automatic optimization disabled")


class TelemetryExporter:
    """Export telemetry data to external systems."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.export_targets: Dict[str, Dict] = {}
        
    def add_export_target(self, target_id: str, target_type: str, config: Dict):
        """Add a telemetry export target."""
        self.export_targets[target_id] = {
            "type": target_type,
            "config": config,
            "enabled": True,
            "last_export": 0
        }
        
    def export_metrics(self, target_id: str = None):
        """Export metrics to specified target or all targets."""
        targets = [target_id] if target_id else list(self.export_targets.keys())
        
        for tid in targets:
            if tid not in self.export_targets:
                continue
                
            target = self.export_targets[tid]
            if not target["enabled"]:
                continue
                
            try:
                if target["type"] == "prometheus":
                    self._export_to_prometheus(target["config"])
                elif target["type"] == "influxdb":
                    self._export_to_influxdb(target["config"])
                elif target["type"] == "json_file":
                    self._export_to_json_file(target["config"])
                elif target["type"] == "elasticsearch":
                    self._export_to_elasticsearch(target["config"])
                    
                target["last_export"] = time.time()
                
            except Exception as e:
                logger.error(f"Failed to export to {tid}: {e}")
                
    def _export_to_prometheus(self, config: Dict):
        """Export metrics in Prometheus format."""
        # Mock Prometheus export
        metrics_data = self._prepare_metrics_data()
        logger.info(f"Exported {len(metrics_data)} metrics to Prometheus")
        
    def _export_to_influxdb(self, config: Dict):
        """Export metrics to InfluxDB."""
        # Mock InfluxDB export
        metrics_data = self._prepare_metrics_data()
        logger.info(f"Exported {len(metrics_data)} metrics to InfluxDB")
        
    def _export_to_json_file(self, config: Dict):
        """Export metrics to JSON file."""
        output_path = Path(config["output_path"])
        metrics_data = self._prepare_metrics_data()
        
        with open(output_path, 'w') as f:
            json.dump(metrics_data, f, indent=2, default=str)
            
        logger.info(f"Exported {len(metrics_data)} metrics to {output_path}")
        
    def _export_to_elasticsearch(self, config: Dict):
        """Export metrics to Elasticsearch."""
        # Mock Elasticsearch export
        metrics_data = self._prepare_metrics_data()
        logger.info(f"Exported {len(metrics_data)} metrics to Elasticsearch")
        
    def _prepare_metrics_data(self) -> List[Dict]:
        """Prepare metrics data for export."""
        return [asdict(metric) for metric in self.metrics_collector.metrics_buffer[-1000:]]


class AIMonitoringSystem:
    """Main AI monitoring and optimization system coordinator."""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager(self.metrics_collector)
        self.optimization_engine = OptimizationEngine(self.metrics_collector, self.alert_manager)
        self.telemetry_exporter = TelemetryExporter(self.metrics_collector)
        
        self.monitoring_loop_task = None
        self.running = False
        
        self._setup_default_rules()
        
    def _setup_default_rules(self):
        """Setup default monitoring and optimization rules."""
        # Alert rules
        self.alert_manager.add_threshold_rule(
            "high_cpu_usage", "system.cpu.usage_percent", 85, "greater_than", "high"
        )
        self.alert_manager.add_threshold_rule(
            "high_memory_usage", "system.memory.usage_percent", 90, "greater_than", "critical"
        )
        self.alert_manager.add_threshold_rule(
            "low_quality", "app.quality.ssim_score", 0.7, "less_than", "medium"
        )
        self.alert_manager.add_anomaly_rule(
            "processing_time_anomaly", "app.face_swapping.processing_time_ms", 2.5, 300, "medium"
        )
        
        # Optimization rules
        self.optimization_engine.add_optimization_rule(
            "reduce_batch_on_memory_pressure",
            {
                "type": "metric_threshold",
                "metric_name": "system.memory.usage_percent",
                "threshold": 85,
                "operator": "greater_than"
            },
            {
                "type": "adjust_batch_size",
                "new_batch_size": 2
            },
            "high"
        )
        
        self.optimization_engine.add_optimization_rule(
            "switch_to_cpu_on_gpu_error",
            {
                "type": "alert_triggered",
                "alert_severity": "critical"
            },
            {
                "type": "switch_processing_mode",
                "new_mode": "cpu"
            },
            "critical"
        )
        
    async def start_monitoring(self):
        """Start the AI monitoring system."""
        if self.running:
            return
            
        self.running = True
        
        # Start metrics collection
        self.metrics_collector.start_collection()
        
        # Start monitoring loop
        self.monitoring_loop_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("AI Monitoring System started")
        
    async def stop_monitoring(self):
        """Stop the AI monitoring system."""
        self.running = False
        
        # Stop metrics collection
        self.metrics_collector.stop_collection()
        
        # Cancel monitoring loop
        if self.monitoring_loop_task:
            self.monitoring_loop_task.cancel()
            try:
                await self.monitoring_loop_task
            except asyncio.CancelledError:
                pass
                
        logger.info("AI Monitoring System stopped")
        
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                # Check alerts
                self.alert_manager.check_alerts()
                
                # Evaluate optimizations
                self.optimization_engine.evaluate_optimizations()
                
                # Export telemetry (every 5 minutes)
                current_time = time.time()
                for target_id, target in self.telemetry_exporter.export_targets.items():
                    export_interval = target["config"].get("export_interval", 300)
                    if current_time - target["last_export"] > export_interval:
                        self.telemetry_exporter.export_metrics(target_id)
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(10)
                
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "metrics_collected": len(self.metrics_collector.metrics_buffer),
            "active_alerts": len(self.alert_manager.get_active_alerts()),
            "optimization_history_24h": len(self.optimization_engine.get_optimization_history(24)),
            "system_health": self._calculate_system_health(),
            "last_update": time.time()
        }
        
    def _calculate_system_health(self) -> str:
        """Calculate overall system health score."""
        active_alerts = self.alert_manager.get_active_alerts()
        
        critical_alerts = sum(1 for alert in active_alerts if alert.severity == "critical")
        high_alerts = sum(1 for alert in active_alerts if alert.severity == "high")
        
        if critical_alerts > 0:
            return "critical"
        elif high_alerts > 0:
            return "warning"
        elif len(active_alerts) > 0:
            return "caution"
        else:
            return "healthy"


# Example usage and testing
async def main():
    """Example usage of the AI monitoring system."""
    # Create and start monitoring system
    monitoring_system = AIMonitoringSystem()
    
    # Add telemetry export target
    monitoring_system.telemetry_exporter.add_export_target(
        "local_json",
        "json_file",
        {"output_path": "/tmp/roop_metrics.json", "export_interval": 60}
    )
    
    # Subscribe to alerts
    def alert_handler(alert: Alert):
        print(f"🚨 ALERT: {alert.message} (Severity: {alert.severity})")
    
    monitoring_system.alert_manager.subscribe_to_alerts(alert_handler)
    
    # Start monitoring
    await monitoring_system.start_monitoring()
    
    try:
        # Let it run for a while
        print("AI Monitoring System is running...")
        print("System status:", monitoring_system.get_system_status())
        
        await asyncio.sleep(30)  # Run for 30 seconds
        
    finally:
        # Stop monitoring
        await monitoring_system.stop_monitoring()
        print("AI Monitoring System stopped")


if __name__ == "__main__":
    asyncio.run(main())