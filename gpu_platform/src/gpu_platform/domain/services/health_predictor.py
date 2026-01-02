"""Predictive health analysis for GPU failure prediction.

Analyzes metric trends to predict GPU failures before they occur,
enabling proactive maintenance and workload migration.

References:
    - design.md Section 6 (Failure Modes & Recovery)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

from gpu_platform.domain.entities.gpu_device import GPUDevice

logger = logging.getLogger(__name__)


@dataclass
class HealthTrend:
    """Health metric trend analysis."""
    gpu_id: str
    metric_name: str
    current_value: float
    trend_direction: str  # "increasing", "decreasing", "stable"
    trend_rate: float  # Change per minute
    predicted_critical_time: Optional[float] = None  # Seconds until critical threshold
    confidence: float = 0.0  # 0.0-1.0


class PredictiveHealthAnalyzer:
    """Analyze GPU health trends and predict failures."""
    
    def __init__(self, window_size: int = 60, update_interval: int = 5):
        """Initialize predictive health analyzer.
        
        Args:
            window_size: Duration in seconds to track metrics.
            update_interval: Seconds between samples.
        """
        self.window_size = window_size
        self.update_interval = update_interval
        self._metric_history: dict[str, list[tuple[float, float]]] = {}  # gpu_id -> [(timestamp, value), ...]
        self._health_predictions: dict[str, list[HealthTrend]] = {}
        self._failure_risk: dict[str, float] = {}  # GPU ID -> risk score (0-1)
    
    def update_metrics(self, device: GPUDevice) -> None:
        """Update metrics for a GPU.
        
        Args:
            device: GPU device with current metrics.
        """
        if device.health is None:
            return
        
        gpu_id = str(device.specs.gpu_id)
        now = time.time()
        
        # Track temperature trend
        temp_key = f"{gpu_id}_temp"
        if temp_key not in self._metric_history:
            self._metric_history[temp_key] = []
        
        self._metric_history[temp_key].append((now, device.health.temperature_c))
        
        # Track power trend
        power_key = f"{gpu_id}_power"
        if power_key not in self._metric_history:
            self._metric_history[power_key] = []
        
        self._metric_history[power_key].append((now, device.health.power_w))
        
        # Track ECC errors
        ecc_key = f"{gpu_id}_ecc"
        if ecc_key not in self._metric_history:
            self._metric_history[ecc_key] = []
        
        total_ecc = (
            device.health.ecc_errors_correctable + 
            device.health.ecc_errors_uncorrectable
        )
        self._metric_history[ecc_key].append((now, total_ecc))
        
        # Clean up old samples
        self._cleanup_old_samples(now)
        
        # Analyze trends
        self._analyze_trends(gpu_id, now)
    
    def _cleanup_old_samples(self, now: float) -> None:
        """Remove samples older than window_size.
        
        Args:
            now: Current timestamp.
        """
        for key in self._metric_history:
            # Keep only recent samples
            self._metric_history[key] = [
                (ts, val) for ts, val in self._metric_history[key]
                if now - ts < self.window_size
            ]
    
    def _analyze_trends(self, gpu_id: str, now: float) -> None:
        """Analyze metric trends for a GPU.
        
        Args:
            gpu_id: GPU identifier.
            now: Current timestamp.
        """
        if gpu_id not in self._health_predictions:
            self._health_predictions[gpu_id] = []
        else:
            self._health_predictions[gpu_id] = []
        
        trends = []
        
        # Analyze temperature trend
        temp_key = f"{gpu_id}_temp"
        if temp_key in self._metric_history and len(self._metric_history[temp_key]) >= 2:
            trend = self._calculate_trend(
                gpu_id,
                "temperature_c",
                self._metric_history[temp_key],
                critical_value=90.0,
            )
            if trend:
                trends.append(trend)
        
        # Analyze power trend
        power_key = f"{gpu_id}_power"
        if power_key in self._metric_history and len(self._metric_history[power_key]) >= 2:
            trend = self._calculate_trend(
                gpu_id,
                "power_w",
                self._metric_history[power_key],
                critical_value=500.0,
            )
            if trend:
                trends.append(trend)
        
        # Analyze ECC trend
        ecc_key = f"{gpu_id}_ecc"
        if ecc_key in self._metric_history and len(self._metric_history[ecc_key]) >= 2:
            trend = self._calculate_trend(
                gpu_id,
                "ecc_errors",
                self._metric_history[ecc_key],
                critical_value=50.0,
            )
            if trend:
                trends.append(trend)
        
        self._health_predictions[gpu_id] = trends
        
        # Calculate failure risk
        self._calculate_failure_risk(gpu_id, trends)
    
    def _calculate_trend(
        self,
        gpu_id: str,
        metric_name: str,
        samples: list[tuple[float, float]],
        critical_value: float,
    ) -> Optional[HealthTrend]:
        """Calculate trend for a metric.
        
        Args:
            gpu_id: GPU identifier.
            metric_name: Name of metric.
            samples: List of (timestamp, value) tuples.
            critical_value: Value at which GPU becomes critical.
            
        Returns:
            HealthTrend or None if insufficient data.
        """
        if len(samples) < 2:
            return None
        
        # Simple linear regression
        timestamps = [ts for ts, _ in samples]
        values = [val for _, val in samples]
        
        # Calculate average change per minute
        time_span = max(timestamps) - min(timestamps)
        if time_span < 1:
            time_span = 1
        
        value_change = values[-1] - values[0]
        trend_rate = (value_change / time_span) * 60  # Per minute
        
        # Determine direction
        if abs(trend_rate) < 0.1:
            direction = "stable"
        elif trend_rate > 0:
            direction = "increasing"
        else:
            direction = "decreasing"
        
        current_value = values[-1]
        predicted_time = None
        confidence = 0.0
        
        # Predict time to critical if increasing
        if direction == "increasing" and trend_rate > 0:
            remaining = critical_value - current_value
            if remaining > 0:
                predicted_time = remaining / (trend_rate / 60)
                # Confidence based on trend consistency
                confidence = min(0.95, abs(trend_rate) / 10.0)
        
        return HealthTrend(
            gpu_id=gpu_id,
            metric_name=metric_name,
            current_value=current_value,
            trend_direction=direction,
            trend_rate=trend_rate,
            predicted_critical_time=predicted_time,
            confidence=confidence,
        )
    
    def _calculate_failure_risk(self, gpu_id: str, trends: list[HealthTrend]) -> None:
        """Calculate failure risk score for GPU.
        
        Args:
            gpu_id: GPU identifier.
            trends: List of health trends.
        """
        risk_score = 0.0
        
        for trend in trends:
            # Risk increases if metric is trending toward critical value
            if trend.predicted_critical_time is not None:
                time_to_critical = trend.predicted_critical_time
                
                # Higher risk if predicted critical is soon
                if time_to_critical < 300:  # Within 5 minutes
                    risk_score += 0.9 * trend.confidence
                elif time_to_critical < 600:  # Within 10 minutes
                    risk_score += 0.6 * trend.confidence
                elif time_to_critical < 1800:  # Within 30 minutes
                    risk_score += 0.3 * trend.confidence
        
        # Cap risk score at 1.0
        self._failure_risk[gpu_id] = min(1.0, risk_score)
        
        if self._failure_risk[gpu_id] > 0.5:
            logger.warning(
                f"GPU {gpu_id} has elevated failure risk: {self._failure_risk[gpu_id]:.2f}"
            )
    
    def get_trends(self, gpu_id: str) -> list[HealthTrend]:
        """Get current health trends for a GPU.
        
        Args:
            gpu_id: GPU identifier.
            
        Returns:
            List of health trends.
        """
        return self._health_predictions.get(gpu_id, [])
    
    def get_failure_risk(self, gpu_id: str) -> float:
        """Get current failure risk score for a GPU.
        
        Args:
            gpu_id: GPU identifier.
            
        Returns:
            Risk score (0.0-1.0). >0.7 indicates high risk.
        """
        return self._failure_risk.get(gpu_id, 0.0)
    
    def should_preempt_job(self, gpu_id: str) -> bool:
        """Determine if job on GPU should be preempted based on failure risk.
        
        Args:
            gpu_id: GPU identifier.
            
        Returns:
            True if failure risk is very high (>0.8).
        """
        return self.get_failure_risk(gpu_id) > 0.8
    
    def get_maintenance_recommendation(self, gpu_id: str) -> Optional[str]:
        """Get maintenance recommendation for a GPU.
        
        Args:
            gpu_id: GPU identifier.
            
        Returns:
            Maintenance recommendation or None.
        """
        risk = self.get_failure_risk(gpu_id)
        trends = self.get_trends(gpu_id)
        
        if risk > 0.9:
            return f"CRITICAL: Quarantine GPU immediately (risk={risk:.2f})"
        
        for trend in trends:
            if trend.predicted_critical_time and trend.predicted_critical_time < 300:
                return (
                    f"Migrate jobs from GPU {gpu_id}: {trend.metric_name} "
                    f"will be critical in {trend.predicted_critical_time:.0f}s"
                )
        
        if risk > 0.5:
            return f"Monitor GPU {gpu_id} closely (risk={risk:.2f})"
        
        return None
