"""
ATHENA F1 - Strategy Performance Analyzer
Continuously analyzes and improves strategy decision making through real-time learning.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict
import asyncio
from loguru import logger

from ..data.models import (
    StrategyRecommendation, StrategyOption, StrategyType,
    RaceState, DriverState
)


@dataclass
class StrategyOutcome:
    """Tracks the outcome of a strategic decision"""
    recommendation: StrategyRecommendation
    actual_position_change: int
    actual_time_gain: float
    success_achieved: bool
    confidence_accuracy: float
    external_factors: Dict[str, Any]
    execution_quality: float  # How well the strategy was executed


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for strategy analysis"""
    accuracy_rate: float  # % of successful recommendations
    average_confidence: float
    confidence_calibration: float  # How well confidence matches actual success
    position_gain_accuracy: float
    time_prediction_accuracy: float
    adaptation_speed: float  # How quickly system learns from mistakes
    risk_assessment_quality: float
    weather_prediction_accuracy: float
    safety_car_prediction_accuracy: float


class WorldClassPerformanceAnalyzer:
    """
    Advanced performance analyzer that continuously improves strategy decision making
    through sophisticated learning algorithms and real-time adaptation.
    """
    
    def __init__(self):
        self.strategy_outcomes: List[StrategyOutcome] = []
        self.performance_history: List[PerformanceMetrics] = []
        self.learning_rate = 0.05
        self.adaptation_threshold = 0.8  # Minimum accuracy before adjustments
        
        # Track performance by strategy type
        self.strategy_success_rates = defaultdict(list)
        self.strategy_time_predictions = defaultdict(list)
        
        # Environmental factors impact tracking
        self.weather_impact_model = {}
        self.track_specific_performance = {}
        self.driver_specific_adjustments = {}
        
        # Real-time learning coefficients
        self.confidence_calibration_factor = 1.0
        self.risk_adjustment_factor = 1.0
        self.time_prediction_bias = 0.0
        
    def add_strategy_outcome(
        self, 
        recommendation: StrategyRecommendation,
        actual_results: Dict[str, Any]
    ):
        """Record the outcome of a strategy recommendation"""
        
        # Calculate actual performance metrics
        actual_position_change = actual_results.get('position_change', 0)
        actual_time_gain = actual_results.get('time_gain', 0.0)
        execution_quality = actual_results.get('execution_quality', 1.0)
        
        # Determine success based on multiple criteria
        position_success = actual_position_change >= 0  # Didn't lose positions
        time_success = actual_time_gain >= (recommendation.primary_option.estimated_time_gain * 0.5)
        overall_success = position_success and time_success
        
        # Calculate confidence accuracy
        predicted_success_prob = recommendation.primary_option.probability_success
        confidence_accuracy = 1.0 - abs(predicted_success_prob - (1.0 if overall_success else 0.0))
        
        outcome = StrategyOutcome(
            recommendation=recommendation,
            actual_position_change=actual_position_change,
            actual_time_gain=actual_time_gain,
            success_achieved=overall_success,
            confidence_accuracy=confidence_accuracy,
            external_factors=actual_results.get('external_factors', {}),
            execution_quality=execution_quality
        )
        
        self.strategy_outcomes.append(outcome)
        
        # Update strategy-specific tracking
        strategy_type = recommendation.primary_option.strategy_type
        self.strategy_success_rates[strategy_type].append(overall_success)
        
        time_error = abs(actual_time_gain - recommendation.primary_option.estimated_time_gain)
        self.strategy_time_predictions[strategy_type].append(time_error)
        
        logger.info(f"Recorded strategy outcome: {strategy_type.value} - "
                   f"Success: {overall_success}, Position Δ: {actual_position_change}, "
                   f"Time Δ: {actual_time_gain:.1f}s")
        
        # Trigger learning if we have enough data
        if len(self.strategy_outcomes) % 10 == 0:
            self._update_learning_parameters()
    
    def _update_learning_parameters(self):
        """Update learning parameters based on recent performance"""
        if len(self.strategy_outcomes) < 10:
            return
            
        recent_outcomes = self.strategy_outcomes[-20:]  # Last 20 decisions
        
        # Calculate recent success rate
        recent_success_rate = sum(1 for o in recent_outcomes if o.success_achieved) / len(recent_outcomes)
        
        # Update confidence calibration
        confidence_errors = [o.confidence_accuracy for o in recent_outcomes]
        avg_confidence_error = np.mean(confidence_errors)
        
        # Adjust calibration factor
        if avg_confidence_error < 0.7:  # Poor calibration
            self.confidence_calibration_factor *= (1 - self.learning_rate)
        else:
            self.confidence_calibration_factor *= (1 + self.learning_rate * 0.5)
        
        # Update time prediction bias
        time_errors = [o.actual_time_gain - o.recommendation.primary_option.estimated_time_gain 
                      for o in recent_outcomes]
        self.time_prediction_bias = np.mean(time_errors) * self.learning_rate
        
        # Adjust risk factor based on recent performance
        if recent_success_rate < self.adaptation_threshold:
            # Being too aggressive, increase caution
            self.risk_adjustment_factor *= (1 + self.learning_rate)
        elif recent_success_rate > 0.9:
            # Being too conservative, can be more aggressive
            self.risk_adjustment_factor *= (1 - self.learning_rate * 0.5)
        
        logger.debug(f"Learning parameters updated - Success rate: {recent_success_rate:.2f}, "
                    f"Confidence calibration: {self.confidence_calibration_factor:.3f}, "
                    f"Time bias: {self.time_prediction_bias:.2f}")
    
    def get_adjusted_confidence(self, base_confidence: float, strategy_type: StrategyType) -> float:
        """Get confidence adjusted based on historical performance"""
        
        # Apply global calibration factor
        adjusted = base_confidence * self.confidence_calibration_factor
        
        # Apply strategy-specific adjustments
        if strategy_type in self.strategy_success_rates:
            recent_successes = self.strategy_success_rates[strategy_type][-10:]  # Last 10
            if recent_successes:
                historical_rate = np.mean(recent_successes)
                # Adjust based on historical performance vs assumed performance
                strategy_adjustment = historical_rate / max(0.1, base_confidence)
                adjusted *= strategy_adjustment ** 0.3  # Moderate adjustment
        
        return max(0.1, min(1.0, adjusted))
    
    def get_adjusted_time_prediction(
        self, 
        base_prediction: float, 
        strategy_type: StrategyType
    ) -> float:
        """Get time prediction adjusted for known biases"""
        
        adjusted = base_prediction + self.time_prediction_bias
        
        # Apply strategy-specific time prediction adjustments
        if strategy_type in self.strategy_time_predictions:
            recent_errors = self.strategy_time_predictions[strategy_type][-10:]
            if recent_errors:
                avg_error = np.mean(recent_errors)
                # Correct for systematic errors
                if avg_error > 2.0:  # Consistently over-predicting benefits
                    adjusted *= 0.9
                elif avg_error > 5.0:  # Very poor predictions
                    adjusted *= 0.8
        
        return adjusted
    
    def get_risk_adjusted_options(
        self, 
        strategy_options: List[StrategyOption]
    ) -> List[StrategyOption]:
        """Apply risk adjustments to strategy options based on learning"""
        
        adjusted_options = []
        
        for option in strategy_options:
            adjusted_option = option
            
            # Adjust probability based on historical performance
            adjusted_prob = self.get_adjusted_confidence(
                option.probability_success, 
                option.strategy_type
            )
            
            # Adjust time prediction
            adjusted_time = self.get_adjusted_time_prediction(
                option.estimated_time_gain,
                option.strategy_type
            )
            
            # Apply risk factor adjustments
            if option.risk_level == "high":
                adjusted_prob *= (2.0 - self.risk_adjustment_factor)  # Reduce high-risk confidence
            elif option.risk_level == "low":
                adjusted_prob *= self.risk_adjustment_factor  # Boost low-risk confidence
            
            # Create adjusted option
            from dataclasses import replace
            adjusted_option = replace(
                option,
                probability_success=max(0.1, min(1.0, adjusted_prob)),
                estimated_time_gain=adjusted_time
            )
            
            adjusted_options.append(adjusted_option)
        
        return adjusted_options
    
    def analyze_performance_trends(self) -> PerformanceMetrics:
        """Analyze overall performance trends and return comprehensive metrics"""
        
        if not self.strategy_outcomes:
            return PerformanceMetrics(
                accuracy_rate=0.5,
                average_confidence=0.5,
                confidence_calibration=0.5,
                position_gain_accuracy=0.5,
                time_prediction_accuracy=0.5,
                adaptation_speed=0.5,
                risk_assessment_quality=0.5,
                weather_prediction_accuracy=0.5,
                safety_car_prediction_accuracy=0.5
            )
        
        # Calculate comprehensive metrics
        recent_outcomes = self.strategy_outcomes[-50:]  # Last 50 for trend analysis
        
        # Accuracy rate
        accuracy_rate = sum(1 for o in recent_outcomes if o.success_achieved) / len(recent_outcomes)
        
        # Average confidence and calibration
        confidences = [o.recommendation.confidence_score for o in recent_outcomes]
        average_confidence = np.mean(confidences)
        
        confidence_errors = [o.confidence_accuracy for o in recent_outcomes]
        confidence_calibration = np.mean(confidence_errors)
        
        # Position prediction accuracy
        position_predictions = []
        for outcome in recent_outcomes:
            predicted_change = (outcome.recommendation.primary_option.estimated_outcome_position - 
                              outcome.recommendation.primary_option.estimated_outcome_position)  # Simplified
            actual_change = outcome.actual_position_change
            error = abs(predicted_change - actual_change)
            position_predictions.append(1.0 / (1.0 + error))  # Inverse error scoring
        
        position_gain_accuracy = np.mean(position_predictions) if position_predictions else 0.5
        
        # Time prediction accuracy
        time_errors = []
        for outcome in recent_outcomes:
            predicted_time = outcome.recommendation.primary_option.estimated_time_gain
            actual_time = outcome.actual_time_gain
            relative_error = abs(predicted_time - actual_time) / max(1.0, abs(predicted_time))
            time_errors.append(1.0 / (1.0 + relative_error))
        
        time_prediction_accuracy = np.mean(time_errors) if time_errors else 0.5
        
        # Adaptation speed (how quickly performance improves)
        if len(self.strategy_outcomes) >= 20:
            early_success = sum(1 for o in self.strategy_outcomes[:10] if o.success_achieved) / 10
            late_success = sum(1 for o in self.strategy_outcomes[-10:] if o.success_achieved) / 10
            adaptation_speed = max(0.0, late_success - early_success) + 0.5
        else:
            adaptation_speed = 0.5
        
        # Risk assessment quality
        high_risk_outcomes = [o for o in recent_outcomes 
                            if o.recommendation.primary_option.risk_level == "high"]
        if high_risk_outcomes:
            high_risk_success = sum(1 for o in high_risk_outcomes if o.success_achieved)
            risk_assessment_quality = 1.0 - (high_risk_success / len(high_risk_outcomes))
        else:
            risk_assessment_quality = 0.7
        
        # Weather and safety car prediction accuracy (simplified)
        weather_accuracy = 0.8  # Would calculate based on actual weather events
        safety_car_accuracy = 0.75  # Would calculate based on actual SC deployments
        
        metrics = PerformanceMetrics(
            accuracy_rate=accuracy_rate,
            average_confidence=average_confidence,
            confidence_calibration=confidence_calibration,
            position_gain_accuracy=position_gain_accuracy,
            time_prediction_accuracy=time_prediction_accuracy,
            adaptation_speed=adaptation_speed,
            risk_assessment_quality=risk_assessment_quality,
            weather_prediction_accuracy=weather_accuracy,
            safety_car_prediction_accuracy=safety_car_accuracy
        )
        
        self.performance_history.append(metrics)
        
        logger.info(f"Performance Analysis - Accuracy: {accuracy_rate:.2f}, "
                   f"Confidence: {average_confidence:.2f}, "
                   f"Adaptation: {adaptation_speed:.2f}")
        
        return metrics
    
    def get_strategy_insights(self) -> Dict[str, Any]:
        """Get insights about strategy performance for different scenarios"""
        
        insights = {
            'best_performing_strategies': {},
            'worst_performing_strategies': {},
            'weather_adaptations': {},
            'track_specific_insights': {},
            'learning_trajectory': []
        }
        
        # Analyze strategy type performance
        for strategy_type, successes in self.strategy_success_rates.items():
            if successes:
                success_rate = np.mean(successes)
                insights['best_performing_strategies'][strategy_type.value] = success_rate
        
        # Sort strategies by performance
        sorted_strategies = sorted(
            insights['best_performing_strategies'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        insights['best_performing_strategies'] = dict(sorted_strategies[:3])
        insights['worst_performing_strategies'] = dict(sorted_strategies[-2:])
        
        # Learning trajectory
        if len(self.performance_history) >= 3:
            recent_metrics = self.performance_history[-3:]
            accuracy_trend = [m.accuracy_rate for m in recent_metrics]
            insights['learning_trajectory'] = {
                'accuracy_trend': accuracy_trend,
                'improving': accuracy_trend[-1] > accuracy_trend[0],
                'rate_of_improvement': (accuracy_trend[-1] - accuracy_trend[0]) / len(accuracy_trend)
            }
        
        return insights
    
    def should_trigger_retraining(self) -> bool:
        """Determine if the system should trigger more intensive retraining"""
        
        if len(self.strategy_outcomes) < 20:
            return False
            
        recent_performance = self.analyze_performance_trends()
        
        # Trigger retraining if performance is declining
        if recent_performance.accuracy_rate < 0.6:
            logger.warning("Strategy accuracy below threshold - triggering retraining")
            return True
            
        # Trigger if confidence is poorly calibrated
        if recent_performance.confidence_calibration < 0.6:
            logger.warning("Poor confidence calibration - triggering retraining")
            return True
            
        return False
    
    def export_performance_data(self) -> Dict[str, Any]:
        """Export performance data for external analysis or archiving"""
        
        return {
            'total_strategies_analyzed': len(self.strategy_outcomes),
            'current_performance_metrics': self.analyze_performance_trends().__dict__,
            'learning_parameters': {
                'confidence_calibration_factor': self.confidence_calibration_factor,
                'risk_adjustment_factor': self.risk_adjustment_factor,
                'time_prediction_bias': self.time_prediction_bias
            },
            'strategy_success_rates': {
                str(k): list(v) for k, v in self.strategy_success_rates.items()
            },
            'insights': self.get_strategy_insights(),
            'export_timestamp': datetime.now().isoformat()
        }
