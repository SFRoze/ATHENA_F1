"""
ATHENA F1 - Tire Degradation Algorithm
Physics-based tire performance prediction model.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from ..data.models import TireCompound, WeatherCondition


@dataclass
class TireParameters:
    """Parameters for tire compound performance modeling"""
    base_grip: float  # Base grip level (0-1.0)
    degradation_rate: float  # Degradation per lap (0-1.0)
    thermal_sensitivity: float  # Temperature impact factor
    optimal_temp_range: Tuple[float, float]  # Optimal track temp range
    life_expectancy: int  # Expected life in laps


class TireDegradationModel:
    """
    Advanced tire degradation model that predicts tire performance
    based on compound, temperature, load, and driving style.
    """
    
    def __init__(self):
        self.compound_parameters = {
            TireCompound.SOFT: TireParameters(
                base_grip=1.0,
                degradation_rate=0.025,  # 2.5% per lap
                thermal_sensitivity=0.8,
                optimal_temp_range=(45.0, 55.0),
                life_expectancy=25
            ),
            TireCompound.MEDIUM: TireParameters(
                base_grip=0.92,
                degradation_rate=0.015,  # 1.5% per lap
                thermal_sensitivity=0.6,
                optimal_temp_range=(50.0, 65.0),
                life_expectancy=35
            ),
            TireCompound.HARD: TireParameters(
                base_grip=0.85,
                degradation_rate=0.008,  # 0.8% per lap
                thermal_sensitivity=0.4,
                optimal_temp_range=(55.0, 70.0),
                life_expectancy=50
            ),
            TireCompound.INTERMEDIATE: TireParameters(
                base_grip=0.78,
                degradation_rate=0.020,
                thermal_sensitivity=0.7,
                optimal_temp_range=(30.0, 45.0),
                life_expectancy=30
            ),
            TireCompound.WET: TireParameters(
                base_grip=0.75,
                degradation_rate=0.018,
                thermal_sensitivity=0.5,
                optimal_temp_range=(25.0, 40.0),
                life_expectancy=35
            )
        }
        
    def calculate_current_performance(
        self,
        compound: TireCompound,
        age_laps: int,
        track_temperature: float,
        fuel_load: float = 100.0,  # kg
        driving_aggression: float = 1.0  # 0.5-1.5 multiplier
    ) -> Dict[str, float]:
        """
        Calculate current tire performance metrics
        
        Returns:
            Dict with 'grip_level', 'degradation_percent', 'remaining_life'
        """
        params = self.compound_parameters[compound]
        
        # Base degradation calculation
        base_degradation = age_laps * params.degradation_rate * driving_aggression
        
        # Temperature impact on degradation
        temp_factor = self._calculate_temperature_factor(
            track_temperature, params.optimal_temp_range, params.thermal_sensitivity
        )
        
        # Fuel load impact (heavier = more degradation)
        fuel_factor = 1.0 + (fuel_load - 50.0) * 0.002  # 0.2% per 10kg above 50kg
        
        # Total degradation
        total_degradation = base_degradation * temp_factor * fuel_factor
        degradation_percent = min(100.0, total_degradation * 100)
        
        # Current grip level
        grip_level = max(0.1, params.base_grip * (1.0 - total_degradation))
        
        # Remaining tire life
        if params.degradation_rate > 0:
            remaining_life = max(0, params.life_expectancy - age_laps)
        else:
            remaining_life = params.life_expectancy
            
        return {
            'grip_level': grip_level,
            'degradation_percent': degradation_percent,
            'remaining_life': remaining_life,
            'temp_factor': temp_factor,
            'fuel_factor': fuel_factor
        }
        
    def predict_performance_degradation(
        self,
        compound: TireCompound,
        current_age: int,
        future_laps: int,
        track_temperature: float,
        fuel_trajectory: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Predict tire performance degradation over future laps
        
        Returns:
            Array of grip levels for each future lap
        """
        params = self.compound_parameters[compound]
        
        if fuel_trajectory is None:
            # Assume linear fuel consumption (2kg per lap)
            fuel_trajectory = np.linspace(100.0, 50.0, future_laps)
            
        grip_levels = []
        
        for lap in range(future_laps):
            total_age = current_age + lap
            fuel_load = fuel_trajectory[lap] if lap < len(fuel_trajectory) else 50.0
            
            performance = self.calculate_current_performance(
                compound, total_age, track_temperature, fuel_load
            )
            grip_levels.append(performance['grip_level'])
            
        return np.array(grip_levels)
        
    def calculate_optimal_pit_window(
        self,
        compound: TireCompound,
        current_age: int,
        track_temperature: float,
        remaining_race_laps: int,
        target_grip_threshold: float = 0.7
    ) -> Tuple[int, int]:
        """
        Calculate optimal pit window based on tire degradation
        
        Returns:
            (earliest_lap, latest_lap) for pit window
        """
        params = self.compound_parameters[compound]
        
        # Find when tire drops below threshold
        critical_lap = None
        for future_lap in range(1, remaining_race_laps + 1):
            performance = self.calculate_current_performance(
                compound, current_age + future_lap, track_temperature
            )
            if performance['grip_level'] < target_grip_threshold:
                critical_lap = future_lap
                break
                
        if critical_lap is None:
            # Tire can last the entire race
            earliest = max(1, remaining_race_laps - 10)
            latest = remaining_race_laps
        else:
            # Pit before tire becomes critical
            earliest = max(1, critical_lap - 5)
            latest = critical_lap - 1
            
        return earliest, latest
        
    def compare_compounds(
        self,
        compounds: list,
        stint_length: int,
        track_temperature: float,
        fuel_load: float = 75.0
    ) -> Dict[TireCompound, Dict[str, float]]:
        """
        Compare performance of different tire compounds for a given stint
        """
        comparison = {}
        
        for compound in compounds:
            if compound not in self.compound_parameters:
                continue
                
            # Calculate average performance over stint
            total_grip = 0.0
            for lap in range(stint_length):
                performance = self.calculate_current_performance(
                    compound, lap, track_temperature, fuel_load
                )
                total_grip += performance['grip_level']
                
            avg_grip = total_grip / stint_length
            
            # Calculate end-of-stint performance
            end_performance = self.calculate_current_performance(
                compound, stint_length, track_temperature, fuel_load
            )
            
            comparison[compound] = {
                'average_grip': avg_grip,
                'end_grip': end_performance['grip_level'],
                'degradation': end_performance['degradation_percent'],
                'recommended': avg_grip > 0.75 and end_performance['grip_level'] > 0.6
            }
            
        return comparison
        
    def _calculate_temperature_factor(
        self,
        current_temp: float,
        optimal_range: Tuple[float, float],
        sensitivity: float
    ) -> float:
        """Calculate temperature impact on tire degradation"""
        optimal_min, optimal_max = optimal_range
        
        if optimal_min <= current_temp <= optimal_max:
            # In optimal range
            return 1.0
        elif current_temp < optimal_min:
            # Too cold
            temp_diff = optimal_min - current_temp
            return 1.0 + (temp_diff * sensitivity * 0.02)  # 2% per degree
        else:
            # Too hot
            temp_diff = current_temp - optimal_max
            return 1.0 + (temp_diff * sensitivity * 0.03)  # 3% per degree
            
    def calculate_lap_time_impact(
        self,
        compound: TireCompound,
        age_laps: int,
        track_temperature: float,
        baseline_laptime: float,
        fuel_load: float = 75.0
    ) -> float:
        """
        Calculate lap time impact due to tire degradation
        
        Returns:
            Predicted lap time (seconds)
        """
        performance = self.calculate_current_performance(
            compound, age_laps, track_temperature, fuel_load
        )
        
        # Grip loss translates to lap time increase
        grip_loss = 1.0 - performance['grip_level']
        lap_time_penalty = grip_loss * 3.0  # ~3 seconds max penalty
        
        # Fuel effect (lighter = faster)
        fuel_benefit = (100.0 - fuel_load) * 0.03  # 0.03s per kg saved
        
        return baseline_laptime + lap_time_penalty - fuel_benefit
