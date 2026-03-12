"""
ATHENA F1 - Pit Strategy Calculator
Calculates optimal pit stop timing and strategy options.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from ..data.models import TireCompound, DriverState, TrackState


@dataclass
class PitStopOutcome:
    """Predicted outcome of a pit stop"""
    new_position: int
    time_lost: float  # seconds
    time_gained_next_stint: float  # seconds
    net_benefit: float  # seconds
    success_probability: float


class PitStrategyCalculator:
    """
    Advanced pit strategy calculator that evaluates pit stop timing
    and predicts outcomes based on track position and tire strategy.
    """
    
    def __init__(self):
        # Track-specific pit lane times (simplified)
        self.pit_lane_times = {
            "monaco": 25.0,
            "silverstone": 22.0,
            "spa": 24.0,
            "monza": 20.0,
            "default": 23.0
        }
        
    def calculate_undercut_opportunity(
        self,
        attacking_driver: DriverState,
        target_driver: DriverState,
        track_temp: float,
        pit_lane_time: float = 23.0
    ) -> PitStopOutcome:
        """
        Calculate the success probability and outcome of an undercut attempt
        """
        # Gap analysis
        current_gap = attacking_driver.gap_to_ahead
        
        # Tire advantage calculation
        target_tire_age = target_driver.tire_state.age_laps
        target_degradation = target_driver.tire_state.degradation_percent
        
        # Fresh tire advantage (simplified model)
        tire_advantage_per_lap = self._calculate_tire_advantage(
            target_degradation, track_temp
        )
        
        # Calculate time needed to close gap
        laps_to_close = current_gap / tire_advantage_per_lap if tire_advantage_per_lap > 0 else float('inf')
        
        # Account for pit lane time loss
        effective_gap = current_gap + pit_lane_time
        effective_laps_to_close = effective_gap / tire_advantage_per_lap if tire_advantage_per_lap > 0 else float('inf')
        
        # Success probability based on gap and tire advantage
        if effective_laps_to_close <= 5:  # Very likely
            success_prob = 0.85
        elif effective_laps_to_close <= 10:  # Possible
            success_prob = 0.6
        elif effective_laps_to_close <= 15:  # Unlikely
            success_prob = 0.3
        else:  # Very unlikely
            success_prob = 0.1
            
        # Predict new position (simplified)
        if success_prob > 0.5:
            new_position = max(1, attacking_driver.current_position - 1)
        else:
            new_position = min(20, attacking_driver.current_position + 1)
            
        return PitStopOutcome(
            new_position=new_position,
            time_lost=pit_lane_time,
            time_gained_next_stint=tire_advantage_per_lap * 10,  # 10 lap benefit
            net_benefit=tire_advantage_per_lap * 10 - pit_lane_time,
            success_probability=success_prob
        )
        
    def calculate_overcut_opportunity(
        self,
        staying_driver: DriverState,
        pitting_drivers: List[DriverState],
        track_temp: float,
        extended_stint_laps: int = 8
    ) -> PitStopOutcome:
        """
        Calculate the success probability and outcome of an overcut strategy
        """
        # Current tire state
        current_degradation = staying_driver.tire_state.degradation_percent
        current_age = staying_driver.tire_state.age_laps
        
        # Predict tire degradation over extended stint
        degradation_rate = 2.0  # Simplified: 2% per lap
        future_degradation = min(100.0, current_degradation + (degradation_rate * extended_stint_laps))
        
        # Calculate lap time impact
        degradation_penalty = (future_degradation - current_degradation) * 0.02  # 0.02s per %
        
        # Track position benefit (clear air)
        clear_air_benefit = 0.3  # 0.3s per lap in clear air
        
        # Net benefit per lap during extended stint
        net_benefit_per_lap = clear_air_benefit - degradation_penalty
        total_benefit = net_benefit_per_lap * extended_stint_laps
        
        # Success probability based on tire degradation
        if future_degradation < 70:
            success_prob = 0.8
        elif future_degradation < 85:
            success_prob = 0.6
        else:
            success_prob = 0.3
            
        # Position prediction
        if success_prob > 0.6 and len(pitting_drivers) >= 2:
            new_position = max(1, staying_driver.current_position - len(pitting_drivers) + 1)
        else:
            new_position = staying_driver.current_position
            
        return PitStopOutcome(
            new_position=new_position,
            time_lost=0.0,  # No pit stop
            time_gained_next_stint=total_benefit,
            net_benefit=total_benefit,
            success_probability=success_prob
        )
        
    def calculate_optimal_pit_window(
        self,
        driver: DriverState,
        track_state: TrackState,
        competitors: List[DriverState]
    ) -> Tuple[int, int, Dict[str, any]]:
        """
        Calculate the optimal pit window for a driver considering all factors
        
        Returns:
            (earliest_optimal_lap, latest_optimal_lap, analysis_details)
        """
        current_lap = track_state.current_lap
        remaining_laps = track_state.total_laps - current_lap
        
        # Tire-based window
        tire_degradation = driver.tire_state.degradation_percent
        tire_age = driver.tire_state.age_laps
        
        if tire_degradation > 80:
            tire_window = (current_lap, current_lap + 2)  # Urgent
        elif tire_degradation > 60:
            tire_window = (current_lap + 1, current_lap + 5)  # Soon
        else:
            tire_window = (current_lap + 5, current_lap + 15)  # Flexible
            
        # Strategy-based window (considering competitors)
        undercut_threats = self._identify_undercut_threats(driver, competitors)
        overcut_opportunities = self._identify_overcut_opportunities(driver, competitors)
        
        # Safety car probability window
        safety_car_risk = self._calculate_safety_car_probability(
            current_lap, track_state.total_laps, track_state.track_name
        )
        
        # Combine factors to determine optimal window
        earliest = max(tire_window[0], current_lap + 1)
        latest = min(tire_window[1], remaining_laps - 5)  # Don't pit too late
        
        # Adjust for strategic factors
        if undercut_threats:
            earliest = max(1, earliest - 2)  # Pit earlier to avoid undercuts
        if overcut_opportunities and tire_degradation < 70:
            latest = min(remaining_laps, latest + 5)  # Extend for overcut
            
        analysis = {
            'tire_urgency': tire_degradation / 100.0,
            'undercut_threats': len(undercut_threats),
            'overcut_opportunities': len(overcut_opportunities),
            'safety_car_risk': safety_car_risk,
            'recommended_compounds': self._recommend_tire_compounds(
                remaining_laps, track_state.weather, track_state.track_temperature
            )
        }
        
        return earliest, latest, analysis
        
    def simulate_pit_stop_scenarios(
        self,
        driver: DriverState,
        track_state: TrackState,
        competitors: List[DriverState],
        scenarios: List[int]  # List of pit lap options
    ) -> Dict[int, PitStopOutcome]:
        """
        Simulate multiple pit stop scenarios and compare outcomes
        """
        results = {}
        pit_lane_time = self.pit_lane_times.get(
            track_state.track_name.lower(), 
            self.pit_lane_times["default"]
        )
        
        for pit_lap in scenarios:
            # Calculate outcome for this scenario
            outcome = self._simulate_single_pit_scenario(
                driver, track_state, competitors, pit_lap, pit_lane_time
            )
            results[pit_lap] = outcome
            
        return results
        
    def _calculate_tire_advantage(self, target_degradation: float, track_temp: float) -> float:
        """Calculate tire advantage per lap for fresh vs degraded tires"""
        # Base advantage increases with degradation
        base_advantage = target_degradation * 0.02  # 0.02s per % degradation
        
        # Temperature factor (higher temp = more advantage)
        temp_factor = 1.0 + (track_temp - 45.0) * 0.01  # 1% per degree above 45C
        
        return base_advantage * temp_factor
        
    def _identify_undercut_threats(
        self, 
        driver: DriverState, 
        competitors: List[DriverState]
    ) -> List[str]:
        """Identify drivers behind who could undercut"""
        threats = []
        for competitor in competitors:
            if (competitor.current_position > driver.current_position and
                competitor.gap_to_ahead < 25.0 and  # Within undercut range
                competitor.tire_state.degradation_percent < driver.tire_state.degradation_percent):
                threats.append(competitor.driver_id)
        return threats
        
    def _identify_overcut_opportunities(
        self,
        driver: DriverState,
        competitors: List[DriverState]
    ) -> List[str]:
        """Identify drivers ahead who might pit soon"""
        opportunities = []
        for competitor in competitors:
            if (competitor.current_position < driver.current_position and
                competitor.tire_state.degradation_percent > 65 and
                driver.gap_to_ahead < 30.0):  # Within striking distance
                opportunities.append(competitor.driver_id)
        return opportunities
        
    def _calculate_safety_car_probability(
        self, 
        current_lap: int, 
        total_laps: int, 
        track_name: str
    ) -> float:
        """Calculate probability of safety car in next 10 laps"""
        race_progress = current_lap / total_laps
        
        # Base probability varies by track
        base_prob = {
            "monaco": 0.15,
            "singapore": 0.12,
            "baku": 0.18,
            "default": 0.08
        }.get(track_name.lower(), 0.08)
        
        # Higher probability in middle of race
        if 0.2 < race_progress < 0.8:
            return base_prob * 1.5
        else:
            return base_prob
            
    def _recommend_tire_compounds(
        self, 
        remaining_laps: int, 
        weather, 
        track_temp: float
    ) -> List[TireCompound]:
        """Recommend tire compounds for next stint"""
        recommendations = []
        
        if weather.value in ['light_rain', 'heavy_rain']:
            recommendations.append(TireCompound.INTERMEDIATE)
            if weather.value == 'heavy_rain':
                recommendations.append(TireCompound.WET)
        else:
            if remaining_laps < 15:
                recommendations.extend([TireCompound.SOFT, TireCompound.MEDIUM])
            elif remaining_laps < 30:
                recommendations.extend([TireCompound.MEDIUM, TireCompound.HARD])
            else:
                recommendations.append(TireCompound.HARD)
                
        return recommendations
        
    def _simulate_single_pit_scenario(
        self,
        driver: DriverState,
        track_state: TrackState,
        competitors: List[DriverState],
        pit_lap: int,
        pit_lane_time: float
    ) -> PitStopOutcome:
        """Simulate a single pit stop scenario"""
        # Simplified simulation
        laps_until_pit = pit_lap - track_state.current_lap
        
        # Predict position after pit stop
        drivers_likely_to_pass = 0
        for competitor in competitors:
            if (competitor.current_position > driver.current_position and
                competitor.gap_to_ahead < pit_lane_time + 5.0):
                drivers_likely_to_pass += 1
                
        new_position = min(20, driver.current_position + drivers_likely_to_pass)
        
        # Calculate benefits
        fresh_tire_benefit = 1.5 * 15  # 1.5s per lap for 15 laps
        net_benefit = fresh_tire_benefit - pit_lane_time
        
        # Success probability
        success_prob = 0.7 if net_benefit > 0 else 0.3
        
        return PitStopOutcome(
            new_position=new_position,
            time_lost=pit_lane_time,
            time_gained_next_stint=fresh_tire_benefit,
            net_benefit=net_benefit,
            success_probability=success_prob
        )
