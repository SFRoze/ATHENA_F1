"""
ATHENA F1 - Gap Analysis Algorithm
Analyzes gaps between drivers for strategic opportunities.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from ..data.models import DriverState, TrackState


@dataclass
class GapOpportunity:
    """Strategic opportunity based on gap analysis"""
    opportunity_type: str  # 'undercut', 'overcut', 'drs_attack', 'defend'
    target_driver_id: str
    gap_seconds: float
    success_probability: float
    time_window_laps: int
    description: str


class GapAnalyzer:
    """
    Analyzes gaps between drivers to identify strategic opportunities
    like undercuts, overcuts, and defensive positioning.
    """
    
    def __init__(self):
        # Gap thresholds for different opportunities (seconds)
        self.undercut_threshold = 25.0
        self.overcut_threshold = 30.0
        self.drs_threshold = 1.0
        self.defend_threshold = 20.0
        
    def analyze_all_gaps(
        self,
        target_driver: DriverState,
        all_drivers: List[DriverState],
        track_state: TrackState
    ) -> List[GapOpportunity]:
        """
        Analyze all gaps around a target driver for strategic opportunities
        """
        opportunities = []
        
        # Find driver ahead and behind
        driver_ahead = self._find_driver_ahead(target_driver, all_drivers)
        driver_behind = self._find_driver_behind(target_driver, all_drivers)
        
        # Analyze undercut opportunities (attack driver ahead)
        if driver_ahead:
            undercut_opp = self._analyze_undercut_opportunity(
                target_driver, driver_ahead, track_state
            )
            if undercut_opp:
                opportunities.append(undercut_opp)
                
        # Analyze overcut opportunities (stay out while others pit)
        overcut_opp = self._analyze_overcut_opportunity(
            target_driver, all_drivers, track_state
        )
        if overcut_opp:
            opportunities.append(overcut_opp)
            
        # Analyze DRS attack opportunities
        if driver_ahead and target_driver.drs_available:
            drs_opp = self._analyze_drs_opportunity(
                target_driver, driver_ahead, track_state
            )
            if drs_opp:
                opportunities.append(drs_opp)
                
        # Analyze defensive positioning
        if driver_behind:
            defend_opp = self._analyze_defensive_opportunity(
                target_driver, driver_behind, track_state
            )
            if defend_opp:
                opportunities.append(defend_opp)
                
        return opportunities
        
    def calculate_undercut_probability(
        self,
        attacking_driver: DriverState,
        target_driver: DriverState,
        pit_lane_time: float = 23.0
    ) -> float:
        """
        Calculate probability of successful undercut
        """
        gap = attacking_driver.gap_to_ahead
        
        # Tire advantage calculation
        tire_advantage = self._calculate_tire_advantage(
            attacking_driver.tire_state.degradation_percent,
            target_driver.tire_state.degradation_percent
        )
        
        # Effective gap including pit stop time
        effective_gap = gap + pit_lane_time
        
        # Laps needed to close gap
        if tire_advantage > 0:
            laps_to_close = effective_gap / tire_advantage
        else:
            return 0.0  # No tire advantage
            
        # Success probability based on laps to close
        if laps_to_close <= 3:
            return 0.9
        elif laps_to_close <= 6:
            return 0.7
        elif laps_to_close <= 10:
            return 0.4
        elif laps_to_close <= 15:
            return 0.2
        else:
            return 0.05
            
    def calculate_gap_evolution(
        self,
        driver1: DriverState,
        driver2: DriverState,
        future_laps: int,
        driver1_pits: bool = False,
        driver2_pits: bool = False,
        pit_lap: int = 0
    ) -> np.ndarray:
        """
        Predict how gap between two drivers will evolve over future laps
        """
        initial_gap = abs(driver1.gap_to_ahead if driver1.current_position > driver2.current_position 
                         else driver1.gap_to_behind)
        
        gaps = []
        
        for lap in range(future_laps):
            # Calculate relative pace difference
            pace_diff = self._calculate_relative_pace(
                driver1, driver2, lap, driver1_pits and lap >= pit_lap, driver2_pits and lap >= pit_lap
            )
            
            # Update gap
            if lap == 0:
                current_gap = initial_gap
            else:
                current_gap = gaps[-1] + pace_diff
                
            # Add pit stop time loss if applicable
            if driver1_pits and lap == pit_lap:
                current_gap += 23.0  # Pit lane time
            if driver2_pits and lap == pit_lap:
                current_gap -= 23.0  # Pit lane time
                
            gaps.append(max(0, current_gap))  # Gap can't be negative
            
        return np.array(gaps)
        
    def find_optimal_attack_window(
        self,
        attacking_driver: DriverState,
        target_driver: DriverState,
        max_window_laps: int = 20
    ) -> Tuple[int, int]:
        """
        Find the optimal window (earliest, latest lap) for attacking a driver
        """
        current_gap = attacking_driver.gap_to_ahead
        
        # Calculate when tire advantage peaks
        tire_advantage_evolution = []
        for lap in range(max_window_laps):
            advantage = self._calculate_future_tire_advantage(
                attacking_driver, target_driver, lap
            )
            tire_advantage_evolution.append(advantage)
            
        # Find peak advantage period
        tire_advantages = np.array(tire_advantage_evolution)
        peak_lap = np.argmax(tire_advantages)
        
        # Window around peak
        earliest = max(0, peak_lap - 3)
        latest = min(max_window_laps - 1, peak_lap + 5)
        
        return earliest, latest
        
    def calculate_drs_overtake_probability(
        self,
        attacking_driver: DriverState,
        target_driver: DriverState,
        track_name: str
    ) -> float:
        """
        Calculate probability of successful DRS overtake
        """
        if not attacking_driver.drs_available:
            return 0.0
            
        gap = attacking_driver.gap_to_ahead
        
        # Track-specific DRS effectiveness
        drs_effectiveness = {
            "monza": 0.8,      # High speed, long straights
            "spa": 0.7,
            "silverstone": 0.6,
            "bahrain": 0.5,
            "monaco": 0.1,     # Very difficult to overtake
            "default": 0.4
        }.get(track_name.lower(), 0.4)
        
        # Gap-based probability
        if gap <= 0.3:
            gap_prob = 0.9
        elif gap <= 0.7:
            gap_prob = 0.6
        elif gap <= 1.0:
            gap_prob = 0.3
        else:
            gap_prob = 0.0
            
        # Pace advantage
        pace_diff = target_driver.lap_time_current - attacking_driver.lap_time_current
        pace_factor = min(1.0, max(0.0, pace_diff / 0.5))  # 0.5s pace advantage = 100%
        
        return drs_effectiveness * gap_prob * (0.5 + 0.5 * pace_factor)
        
    def _find_driver_ahead(self, driver: DriverState, all_drivers: List[DriverState]) -> Optional[DriverState]:
        """Find the driver directly ahead in the race"""
        target_position = driver.current_position - 1
        return next((d for d in all_drivers if d.current_position == target_position), None)
        
    def _find_driver_behind(self, driver: DriverState, all_drivers: List[DriverState]) -> Optional[DriverState]:
        """Find the driver directly behind in the race"""
        target_position = driver.current_position + 1
        return next((d for d in all_drivers if d.current_position == target_position), None)
        
    def _analyze_undercut_opportunity(
        self,
        driver: DriverState,
        target_ahead: DriverState,
        track_state: TrackState
    ) -> Optional[GapOpportunity]:
        """Analyze undercut opportunity against driver ahead"""
        gap = driver.gap_to_ahead
        
        if gap > self.undercut_threshold:
            return None  # Gap too large
            
        # Calculate success probability
        success_prob = self.calculate_undercut_probability(driver, target_ahead)
        
        if success_prob < 0.3:
            return None  # Low probability
            
        # Time window calculation
        tire_advantage = self._calculate_tire_advantage(
            driver.tire_state.degradation_percent,
            target_ahead.tire_state.degradation_percent
        )
        
        time_window = min(10, int(25.0 / max(0.5, tire_advantage)))  # Max 10 laps
        
        return GapOpportunity(
            opportunity_type='undercut',
            target_driver_id=target_ahead.driver_id,
            gap_seconds=gap,
            success_probability=success_prob,
            time_window_laps=time_window,
            description=f"Undercut {target_ahead.driver_name} - {gap:.1f}s gap, {success_prob:.0%} success"
        )
        
    def _analyze_overcut_opportunity(
        self,
        driver: DriverState,
        all_drivers: List[DriverState],
        track_state: TrackState
    ) -> Optional[GapOpportunity]:
        """Analyze overcut opportunity by staying out longer"""
        if driver.tire_state.degradation_percent > 75:
            return None  # Tires too degraded
            
        # Find drivers ahead that might pit soon
        potential_targets = []
        for other_driver in all_drivers:
            if (other_driver.current_position < driver.current_position and
                other_driver.tire_state.degradation_percent > 60 and
                driver.gap_to_ahead < self.overcut_threshold):
                potential_targets.append(other_driver)
                
        if not potential_targets:
            return None
            
        # Calculate success probability based on tire degradation
        max_extended_laps = min(15, (85 - driver.tire_state.degradation_percent) // 2)
        success_prob = min(0.8, max_extended_laps / 10.0)
        
        target = potential_targets[0]  # Focus on closest ahead
        
        return GapOpportunity(
            opportunity_type='overcut',
            target_driver_id=target.driver_id,
            gap_seconds=driver.gap_to_ahead,
            success_probability=success_prob,
            time_window_laps=max_extended_laps,
            description=f"Overcut {target.driver_name} - extend stint {max_extended_laps} laps"
        )
        
    def _analyze_drs_opportunity(
        self,
        driver: DriverState,
        target_ahead: DriverState,
        track_state: TrackState
    ) -> Optional[GapOpportunity]:
        """Analyze DRS overtaking opportunity"""
        gap = driver.gap_to_ahead
        
        if gap > self.drs_threshold:
            return None
            
        success_prob = self.calculate_drs_overtake_probability(
            driver, target_ahead, track_state.track_name
        )
        
        if success_prob < 0.2:
            return None
            
        return GapOpportunity(
            opportunity_type='drs_attack',
            target_driver_id=target_ahead.driver_id,
            gap_seconds=gap,
            success_probability=success_prob,
            time_window_laps=3,  # DRS opportunities are immediate
            description=f"DRS attack on {target_ahead.driver_name} - {gap:.2f}s gap"
        )
        
    def _analyze_defensive_opportunity(
        self,
        driver: DriverState,
        threat_behind: DriverState,
        track_state: TrackState
    ) -> Optional[GapOpportunity]:
        """Analyze defensive positioning opportunity"""
        gap = driver.gap_to_behind
        
        if gap > self.defend_threshold:
            return None  # Safe gap
            
        # Check if driver behind has DRS or tire advantage
        threat_level = 0.0
        
        if threat_behind.drs_available and gap < 1.0:
            threat_level += 0.5
            
        if threat_behind.tire_state.degradation_percent < driver.tire_state.degradation_percent - 10:
            threat_level += 0.4
            
        if threat_level < 0.3:
            return None
            
        return GapOpportunity(
            opportunity_type='defend',
            target_driver_id=threat_behind.driver_id,
            gap_seconds=gap,
            success_probability=1.0 - threat_level,  # Inverse of threat
            time_window_laps=5,
            description=f"Defend against {threat_behind.driver_name} - {gap:.1f}s behind"
        )
        
    def _calculate_tire_advantage(self, attacker_degradation: float, target_degradation: float) -> float:
        """Calculate tire advantage in seconds per lap"""
        degradation_diff = target_degradation - attacker_degradation
        return max(0, degradation_diff * 0.02)  # 0.02s per % degradation difference
        
    def _calculate_relative_pace(
        self,
        driver1: DriverState,
        driver2: DriverState,
        lap_offset: int,
        driver1_fresh_tires: bool = False,
        driver2_fresh_tires: bool = False
    ) -> float:
        """Calculate relative pace difference between two drivers"""
        # Base pace difference
        pace_diff = driver2.lap_time_current - driver1.lap_time_current
        
        # Tire degradation effects
        if driver1_fresh_tires:
            pace_diff -= 1.5  # Fresh tire advantage
        if driver2_fresh_tires:
            pace_diff += 1.5  # Fresh tire advantage
            
        # Fuel effect (both drivers lose fuel at similar rate)
        # Simplified - no significant difference
        
        return pace_diff
        
    def _calculate_future_tire_advantage(
        self,
        attacker: DriverState,
        target: DriverState,
        laps_ahead: int
    ) -> float:
        """Calculate tire advantage at future lap"""
        # Simple degradation model
        degradation_rate = 2.0  # % per lap
        
        attacker_future_deg = attacker.tire_state.degradation_percent + (degradation_rate * laps_ahead)
        target_future_deg = target.tire_state.degradation_percent + (degradation_rate * laps_ahead)
        
        return self._calculate_tire_advantage(attacker_future_deg, target_future_deg)
