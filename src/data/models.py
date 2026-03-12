"""
ATHENA F1 - Data Models
Core data structures for race state, driver information, and strategic options.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
from datetime import datetime, timedelta
import uuid


class TireCompound(Enum):
    """F1 tire compounds with degradation characteristics"""
    SOFT = "soft"
    MEDIUM = "medium" 
    HARD = "hard"
    INTERMEDIATE = "intermediate"
    WET = "wet"


class WeatherCondition(Enum):
    """Weather conditions affecting strategy"""
    DRY = "dry"
    LIGHT_RAIN = "light_rain"
    HEAVY_RAIN = "heavy_rain"
    CHANGING = "changing"


class StrategyType(Enum):
    """Types of strategic moves"""
    PIT_STOP = "pit_stop"
    STAY_OUT = "stay_out"
    UNDERCUT = "undercut"
    OVERCUT = "overcut"
    SAFETY_CAR_PIT = "safety_car_pit"
    DEFENSIVE = "defensive"


@dataclass
class TireState:
    """Current tire information for a driver"""
    compound: TireCompound
    age_laps: int
    degradation_percent: float  # 0-100%
    estimated_life_remaining: int  # laps
    grip_level: float  # 0-1.0 multiplier
    
    def __post_init__(self):
        # Ensure values are within valid ranges
        self.degradation_percent = max(0, min(100, self.degradation_percent))
        self.grip_level = max(0, min(1.0, self.grip_level))


@dataclass
class DriverState:
    """Complete state information for a single driver"""
    driver_id: str
    driver_name: str
    team: str
    current_position: int
    lap_time_current: float  # seconds
    lap_time_best: float
    gap_to_leader: float  # seconds
    gap_to_ahead: float  # seconds  
    gap_to_behind: float  # seconds
    tire_state: TireState
    fuel_remaining: float  # kg
    pit_stops_completed: int
    pit_stop_time_avg: float  # seconds (team average)
    in_pit_window: bool = False
    drs_available: bool = False
    penalties: List[str] = field(default_factory=list)
    
    @property
    def pace_potential(self) -> float:
        """Calculate potential lap time with fresh tires"""
        tire_degradation_factor = 1 + (self.tire_state.degradation_percent * 0.001)
        fuel_factor = 1 + (self.fuel_remaining * 0.0003)  # ~0.03s per kg
        return self.lap_time_best * tire_degradation_factor * fuel_factor


@dataclass 
class TrackState:
    """Track and environmental conditions"""
    track_name: str
    total_laps: int
    current_lap: int
    weather: WeatherCondition
    track_temperature: float  # Celsius
    air_temperature: float
    rain_probability: float  # 0-1.0
    safety_car_deployed: bool = False
    virtual_safety_car: bool = False
    drs_enabled: bool = True
    pit_lane_open: bool = True
    
    @property
    def race_completion(self) -> float:
        """Percentage of race completed"""
        return (self.current_lap / self.total_laps) * 100


@dataclass
class RaceState:
    """Complete race state at a given moment"""
    timestamp: datetime
    track_state: TrackState
    drivers: List[DriverState]
    race_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def get_driver(self, driver_id: str) -> Optional[DriverState]:
        """Get driver state by ID"""
        return next((d for d in self.drivers if d.driver_id == driver_id), None)
    
    def get_driver_by_position(self, position: int) -> Optional[DriverState]:
        """Get driver state by current position"""
        return next((d for d in self.drivers if d.current_position == position), None)
    
    @property
    def leaderboard(self) -> List[DriverState]:
        """Drivers sorted by current position"""
        return sorted(self.drivers, key=lambda d: d.current_position)


@dataclass
class StrategyOption:
    """A potential strategic move"""
    strategy_type: StrategyType
    target_driver: str
    estimated_outcome_position: int
    probability_success: float  # 0-1.0
    estimated_time_gain: float  # seconds (positive = gain, negative = loss)
    tire_compound_target: Optional[TireCompound] = None
    execute_on_lap: Optional[int] = None
    risk_level: str = "medium"  # low, medium, high
    description: str = ""
    
    @property
    def expected_value(self) -> float:
        """Expected value calculation for strategy ranking"""
        position_gain = max(0, self.estimated_outcome_position)
        return (self.probability_success * position_gain * 100) + self.estimated_time_gain


@dataclass
class StrategyRecommendation:
    """Complete strategy recommendation with multiple options"""
    target_driver: str
    timestamp: datetime
    primary_option: StrategyOption
    alternative_options: List[StrategyOption]
    confidence_score: float  # 0-1.0
    reasoning: str
    valid_until_lap: int
    
    @property
    def all_options(self) -> List[StrategyOption]:
        """All strategy options sorted by expected value"""
        options = [self.primary_option] + self.alternative_options
        return sorted(options, key=lambda x: x.expected_value, reverse=True)


@dataclass
class PitWindowAnalysis:
    """Analysis of optimal pit stop windows"""
    driver_id: str
    current_lap: int
    optimal_lap_range: Tuple[int, int]  # (earliest, latest)
    undercut_opportunities: List[str]  # driver IDs that can be undercut
    overcut_opportunities: List[str]   # driver IDs that can be overcut
    safety_car_probability: float      # probability of SC in window
    track_position_risk: float         # risk of losing positions
    tire_degradation_urgency: float    # 0-1.0 urgency based on tire state
