"""
ATHENA F1 - Race Data Simulator
Generates realistic F1 race data for testing and demonstration.
"""

import asyncio
import random
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Callable
from dataclasses import replace
from loguru import logger

from .models import (
    RaceState, DriverState, TireState, TrackState,
    TireCompound, WeatherCondition
)


class RaceSimulator:
    """
    Simulates realistic F1 race progression with dynamic driver states,
    tire degradation, and strategic events.
    """
    
    def __init__(self, track_name: str = "Silverstone", total_laps: int = 52):
        self.track_name = track_name
        self.total_laps = total_laps
        self.current_lap = 1
        self.race_state: Optional[RaceState] = None
        self.simulation_speed = 1.0  # Real-time multiplier
        self.is_running = False
        
        # Track-specific parameters
        self.track_params = self._get_track_parameters(track_name)
        
        # Event probabilities
        self.safety_car_probability = 0.02  # Per lap
        self.rain_probability = 0.01  # Per lap
        
        # Callbacks for real-time updates
        self.update_callbacks: List[Callable[[RaceState], None]] = []
        
    def initialize_race(self, driver_names: List[str] = None) -> RaceState:
        """Initialize race with drivers and starting conditions"""
        if driver_names is None:
            driver_names = [
                "Max Verstappen", "Lewis Hamilton", "Charles Leclerc", "Lando Norris",
                "George Russell", "Carlos Sainz", "Fernando Alonso", "Sergio Perez",
                "Oscar Piastri", "Alexander Albon", "Lance Stroll", "Yuki Tsunoda",
                "Daniel Ricciardo", "Pierre Gasly", "Nico Hulkenberg", "Kevin Magnussen",
                "Esteban Ocon", "Valtteri Bottas", "Zhou Guanyu", "Logan Sargeant"
            ]
            
        teams = [
            "Red Bull Racing", "Mercedes", "Ferrari", "McLaren",
            "Mercedes", "Ferrari", "Aston Martin", "Red Bull Racing",
            "McLaren", "Williams", "Aston Martin", "AlphaTauri",
            "AlphaTauri", "Alpine", "Haas", "Haas",
            "Alpine", "Alfa Romeo", "Alfa Romeo", "Williams"
        ]
        
        # Create initial driver states
        drivers = []
        base_lap_time = self.track_params['base_lap_time']
        
        for i, (name, team) in enumerate(zip(driver_names[:20], teams[:20])):
            # Vary driver pace slightly
            pace_variation = random.uniform(-0.8, 1.5)  # Some drivers faster/slower
            
            driver = DriverState(
                driver_id=f"driver_{i+1}",
                driver_name=name,
                team=team,
                current_position=i + 1,
                lap_time_current=base_lap_time + pace_variation,
                lap_time_best=base_lap_time + pace_variation,
                gap_to_leader=i * 1.2,  # Start with small gaps
                gap_to_ahead=1.2 if i > 0 else 0.0,
                gap_to_behind=1.2 if i < 19 else 0.0,
                tire_state=TireState(
                    compound=TireCompound.MEDIUM,
                    age_laps=0,
                    degradation_percent=0.0,
                    estimated_life_remaining=35,
                    grip_level=0.92
                ),
                fuel_remaining=110.0,  # Starting fuel load
                pit_stops_completed=0,
                pit_stop_time_avg=self.track_params['pit_stop_time'],
                drs_available=(i > 0)  # Leader doesn't have DRS
            )
            drivers.append(driver)
            
        # Create track state
        track_state = TrackState(
            track_name=self.track_name,
            total_laps=self.total_laps,
            current_lap=self.current_lap,
            weather=WeatherCondition.DRY,
            track_temperature=45.0,
            air_temperature=22.0,
            rain_probability=0.1
        )
        
        # Create race state
        self.race_state = RaceState(
            timestamp=datetime.now(),
            track_state=track_state,
            drivers=drivers
        )
        
        logger.info(f"Race initialized: {self.track_name} GP, {self.total_laps} laps, {len(drivers)} drivers")
        return self.race_state
        
    async def start_simulation(self, update_interval: float = 5.0):
        """Start the race simulation with real-time updates"""
        if not self.race_state:
            raise ValueError("Race not initialized. Call initialize_race() first.")
            
        self.is_running = True
        logger.info(f"Starting race simulation - {update_interval}s intervals")
        
        while self.is_running and self.current_lap <= self.total_laps:
            # Simulate one lap
            await self._simulate_lap()
            
            # Notify callbacks
            for callback in self.update_callbacks:
                try:
                    callback(self.race_state)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
                    
            # Wait for next update
            await asyncio.sleep(update_interval / self.simulation_speed)
            
        logger.info("Race simulation completed")
        
    def stop_simulation(self):
        """Stop the race simulation"""
        self.is_running = False
        logger.info("Race simulation stopped")
        
    def add_update_callback(self, callback: Callable[[RaceState], None]):
        """Add callback for race state updates"""
        self.update_callbacks.append(callback)
        
    def remove_update_callback(self, callback: Callable[[RaceState], None]):
        """Remove callback for race state updates"""
        if callback in self.update_callbacks:
            self.update_callbacks.remove(callback)
            
    async def _simulate_lap(self):
        """Simulate one lap of the race"""
        self.current_lap += 1
        
        # Update track state
        self.race_state.track_state.current_lap = self.current_lap
        self.race_state.timestamp = datetime.now()
        
        # Check for events (safety car, rain, etc.)
        await self._check_race_events()
        
        # Update each driver
        updated_drivers = []
        for driver in self.race_state.drivers:
            updated_driver = await self._simulate_driver_lap(driver)
            updated_drivers.append(updated_driver)
            
        self.race_state.drivers = updated_drivers
        
        # Update positions and gaps
        self._update_positions_and_gaps()
        
        # Simulate pit stops (random strategy)
        await self._simulate_pit_stops()
        
        # Update DRS availability
        self._update_drs_availability()
        
        logger.debug(f"Lap {self.current_lap} completed")
        
    async def _simulate_driver_lap(self, driver: DriverState) -> DriverState:
        """Simulate one lap for a single driver"""
        # Calculate lap time based on tire degradation and fuel
        base_time = driver.lap_time_best
        
        # Tire degradation effect
        degradation_penalty = driver.tire_state.degradation_percent * 0.02  # 2% degradation = 0.02s
        
        # Fuel effect (lighter = faster)
        fuel_benefit = (110.0 - driver.fuel_remaining) * 0.03  # 0.03s per kg saved
        
        # Random variation
        variation = random.uniform(-0.2, 0.3)
        
        # Weather effect
        weather_penalty = 0.0
        if self.race_state.track_state.weather != WeatherCondition.DRY:
            weather_penalty = random.uniform(1.0, 3.0)
            
        # Calculate final lap time
        lap_time = base_time + degradation_penalty - fuel_benefit + variation + weather_penalty
        
        # Update tire state
        new_tire_age = driver.tire_state.age_laps + 1
        degradation_rate = self._get_tire_degradation_rate(
            driver.tire_state.compound, 
            self.race_state.track_state.track_temperature
        )
        new_degradation = min(100.0, driver.tire_state.degradation_percent + degradation_rate)
        new_grip_level = max(0.1, 1.0 - (new_degradation / 100.0))
        
        updated_tire_state = replace(
            driver.tire_state,
            age_laps=new_tire_age,
            degradation_percent=new_degradation,
            grip_level=new_grip_level,
            estimated_life_remaining=max(0, driver.tire_state.estimated_life_remaining - 1)
        )
        
        # Update fuel (consume ~2kg per lap)
        new_fuel = max(5.0, driver.fuel_remaining - random.uniform(1.8, 2.2))
        
        # Create updated driver state
        return replace(
            driver,
            lap_time_current=lap_time,
            tire_state=updated_tire_state,
            fuel_remaining=new_fuel,
            lap_time_best=min(driver.lap_time_best, lap_time) if lap_time > 0 else driver.lap_time_best
        )
        
    def _update_positions_and_gaps(self):
        """Update driver positions and gaps based on lap times"""
        # Sort drivers by total time (simplified - just use gap to leader)
        drivers_with_times = []
        for driver in self.race_state.drivers:
            # Simulate total race time evolution
            if self.current_lap > 1:
                # Update gap to leader based on lap time difference
                leader_lap_time = min(d.lap_time_current for d in self.race_state.drivers)
                time_diff = driver.lap_time_current - leader_lap_time
                new_gap_to_leader = max(0, driver.gap_to_leader + time_diff)
            else:
                new_gap_to_leader = 0 if driver.current_position == 1 else driver.gap_to_leader
                
            drivers_with_times.append((driver, new_gap_to_leader))
            
        # Sort by gap to leader
        drivers_with_times.sort(key=lambda x: x[1])
        
        # Update positions and gaps
        updated_drivers = []
        for i, (driver, gap_to_leader) in enumerate(drivers_with_times):
            new_position = i + 1
            
            # Calculate gaps to ahead/behind
            gap_to_ahead = 0.0 if i == 0 else drivers_with_times[i][1] - drivers_with_times[i-1][1]
            gap_to_behind = 0.0 if i == len(drivers_with_times)-1 else drivers_with_times[i+1][1] - gap_to_leader
            
            updated_driver = replace(
                driver,
                current_position=new_position,
                gap_to_leader=gap_to_leader,
                gap_to_ahead=gap_to_ahead,
                gap_to_behind=gap_to_behind
            )
            updated_drivers.append(updated_driver)
            
        self.race_state.drivers = updated_drivers
        
    async def _simulate_pit_stops(self):
        """Simulate strategic pit stops"""
        for i, driver in enumerate(self.race_state.drivers):
            # Simple pit stop decision logic
            should_pit = (
                driver.tire_state.degradation_percent > 70 and
                random.random() < 0.3  # 30% chance if tires degraded
            ) or (
                driver.tire_state.age_laps > 25 and
                random.random() < 0.15  # 15% chance if tires old
            )
            
            if should_pit:
                await self._execute_pit_stop(i)
                
    async def _execute_pit_stop(self, driver_index: int):
        """Execute a pit stop for a driver"""
        driver = self.race_state.drivers[driver_index]
        
        # Choose new tire compound
        remaining_laps = self.total_laps - self.current_lap
        if remaining_laps < 15:
            new_compound = TireCompound.SOFT
        elif remaining_laps < 30:
            new_compound = random.choice([TireCompound.SOFT, TireCompound.MEDIUM])
        else:
            new_compound = random.choice([TireCompound.MEDIUM, TireCompound.HARD])
            
        # Create new tire state
        compound_params = {
            TireCompound.SOFT: (1.0, 25),
            TireCompound.MEDIUM: (0.92, 35),
            TireCompound.HARD: (0.85, 50)
        }
        grip, life = compound_params[new_compound]
        
        new_tire_state = TireState(
            compound=new_compound,
            age_laps=0,
            degradation_percent=0.0,
            estimated_life_remaining=life,
            grip_level=grip
        )
        
        # Time penalty for pit stop
        pit_time_penalty = driver.pit_stop_time_avg + random.uniform(-1.0, 2.0)
        new_gap_to_leader = driver.gap_to_leader + pit_time_penalty
        
        # Update driver
        updated_driver = replace(
            driver,
            tire_state=new_tire_state,
            pit_stops_completed=driver.pit_stops_completed + 1,
            gap_to_leader=new_gap_to_leader
        )
        
        self.race_state.drivers[driver_index] = updated_driver
        
        logger.info(f"{driver.driver_name} pitted for {new_compound.value} tires (Lap {self.current_lap})")
        
    def _update_drs_availability(self):
        """Update DRS availability based on gaps"""
        for i, driver in enumerate(self.race_state.drivers):
            if driver.current_position == 1:
                # Leader never has DRS
                drs_available = False
            else:
                # DRS available if within 1 second of car ahead
                drs_available = driver.gap_to_ahead < 1.0 and self.race_state.track_state.drs_enabled
                
            self.race_state.drivers[i] = replace(driver, drs_available=drs_available)
            
    async def _check_race_events(self):
        """Check for and simulate race events like safety cars"""
        # Safety car check
        if random.random() < self.safety_car_probability:
            logger.info(f"Safety Car deployed on lap {self.current_lap}")
            self.race_state.track_state.safety_car_deployed = True
            # Reset in a few laps
            await asyncio.sleep(15.0 / self.simulation_speed)  # 3 laps at 5s/lap
            self.race_state.track_state.safety_car_deployed = False
            logger.info("Safety Car returns to pits")
            
        # Weather change check
        if random.random() < self.rain_probability:
            current_weather = self.race_state.track_state.weather
            if current_weather == WeatherCondition.DRY:
                new_weather = WeatherCondition.LIGHT_RAIN
                logger.info(f"Rain starts on lap {self.current_lap}")
            else:
                new_weather = WeatherCondition.DRY
                logger.info(f"Rain stops on lap {self.current_lap}")
                
            self.race_state.track_state.weather = new_weather
            
    def _get_track_parameters(self, track_name: str) -> Dict[str, float]:
        """Get track-specific parameters"""
        tracks = {
            "silverstone": {
                "base_lap_time": 88.0,
                "pit_stop_time": 22.0,
                "overtaking_difficulty": 0.6
            },
            "monaco": {
                "base_lap_time": 72.0,
                "pit_stop_time": 25.0,
                "overtaking_difficulty": 0.9
            },
            "monza": {
                "base_lap_time": 82.0,
                "pit_stop_time": 20.0,
                "overtaking_difficulty": 0.3
            },
            "default": {
                "base_lap_time": 85.0,
                "pit_stop_time": 23.0,
                "overtaking_difficulty": 0.5
            }
        }
        return tracks.get(track_name.lower(), tracks["default"])
        
    def _get_tire_degradation_rate(self, compound: TireCompound, track_temp: float) -> float:
        """Get tire degradation rate per lap"""
        base_rates = {
            TireCompound.SOFT: 2.5,
            TireCompound.MEDIUM: 1.5,
            TireCompound.HARD: 0.8,
            TireCompound.INTERMEDIATE: 2.0,
            TireCompound.WET: 1.8
        }
        
        base_rate = base_rates[compound]
        
        # Temperature effect
        if track_temp > 50:
            temp_multiplier = 1.0 + ((track_temp - 50) * 0.02)
        else:
            temp_multiplier = 1.0
            
        return base_rate * temp_multiplier
        
    @property
    def current_state(self) -> Optional[RaceState]:
        """Get current race state"""
        return self.race_state
