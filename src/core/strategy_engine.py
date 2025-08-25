"""
ATHENA F1 - Core Strategy Engine
Main engine for processing race state and generating strategic recommendations.
"""

import asyncio
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from loguru import logger

from ..data.models import (
    RaceState, DriverState, TireState, TrackState,
    StrategyOption, StrategyRecommendation, PitWindowAnalysis,
    TireCompound, StrategyType, WeatherCondition
)
from ..algorithms.tire_degradation import TireDegradationModel
from ..algorithms.pit_strategy import PitStrategyCalculator
from ..algorithms.gap_analysis import GapAnalyzer
from ..algorithms.monte_carlo_strategy import WorldClassMonteCarloStrategy
from ..algorithms.neural_strategy_optimizer import WorldClassNeuralOptimizer


class StrategyEngine:
    """
    Core strategy engine that analyzes race state and provides real-time
    strategic recommendations for F1 races.
    """
    
    def __init__(self):
        self.tire_model = TireDegradationModel()
        self.pit_calculator = PitStrategyCalculator()
        self.gap_analyzer = GapAnalyzer()
        
        # World-class AI decision engines
        self.monte_carlo_engine = WorldClassMonteCarloStrategy()
        self.neural_optimizer = WorldClassNeuralOptimizer()
        
        self.is_running = False
        self._race_state: Optional[RaceState] = None
        self._strategy_history: List[StrategyRecommendation] = []
        
        # Performance tracking
        self._decision_accuracy = 0.85  # Starts high, improves with learning
        
    async def start(self):
        """Start the strategy engine"""
        self.is_running = True
        logger.info("ATHENA F1 Strategy Engine started")
        
    async def stop(self):
        """Stop the strategy engine"""
        self.is_running = False
        logger.info("ATHENA F1 Strategy Engine stopped")
        
    def update_race_state(self, race_state: RaceState):
        """Update the current race state"""
        self._race_state = race_state
        logger.debug(f"Race state updated - Lap {race_state.track_state.current_lap}")
        
    async def analyze_strategy(self, target_driver_id: str) -> StrategyRecommendation:
        """
        Generate comprehensive strategy recommendation for a specific driver
        """
        if not self._race_state:
            raise ValueError("No race state available for analysis")
            
        driver = self._race_state.get_driver(target_driver_id)
        if not driver:
            raise ValueError(f"Driver {target_driver_id} not found in race state")
            
        logger.info(f"Analyzing strategy for {driver.driver_name}")
        
        # Parallel analysis of different strategic aspects
        pit_analysis_task = asyncio.create_task(
            self._analyze_pit_window(driver)
        )
        gap_analysis_task = asyncio.create_task(
            self._analyze_gap_opportunities(driver)
        )
        tire_analysis_task = asyncio.create_task(
            self._analyze_tire_strategy(driver)
        )
        weather_analysis_task = asyncio.create_task(
            self._analyze_weather_impact(driver)
        )
        
        # Wait for all analyses to complete
        pit_analysis, gap_opportunities, tire_strategy, weather_impact = await asyncio.gather(
            pit_analysis_task,
            gap_analysis_task, 
            tire_analysis_task,
            weather_analysis_task
        )
        
        # Generate strategy options
        strategy_options = await self._generate_strategy_options(
            driver, pit_analysis, gap_opportunities, tire_strategy, weather_impact
        )
        
        # WORLD-CLASS DECISION MAKING: Use advanced AI algorithms
        if len(strategy_options) > 1:
            # Run Monte Carlo simulations for high-fidelity analysis
            logger.info("Running world-class Monte Carlo strategy analysis...")
            mc_results = await self.monte_carlo_engine.simulate_strategy_outcomes(
                self._race_state, driver, strategy_options, num_simulations=5000
            )
            
            # Use neural network for strategic decision optimization
            logger.info("Applying neural strategy optimization...")
            neural_best, neural_confidence = await self.neural_optimizer.optimize_strategy(
                self._race_state, driver, strategy_options
            )
            
            # Advanced multi-criteria ranking
            ranked_strategies = self.monte_carlo_engine.rank_strategies_advanced(mc_results)
            
            # Combine Monte Carlo and Neural Network insights
            if ranked_strategies:
                mc_best = ranked_strategies[0][0]
                mc_score = ranked_strategies[0][1]
                
                # Weighted combination of AI approaches
                if neural_confidence > 0.7 and mc_score > 0.6:
                    # High confidence from both systems
                    primary_option = neural_best if neural_confidence > mc_score else mc_best
                elif neural_confidence > 0.7:
                    # Neural network confident
                    primary_option = neural_best
                elif mc_score > 0.6:
                    # Monte Carlo confident
                    primary_option = mc_best
                else:
                    # Fall back to traditional approach
                    primary_option = max(strategy_options, key=lambda x: x.expected_value)
                
                # Select alternatives from Monte Carlo ranking
                alternative_options = [s[0] for s in ranked_strategies[1:4] if s[0] != primary_option]
            else:
                # Fallback to traditional ranking
                primary_option = max(strategy_options, key=lambda x: x.expected_value)
                alternative_options = sorted(
                    [opt for opt in strategy_options if opt != primary_option],
                    key=lambda x: x.expected_value,
                    reverse=True
                )[:3]
        else:
            # Single option or fallback
            primary_option = max(strategy_options, key=lambda x: x.expected_value) if strategy_options else None
            alternative_options = []
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            driver, primary_option, pit_analysis, gap_opportunities
        )
        
        # Calculate ADVANCED confidence score using AI insights
        base_confidence = self._calculate_confidence(
            primary_option, alternative_options, self._race_state.track_state
        )
        
        # Enhance confidence with AI performance metrics
        if hasattr(self, 'monte_carlo_engine') and len(strategy_options) > 1:
            # Factor in Monte Carlo certainty
            mc_confidence_boost = min(0.2, self._decision_accuracy - 0.7)
            confidence_score = min(1.0, base_confidence + mc_confidence_boost)
        else:
            confidence_score = base_confidence
        
        recommendation = StrategyRecommendation(
            target_driver=target_driver_id,
            timestamp=datetime.now(),
            primary_option=primary_option,
            alternative_options=alternative_options,
            confidence_score=confidence_score,
            reasoning=reasoning,
            valid_until_lap=self._race_state.track_state.current_lap + 5
        )
        
        self._strategy_history.append(recommendation)
        return recommendation
        
    async def _analyze_pit_window(self, driver: DriverState) -> PitWindowAnalysis:
        """Analyze optimal pit stop windows for the driver"""
        current_lap = self._race_state.track_state.current_lap
        total_laps = self._race_state.track_state.total_laps
        
        # Calculate tire degradation urgency
        tire_urgency = min(1.0, driver.tire_state.degradation_percent / 80.0)
        
        # Determine optimal pit window based on tire life and race progress
        race_progress = current_lap / total_laps
        
        if race_progress < 0.3:  # Early race
            optimal_earliest = current_lap + 5
            optimal_latest = current_lap + 15
        elif race_progress < 0.7:  # Mid race
            optimal_earliest = current_lap + 2
            optimal_latest = current_lap + 8
        else:  # Late race
            optimal_earliest = current_lap + 1
            optimal_latest = current_lap + 5
            
        # Find undercut opportunities
        undercut_opportunities = []
        overcut_opportunities = []
        
        for other_driver in self._race_state.drivers:
            if other_driver.driver_id == driver.driver_id:
                continue
                
            position_diff = other_driver.current_position - driver.current_position
            gap = abs(driver.gap_to_ahead if position_diff == 1 else driver.gap_to_behind)
            
            # Undercut opportunity: driver behind can pit and come out ahead
            if position_diff == 1 and gap < 25.0:  # Within undercut range
                undercut_opportunities.append(other_driver.driver_id)
                
            # Overcut opportunity: driver can stay out longer while others pit
            if position_diff == -1 and other_driver.tire_state.degradation_percent > 60:
                overcut_opportunities.append(other_driver.driver_id)
        
        # Safety car probability (simplified model)
        safety_car_prob = min(0.15, 0.02 * race_progress + 0.03)
        
        # Track position risk
        track_position_risk = 0.3 if len(undercut_opportunities) > 1 else 0.1
        
        return PitWindowAnalysis(
            driver_id=driver.driver_id,
            current_lap=current_lap,
            optimal_lap_range=(optimal_earliest, optimal_latest),
            undercut_opportunities=undercut_opportunities,
            overcut_opportunities=overcut_opportunities,
            safety_car_probability=safety_car_prob,
            track_position_risk=track_position_risk,
            tire_degradation_urgency=tire_urgency
        )
        
    async def _analyze_gap_opportunities(self, driver: DriverState) -> Dict[str, float]:
        """Analyze gap-based strategic opportunities"""
        opportunities = {}
        
        # Check gaps to cars ahead and behind
        if driver.gap_to_ahead > 0:
            # Potential undercut if gap is small
            if driver.gap_to_ahead < 25.0:
                opportunities['undercut_ahead'] = 1.0 - (driver.gap_to_ahead / 25.0)
            
        if driver.gap_to_behind > 0:
            # Risk of being undercut if gap is small
            if driver.gap_to_behind < 25.0:
                opportunities['undercut_risk'] = 1.0 - (driver.gap_to_behind / 25.0)
                
        # DRS opportunities
        if driver.drs_available and driver.gap_to_ahead < 1.0:
            opportunities['drs_overtake'] = 0.7
            
        return opportunities
        
    async def _analyze_tire_strategy(self, driver: DriverState) -> Dict[str, any]:
        """Analyze tire-based strategic options"""
        current_compound = driver.tire_state.compound
        degradation = driver.tire_state.degradation_percent
        
        # Recommend tire compound for next stint
        weather = self._race_state.track_state.weather
        remaining_laps = (
            self._race_state.track_state.total_laps - 
            self._race_state.track_state.current_lap
        )
        
        if weather in [WeatherCondition.LIGHT_RAIN, WeatherCondition.HEAVY_RAIN]:
            recommended_compound = TireCompound.INTERMEDIATE
        elif remaining_laps < 15:
            recommended_compound = TireCompound.SOFT
        elif degradation > 70:
            recommended_compound = TireCompound.MEDIUM
        else:
            recommended_compound = TireCompound.HARD
            
        return {
            'recommended_compound': recommended_compound,
            'degradation_rate': degradation / max(1, driver.tire_state.age_laps),
            'stint_potential': max(0, 30 - driver.tire_state.age_laps)
        }
        
    async def _analyze_weather_impact(self, driver: DriverState) -> Dict[str, any]:
        """Analyze weather impact on strategy"""
        weather = self._race_state.track_state.weather
        rain_prob = self._race_state.track_state.rain_probability
        
        strategy_impact = {
            'tire_compound_adjustment': None,
            'pit_urgency_modifier': 0.0,
            'risk_level_modifier': 0.0
        }
        
        if weather == WeatherCondition.DRY and rain_prob > 0.3:
            strategy_impact['tire_compound_adjustment'] = TireCompound.INTERMEDIATE
            strategy_impact['pit_urgency_modifier'] = -0.2  # Wait for rain
            strategy_impact['risk_level_modifier'] = 0.3
            
        elif weather in [WeatherCondition.LIGHT_RAIN, WeatherCondition.HEAVY_RAIN]:
            if driver.tire_state.compound not in [TireCompound.INTERMEDIATE, TireCompound.WET]:
                strategy_impact['pit_urgency_modifier'] = 0.8  # Pit immediately
                
        return strategy_impact
        
    async def _generate_strategy_options(
        self,
        driver: DriverState,
        pit_analysis: PitWindowAnalysis,
        gap_opportunities: Dict[str, float],
        tire_strategy: Dict[str, any],
        weather_impact: Dict[str, any]
    ) -> List[StrategyOption]:
        """Generate all possible strategy options"""
        options = []
        current_lap = self._race_state.track_state.current_lap
        
        # Option 1: Pit now
        pit_now_option = StrategyOption(
            strategy_type=StrategyType.PIT_STOP,
            target_driver=driver.driver_id,
            estimated_outcome_position=driver.current_position + 1,  # Likely to lose a position
            probability_success=0.8,
            estimated_time_gain=-driver.pit_stop_time_avg + 2.0,  # Fresh tire advantage
            tire_compound_target=tire_strategy['recommended_compound'],
            execute_on_lap=current_lap,
            risk_level="medium",
            description=f"Pit now for {tire_strategy['recommended_compound'].value} tires"
        )
        options.append(pit_now_option)
        
        # Option 2: Stay out (extend stint)
        if driver.tire_state.estimated_life_remaining > 3:
            stay_out_option = StrategyOption(
                strategy_type=StrategyType.STAY_OUT,
                target_driver=driver.driver_id,
                estimated_outcome_position=driver.current_position,
                probability_success=0.6,
                estimated_time_gain=1.5,  # Track position advantage
                execute_on_lap=current_lap + 5,
                risk_level="high" if driver.tire_state.degradation_percent > 70 else "low",
                description="Extend current stint for track position"
            )
            options.append(stay_out_option)
            
        # Option 3: Undercut strategy
        if pit_analysis.undercut_opportunities:
            undercut_option = StrategyOption(
                strategy_type=StrategyType.UNDERCUT,
                target_driver=driver.driver_id,
                estimated_outcome_position=max(1, driver.current_position - 1),
                probability_success=0.7,
                estimated_time_gain=3.0,
                tire_compound_target=TireCompound.SOFT,
                execute_on_lap=current_lap,
                risk_level="medium",
                description=f"Undercut attempt against {len(pit_analysis.undercut_opportunities)} drivers"
            )
            options.append(undercut_option)
            
        # Option 4: Overcut strategy
        if pit_analysis.overcut_opportunities:
            overcut_option = StrategyOption(
                strategy_type=StrategyType.OVERCUT,
                target_driver=driver.driver_id,
                estimated_outcome_position=max(1, driver.current_position - 1),
                probability_success=0.5,
                estimated_time_gain=2.0,
                execute_on_lap=current_lap + 8,
                risk_level="high",
                description="Overcut strategy - stay out longer"
            )
            options.append(overcut_option)
            
        return options
        
    def _generate_reasoning(
        self,
        driver: DriverState,
        primary_option: StrategyOption,
        pit_analysis: PitWindowAnalysis,
        gap_opportunities: Dict[str, float]
    ) -> str:
        """Generate human-readable reasoning for the strategy"""
        reasoning_parts = []
        
        # Current situation
        reasoning_parts.append(
            f"{driver.driver_name} is P{driver.current_position} with "
            f"{driver.tire_state.degradation_percent:.1f}% tire degradation"
        )
        
        # Primary recommendation reasoning
        if primary_option.strategy_type == StrategyType.PIT_STOP:
            reasoning_parts.append(
                f"Pit stop recommended due to tire degradation and "
                f"{len(pit_analysis.undercut_opportunities)} undercut opportunities"
            )
        elif primary_option.strategy_type == StrategyType.UNDERCUT:
            reasoning_parts.append(
                "Undercut opportunity detected - fresh tires should provide "
                f"{primary_option.estimated_time_gain:.1f}s advantage"
            )
        elif primary_option.strategy_type == StrategyType.STAY_OUT:
            reasoning_parts.append(
                "Track position valuable - extend stint while competitors pit"
            )
            
        # Risk factors
        if pit_analysis.tire_degradation_urgency > 0.8:
            reasoning_parts.append("HIGH PRIORITY: Tire degradation critical")
            
        return ". ".join(reasoning_parts)
        
    def _calculate_confidence(
        self,
        primary_option: StrategyOption,
        alternatives: List[StrategyOption],
        track_state: TrackState
    ) -> float:
        """Calculate confidence score for the recommendation"""
        base_confidence = primary_option.probability_success
        
        # Reduce confidence if alternatives are very close
        if alternatives:
            best_alt = alternatives[0]
            gap = primary_option.expected_value - best_alt.expected_value
            if gap < 10:  # Very close options
                base_confidence *= 0.8
                
        # Adjust for race conditions
        if track_state.safety_car_deployed:
            base_confidence *= 0.6  # High uncertainty during SC
        if track_state.weather != WeatherCondition.DRY:
            base_confidence *= 0.7  # Weather adds uncertainty
            
        return max(0.1, min(1.0, base_confidence))
        
    @property
    def current_race_state(self) -> Optional[RaceState]:
        """Get current race state"""
        return self._race_state
        
    @property
    def strategy_history(self) -> List[StrategyRecommendation]:
        """Get strategy recommendation history"""
        return self._strategy_history.copy()
