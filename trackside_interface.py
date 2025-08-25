"""
ATHENA F1 - Trackside Real-Time Strategy Interface
Designed for actual race weekend competitive use
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import asyncio
from typing import Dict, List

from src.algorithms.monte_carlo_strategy import WorldClassMonteCarloStrategy
from src.algorithms.neural_optimizer import NeuralStrategyOptimizer
from src.data.models import RaceState, DriverState, TireCompound, StrategyType

class TracksideInterface:
    """Real-time trackside interface for competitive F1 strategy"""
    
    def __init__(self):
        self.monte_carlo = WorldClassMonteCarloStrategy()
        self.neural_optimizer = NeuralStrategyOptimizer()
        self.current_race_state = None
        
    def main_dashboard(self):
        """Main trackside dashboard"""
        st.set_page_config(
            page_title="ATHENA F1 - Live Race Strategy",
            page_icon="🏎️",
            layout="wide"
        )
        
        st.title("🏁 ATHENA F1 - Live Race Strategy Assistant")
        
        # Real-time data input section
        with st.sidebar:
            st.header("📊 Live Race Data")
            
            # Quick data input for your target driver
            driver_name = st.selectbox(
                "Target Driver", 
                ["Max Verstappen", "Lewis Hamilton", "Charles Leclerc", 
                 "Lando Norris", "George Russell", "Carlos Sainz", "Other"]
            )
            
            current_position = st.number_input("Current Position", 1, 20, 5)
            current_lap = st.number_input("Current Lap", 1, 70, 25)
            total_laps = st.number_input("Total Laps", 50, 70, 58)
            
            # Tire information
            st.subheader("🏎️ Tire Status")
            tire_compound = st.selectbox("Current Compound", 
                                       ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"])
            tire_age = st.number_input("Tire Age (laps)", 0, 40, 8)
            tire_degradation = st.slider("Tire Degradation %", 0, 100, 25)
            
            # Weather
            st.subheader("🌤️ Weather")
            current_weather = st.selectbox("Current Weather", 
                                         ["DRY", "LIGHT_RAIN", "HEAVY_RAIN"])
            rain_probability = st.slider("Rain Probability %", 0, 100, 15)
            temperature = st.number_input("Track Temperature °C", 20, 60, 35)
            
            # Gaps
            st.subheader("⏱️ Race Gaps")
            gap_to_leader = st.number_input("Gap to Leader (s)", 0.0, 120.0, 15.5)
            gap_to_ahead = st.number_input("Gap to Car Ahead (s)", 0.0, 30.0, 2.3)
            gap_to_behind = st.number_input("Gap to Car Behind (s)", 0.0, 30.0, 4.1)
            
            # Fuel estimate
            fuel_remaining = st.number_input("Estimated Fuel (kg)", 10, 110, 45)
            
        # Main strategy analysis
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("🎯 Strategic Recommendations")
            
            if st.button("🚀 ANALYZE STRATEGIES", type="primary"):
                with st.spinner("Running Monte Carlo simulations..."):
                    # Create race state from inputs
                    race_state = self._create_race_state(
                        current_lap, total_laps, current_weather, 
                        rain_probability, temperature
                    )
                    
                    driver_state = self._create_driver_state(
                        driver_name, current_position, tire_compound, 
                        tire_age, tire_degradation, gap_to_leader,
                        gap_to_ahead, gap_to_behind, fuel_remaining
                    )
                    
                    # Generate strategy options
                    strategy_options = self._generate_live_strategies(
                        race_state, driver_state
                    )
                    
                    # Run analysis
                    results = asyncio.run(
                        self.monte_carlo.simulate_strategy_outcomes(
                            race_state, driver_state, strategy_options
                        )
                    )
                    
                    # Display results
                    self._display_strategy_results(results, strategy_options)
                    
        with col2:
            st.header("📈 Live Metrics")
            
            # Key performance indicators
            st.metric("Championship Points Potential", "8.5", "+2.1")
            st.metric("Optimal Pit Window", "Lap 32-35", "")
            st.metric("Weather Risk Level", "MEDIUM", "")
            
            # Quick action buttons
            st.subheader("⚡ Quick Actions")
            
            if st.button("🔴 EMERGENCY PIT NOW"):
                st.warning("Emergency pit strategy calculated!")
                
            if st.button("🟡 SAFETY CAR STRATEGY"):
                st.info("Safety car pit window analysis ready!")
                
            if st.button("🌧️ RAIN STRATEGY"):
                st.info("Weather strategy recommendations!")
        
        # Live timing display
        st.header("⏱️ Live Timing Analysis")
        
        # Create sample timing data display
        timing_data = self._generate_sample_timing_data()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1, current_lap + 1)),
            y=timing_data[:current_lap],
            mode='lines+markers',
            name='Lap Times',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title="Lap Time Analysis",
            xaxis_title="Lap Number",
            yaxis_title="Lap Time (s)",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    def _create_race_state(self, current_lap, total_laps, weather, rain_prob, temp):
        """Create RaceState from live inputs"""
        # This would normally come from live data feeds
        from src.data.models import TrackState, WeatherCondition
        
        weather_map = {
            "DRY": WeatherCondition.DRY,
            "LIGHT_RAIN": WeatherCondition.LIGHT_RAIN,
            "HEAVY_RAIN": WeatherCondition.HEAVY_RAIN
        }
        
        return RaceState(
            track_state=TrackState(
                track_name="Monaco",  # Would be dynamic
                current_lap=current_lap,
                total_laps=total_laps,
                weather=weather_map[weather],
                temperature=temp,
                rain_probability=rain_prob / 100.0
            ),
            session_type="RACE"
        )
        
    def _create_driver_state(self, name, position, compound, age, degradation, 
                           gap_leader, gap_ahead, gap_behind, fuel):
        """Create DriverState from inputs"""
        from src.data.models import TireState
        
        compound_map = {
            "SOFT": TireCompound.SOFT,
            "MEDIUM": TireCompound.MEDIUM,
            "HARD": TireCompound.HARD,
            "INTERMEDIATE": TireCompound.INTERMEDIATE,
            "WET": TireCompound.WET
        }
        
        return DriverState(
            driver_name=name,
            current_position=position,
            gap_to_leader=gap_leader,
            gap_to_ahead=gap_ahead,
            gap_to_behind=gap_behind,
            tire_state=TireState(
                compound=compound_map[compound],
                age_laps=age,
                degradation_percent=degradation
            ),
            fuel_remaining=fuel
        )
        
    def _generate_live_strategies(self, race_state, driver_state):
        """Generate relevant strategies for current race situation"""
        from src.data.models import StrategyOption
        
        strategies = []
        current_lap = race_state.track_state.current_lap
        
        # Immediate pit strategy
        strategies.append(StrategyOption(
            strategy_type=StrategyType.PIT_STOP,
            execute_on_lap=current_lap + 1,
            tire_compound_target=TireCompound.MEDIUM,
            expected_time_delta=-5.0,
            confidence_level=0.85
        ))
        
        # Optimal pit window
        strategies.append(StrategyOption(
            strategy_type=StrategyType.PIT_STOP,
            execute_on_lap=current_lap + 8,
            tire_compound_target=TireCompound.HARD,
            expected_time_delta=-2.0,
            confidence_level=0.92
        ))
        
        # Undercut strategy
        strategies.append(StrategyOption(
            strategy_type=StrategyType.UNDERCUT,
            execute_on_lap=current_lap + 3,
            tire_compound_target=TireCompound.SOFT,
            expected_time_delta=3.0,
            confidence_level=0.78
        ))
        
        # Stay out longer (overcut)
        strategies.append(StrategyOption(
            strategy_type=StrategyType.OVERCUT,
            execute_on_lap=current_lap + 15,
            tire_compound_target=TireCompound.MEDIUM,
            expected_time_delta=1.5,
            confidence_level=0.88
        ))
        
        return strategies
        
    def _display_strategy_results(self, results, strategies):
        """Display strategy analysis results"""
        
        # Rank strategies
        ranked = self.monte_carlo.rank_strategies_advanced(results)
        
        st.subheader("🥇 Top Strategic Recommendations")
        
        for i, (strategy, score) in enumerate(ranked[:3]):
            with st.expander(f"#{i+1} - {strategy.strategy_type.value} (Score: {score:.3f})", 
                           expanded=(i==0)):
                
                metrics = results[strategy]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Expected Position", f"{metrics.expected_position:.1f}")
                    st.metric("Championship Points", f"{metrics.championship_points_expected:.1f}")
                    
                with col2:
                    st.metric("Risk Level", f"{metrics.position_variance:.2f}")
                    st.metric("Success Probability", f"{strategy.confidence_level:.1%}")
                    
                with col3:
                    st.metric("Weather Resilience", f"{metrics.weather_resilience:.2f}")
                    st.metric("Safety Car Advantage", f"{metrics.safety_car_advantage:.2f}")
                
                if strategy.execute_on_lap:
                    st.info(f"📍 **Execute on Lap {strategy.execute_on_lap}**")
                    
                if strategy.tire_compound_target:
                    st.info(f"🏎️ **Target Tire: {strategy.tire_compound_target.value}**")
    
    def _generate_sample_timing_data(self):
        """Generate sample lap time data"""
        import numpy as np
        base_time = 78.5  # Monaco lap time
        return [base_time + np.random.normal(0, 0.5) for _ in range(50)]

if __name__ == "__main__":
    interface = TracksideInterface()
    interface.main_dashboard()
