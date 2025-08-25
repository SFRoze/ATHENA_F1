"""
ATHENA F1 - Monaco GP Demo Scenario
Demonstrates strategy analysis for the challenging Monaco circuit.
"""

import asyncio
import sys
import os

# Add src to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.strategy_engine import StrategyEngine
from data.race_simulator import RaceSimulator
from data.models import TireCompound, WeatherCondition
from rich.console import Console
from rich.panel import Panel


async def run_monaco_demo():
    """Run Monaco GP strategy demonstration"""
    console = Console()
    
    console.print(Panel.fit(
        "🏁 ATHENA F1 - Monaco GP Strategy Demo",
        style="bold magenta"
    ))
    
    # Initialize Monaco simulation
    simulator = RaceSimulator("Monaco", total_laps=78)
    strategy_engine = StrategyEngine()
    
    # Initialize race with Monaco-specific challenges
    console.print("🏎️ Initializing Monaco GP simulation...")
    race_state = simulator.initialize_race()
    
    # Modify some drivers to have strategic scenarios
    # Make driver 2 (Hamilton) have high tire degradation for undercut scenario
    race_state.drivers[1].tire_state.degradation_percent = 65.0
    race_state.drivers[1].tire_state.age_laps = 25
    
    # Make driver 3 (Leclerc) have fresh tires but poor position
    race_state.drivers[2].tire_state.degradation_percent = 10.0
    race_state.drivers[2].tire_state.age_laps = 5
    race_state.drivers[2].current_position = 8  # Further back
    
    # Set lap 35 for mid-race strategy decisions
    race_state.track_state.current_lap = 35
    
    await strategy_engine.start()
    strategy_engine.update_race_state(race_state)
    
    console.print("\n📊 Analyzing strategies for key drivers...\n")
    
    # Analyze strategy for multiple drivers
    key_drivers = [
        ("driver_1", "Max Verstappen - Race Leader"),
        ("driver_2", "Lewis Hamilton - High Tire Degradation"),
        ("driver_3", "Charles Leclerc - Recovery Drive")
    ]
    
    for driver_id, description in key_drivers:
        console.print(Panel(f"Analyzing: {description}", style="cyan"))
        
        try:
            recommendation = await strategy_engine.analyze_strategy(driver_id)
            
            # Display detailed analysis
            driver = race_state.get_driver(driver_id)
            if driver:
                console.print(f"📍 Current Position: P{driver.current_position}")
                console.print(f"🏁 Gap to Leader: {driver.gap_to_leader:.1f}s")
                console.print(f"🔧 Tire: {driver.tire_state.compound.value} ({driver.tire_state.age_laps} laps)")
                console.print(f"📉 Degradation: {driver.tire_state.degradation_percent:.1f}%")
                console.print()
                
                # Primary recommendation
                primary = recommendation.primary_option
                console.print(f"🎯 PRIMARY STRATEGY: {primary.strategy_type.value.upper()}")
                console.print(f"✅ Success Probability: {primary.probability_success:.0%}")
                console.print(f"⏱️ Expected Time Impact: {primary.estimated_time_gain:+.1f}s")
                console.print(f"🎲 Risk Level: {primary.risk_level.upper()}")
                console.print(f"💭 Description: {primary.description}")
                console.print()
                
                # Reasoning
                console.print(f"🧠 Analysis: {recommendation.reasoning}")
                console.print(f"🎯 Confidence: {recommendation.confidence_score:.0%}")
                
                # Show alternatives
                if recommendation.alternative_options:
                    console.print("\n📋 Alternative Strategies:")
                    for i, alt in enumerate(recommendation.alternative_options[:2], 1):
                        console.print(f"  {i}. {alt.strategy_type.value.title()} - "
                                    f"{alt.probability_success:.0%} success, "
                                    f"{alt.estimated_time_gain:+.1f}s impact")
                
            console.print("\n" + "="*60 + "\n")
            
        except Exception as e:
            console.print(f"❌ Error analyzing {driver_id}: {e}")
            
    # Simulate some strategy scenarios
    console.print(Panel("🎮 Simulating Strategic Scenarios", style="yellow"))
    
    # Scenario 1: Safety Car deployment
    console.print("🚗 SCENARIO: Safety Car deployed!")
    race_state.track_state.safety_car_deployed = True
    strategy_engine.update_race_state(race_state)
    
    safety_car_recommendation = await strategy_engine.analyze_strategy("driver_1")
    console.print(f"Leader strategy during SC: {safety_car_recommendation.primary_option.strategy_type.value}")
    console.print(f"Reasoning: {safety_car_recommendation.reasoning}\n")
    
    # Scenario 2: Rain threat
    console.print("🌧️ SCENARIO: Rain probability increases to 60%!")
    race_state.track_state.rain_probability = 0.6
    race_state.track_state.safety_car_deployed = False
    strategy_engine.update_race_state(race_state)
    
    rain_recommendation = await strategy_engine.analyze_strategy("driver_2")
    console.print(f"Strategy with rain threat: {rain_recommendation.primary_option.strategy_type.value}")
    console.print(f"Tire recommendation: {rain_recommendation.primary_option.tire_compound_target}")
    console.print(f"Reasoning: {rain_recommendation.reasoning}\n")
    
    await strategy_engine.stop()
    
    console.print(Panel(
        "Demo completed! ATHENA F1 analyzed multiple strategic scenarios\n"
        "for the Monaco GP, considering track position, tire degradation,\n"
        "weather conditions, and safety car deployment.",
        title="🏁 Monaco GP Demo Summary",
        style="green"
    ))


if __name__ == "__main__":
    asyncio.run(run_monaco_demo())
