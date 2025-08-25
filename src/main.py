"""
ATHENA F1 - Main Application
Entry point for the F1 Live Race Commentary Strategy Assistant.
"""

import asyncio
import signal
import sys
from typing import Optional
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich import box

from core.strategy_engine import StrategyEngine
from data.race_simulator import RaceSimulator
from data.models import RaceState


class AthenaF1Application:
    """
    Main application class that orchestrates the F1 strategy assistant
    """
    
    def __init__(self, track_name: str = "Silverstone", total_laps: int = 52):
        self.console = Console()
        self.strategy_engine = StrategyEngine()
        self.race_simulator = RaceSimulator(track_name, total_laps)
        self.is_running = False
        self.target_driver_id = "driver_1"  # Default to first driver
        
    async def start(self):
        """Start the ATHENA F1 system"""
        self.console.print(Panel.fit(
            "🏎️  ATHENA F1 - Live Race Commentary Strategy Assistant",
            style="bold blue"
        ))
        
        # Initialize race simulation
        self.console.print("Initializing race simulation...")
        race_state = self.race_simulator.initialize_race()
        
        # Start strategy engine
        await self.strategy_engine.start()
        self.strategy_engine.update_race_state(race_state)
        
        # Add callback for race updates
        self.race_simulator.add_update_callback(self._on_race_update)
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.is_running = True
        
        # Start race simulation
        self.console.print("Starting race simulation...")
        simulation_task = asyncio.create_task(
            self.race_simulator.start_simulation(update_interval=3.0)
        )
        
        # Start strategy analysis loop
        strategy_task = asyncio.create_task(self._strategy_analysis_loop())
        
        # Start user interface loop
        ui_task = asyncio.create_task(self._user_interface_loop())
        
        # Wait for tasks to complete
        try:
            await asyncio.gather(simulation_task, strategy_task, ui_task)
        except KeyboardInterrupt:
            await self.stop()
            
    async def stop(self):
        """Stop the ATHENA F1 system"""
        self.console.print("\n🏁 Shutting down ATHENA F1...")
        
        self.is_running = False
        self.race_simulator.stop_simulation()
        await self.strategy_engine.stop()
        
        self.console.print("ATHENA F1 stopped successfully")
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.console.print(f"\nReceived signal {signum}, initiating shutdown...")
        asyncio.create_task(self.stop())
        
    async def _strategy_analysis_loop(self):
        """Main strategy analysis loop"""
        while self.is_running:
            try:
                if self.strategy_engine.current_race_state:
                    # Generate strategy recommendation for target driver
                    recommendation = await self.strategy_engine.analyze_strategy(
                        self.target_driver_id
                    )
                    
                    # Display recommendation in console
                    self._display_strategy_recommendation(recommendation)
                    
                # Wait before next analysis
                await asyncio.sleep(10.0)  # Analyze every 10 seconds
                
            except Exception as e:
                logger.error(f"Strategy analysis error: {e}")
                await asyncio.sleep(5.0)
                
    async def _user_interface_loop(self):
        """Simple user interface for interaction"""
        while self.is_running:
            try:
                await asyncio.sleep(5.0)  # Update UI every 5 seconds
                
                if self.strategy_engine.current_race_state:
                    self._display_race_state()
                    
            except Exception as e:
                logger.error(f"UI error: {e}")
                await asyncio.sleep(2.0)
                
    def _on_race_update(self, race_state: RaceState):
        """Called when race state is updated"""
        # Update strategy engine with new race state
        self.strategy_engine.update_race_state(race_state)
        
    def _display_race_state(self):
        """Display current race state in console"""
        race_state = self.strategy_engine.current_race_state
        if not race_state:
            return
            
        # Create race status table
        table = Table(title=f"🏁 {race_state.track_state.track_name} GP - Lap {race_state.track_state.current_lap}/{race_state.track_state.total_laps}")
        
        table.add_column("Pos", style="cyan", no_wrap=True)
        table.add_column("Driver", style="white")
        table.add_column("Team", style="dim")
        table.add_column("Tire", style="green")
        table.add_column("Age", style="yellow")
        table.add_column("Deg%", style="red")
        table.add_column("Gap", style="blue")
        
        for driver in race_state.leaderboard[:10]:  # Top 10 only
            gap_str = f"+{driver.gap_to_leader:.1f}s" if driver.gap_to_leader > 0 else "Leader"
            tire_age = driver.tire_state.age_laps
            degradation = f"{driver.tire_state.degradation_percent:.1f}%"
            
            table.add_row(
                str(driver.current_position),
                driver.driver_name,
                driver.team,
                driver.tire_state.compound.value.title(),
                str(tire_age),
                degradation,
                gap_str
            )
            
        # Clear screen and display
        self.console.clear()
        self.console.print(table)
        
        # Display weather and track conditions
        weather_info = (
            f"Weather: {race_state.track_state.weather.value.title()} | "
            f"Track Temp: {race_state.track_state.track_temperature}°C | "
            f"Safety Car: {'DEPLOYED' if race_state.track_state.safety_car_deployed else 'NO'}"
        )
        self.console.print(Panel(weather_info, title="Track Conditions", style="dim"))
        
    def _display_strategy_recommendation(self, recommendation):
        """Display strategy recommendation"""
        target_driver = self.strategy_engine.current_race_state.get_driver(
            recommendation.target_driver
        )
        
        if not target_driver:
            return
            
        # Create strategy panel
        primary = recommendation.primary_option
        
        strategy_info = (
            f"🎯 Target: {target_driver.driver_name} (P{target_driver.current_position})\n"
            f"📋 Primary Strategy: {primary.strategy_type.value.title()}\n"
            f"📊 Success Probability: {primary.probability_success:.0%}\n"
            f"⏱️  Expected Time Gain: {primary.estimated_time_gain:+.1f}s\n"
            f"🧠 Confidence: {recommendation.confidence_score:.0%}\n\n"
            f"💡 Reasoning: {recommendation.reasoning}"
        )
        
        self.console.print(Panel(
            strategy_info,
            title=f"Strategy Recommendation - Lap {self.strategy_engine.current_race_state.track_state.current_lap}",
            style="green"
        ))
        
        # Show alternatives if available
        if recommendation.alternative_options:
            alt_table = Table(title="Alternative Strategies")
            alt_table.add_column("Strategy", style="cyan")
            alt_table.add_column("Success %", style="green")
            alt_table.add_column("Time Gain", style="yellow")
            alt_table.add_column("Risk", style="red")
            
            for alt in recommendation.alternative_options[:3]:  # Top 3 alternatives
                alt_table.add_row(
                    alt.strategy_type.value.title(),
                    f"{alt.probability_success:.0%}",
                    f"{alt.estimated_time_gain:+.1f}s",
                    alt.risk_level.title()
                )
                
            self.console.print(alt_table)
            
        self.console.print()  # Add spacing


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ATHENA F1 - Live Race Strategy Assistant")
    parser.add_argument("--track", default="Silverstone", help="Track name for simulation")
    parser.add_argument("--laps", type=int, default=52, help="Number of laps")
    parser.add_argument("--driver", default="driver_1", help="Target driver ID for strategy analysis")
    parser.add_argument("--speed", type=float, default=1.0, help="Simulation speed multiplier")
    
    args = parser.parse_args()
    
    # Create and start application
    app = AthenaF1Application(args.track, args.laps)
    app.target_driver_id = args.driver
    app.race_simulator.simulation_speed = args.speed
    
    try:
        await app.start()
    except KeyboardInterrupt:
        await app.stop()


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(
        "logs/athena_f1_{time}.log",
        rotation="1 day",
        retention="7 days",
        level="INFO"
    )
    logger.add(sys.stderr, level="WARNING")
    
    # Run application
    asyncio.run(main())
