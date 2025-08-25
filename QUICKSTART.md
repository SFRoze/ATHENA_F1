# ATHENA F1 - Quick Start Guide

🏎️ **Get your F1 Live Race Commentary Strategy Assistant running in minutes!**

## Prerequisites

- Python 3.9 or higher
- Windows PowerShell or Command Prompt

## Installation

1. **Clone or download** the ATHENA F1 project to your local machine.

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Create logs directory**:
   ```bash
   mkdir logs
   ```

## Running ATHENA F1

### Basic Usage

Run the main application with default settings (Silverstone GP):
```bash
cd src
python main.py
```

### Custom Race Configuration

Specify different parameters:
```bash
python main.py --track Monaco --laps 78 --speed 2.0
```

**Parameters:**
- `--track`: Circuit name (Monaco, Silverstone, Monza, etc.)
- `--laps`: Number of race laps
- `--speed`: Simulation speed multiplier (2.0 = 2x speed)
- `--driver`: Target driver ID for strategy analysis

### Demo Scenarios

Run pre-configured demo scenarios:
```bash
# Monaco GP strategic scenarios
cd examples
python monaco_demo.py
```

## What You'll See

1. **Race Status Table**: Live positions, tire states, and gaps
2. **Strategy Recommendations**: Real-time strategic analysis
3. **Track Conditions**: Weather, temperature, safety car status
4. **Alternative Strategies**: Multiple strategic options with probabilities

## Sample Output

```
🏁 Silverstone GP - Lap 25/52
┏━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━┳━━━━━━┳━━━━━━━━━┓
┃ Pos ┃ Driver          ┃ Team           ┃ Tire  ┃ Age ┃ Deg% ┃ Gap     ┃
┡━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━╇━━━━━━╇━━━━━━━━━┩
│ 1   │ Max Verstappen  │ Red Bull       │ Medium│ 12  │ 18.5%│ Leader  │
│ 2   │ Lewis Hamilton  │ Mercedes       │ Medium│ 15  │ 25.2%│ +8.3s   │
└─────┴─────────────────┴────────────────┴───────┴─────┴──────┴─────────┘

Strategy Recommendation - Lap 25
🎯 Target: Lewis Hamilton (P2)
📋 Primary Strategy: Undercut
📊 Success Probability: 70%
⏱️ Expected Time Gain: +3.2s
🧠 Confidence: 85%

💡 Reasoning: Hamilton is P2 with 25.2% tire degradation. Undercut opportunity 
detected - fresh tires should provide 3.2s advantage...
```

## Key Features Demonstrated

- **Real-time Strategy Analysis**: Continuously evaluates optimal moves
- **Tire Degradation Modeling**: Physics-based tire performance prediction
- **Undercut/Overcut Detection**: Identifies strategic pit stop opportunities  
- **Weather Impact Analysis**: Adapts strategy for changing conditions
- **Safety Car Optimization**: Capitalizes on safety car periods
- **Multi-driver Analysis**: Compares strategies across the field

## Controls

- **Ctrl+C**: Stop the simulation gracefully
- **Monitor the console**: Strategy updates appear every 10 seconds
- **Race state updates**: Every 5 seconds during simulation

## Customization

### Change Target Driver
Focus analysis on a different driver:
```bash
python main.py --driver driver_3  # Focuses on 3rd driver (usually Leclerc)
```

### Different Tracks
```bash
python main.py --track Monza      # High-speed strategy
python main.py --track Monaco     # Street circuit strategy
```

### Faster Simulation
```bash
python main.py --speed 5.0        # 5x speed for quick demos
```

## What Makes ATHENA F1 Special

1. **Real-time Decision Making**: Continuously calculates optimal strategies
2. **Multi-factor Analysis**: Considers tires, fuel, position, weather, competitors
3. **Probabilistic Outcomes**: Provides success probabilities for each strategy
4. **Live Commentary Style**: Designed for race commentary and analysis
5. **Employer-friendly**: Demonstrates advanced algorithmic thinking and F1 knowledge

## Next Steps

- Experiment with different race scenarios
- Analyze the strategy recommendations
- Review the code to understand the algorithms
- Consider extending with additional features

**Ready to become an F1 strategy engineer?** 🏁

For technical details, see the full documentation in `docs/`.
