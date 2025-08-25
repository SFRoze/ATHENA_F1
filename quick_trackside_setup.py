#!/usr/bin/env python3
"""
ATHENA F1 - Quick Trackside Setup
One-command deployment for race day use
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def print_banner():
    """Display ATHENA F1 banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════╗
    ║                    🏁 ATHENA F1 🏁                        ║
    ║              Quick Trackside Setup v1.0                  ║
    ║         World-Class F1 Strategy Assistant                ║
    ╚══════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_python():
    """Verify Python installation"""
    print("🐍 Checking Python installation...")
    try:
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 9):
            print("❌ Python 3.9+ required. Current version:", sys.version)
            return False
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    except Exception as e:
        print(f"❌ Python check failed: {e}")
        return False

def install_dependencies():
    """Install required packages"""
    print("📦 Installing dependencies...")
    
    packages = [
        'streamlit>=1.28.0',
        'plotly>=5.15.0',
        'pandas>=2.0.0',
        'numpy>=1.24.0',
        'loguru>=0.7.0',
        'fastapi>=0.100.0',
        'uvicorn>=0.23.0'
    ]
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                         check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            return False
    
    print("✅ All dependencies installed successfully")
    return True

def verify_athena_structure():
    """Verify ATHENA F1 file structure"""
    print("📁 Verifying ATHENA F1 structure...")
    
    required_paths = [
        'src/algorithms/monte_carlo_strategy.py',
        'src/algorithms/neural_optimizer.py',
        'src/data/models.py',
        'trackside_interface.py'
    ]
    
    missing = []
    for path in required_paths:
        if not Path(path).exists():
            missing.append(path)
    
    if missing:
        print("❌ Missing required files:")
        for file in missing:
            print(f"   - {file}")
        return False
    
    print("✅ ATHENA F1 structure verified")
    return True

def create_config_file():
    """Create race day configuration file"""
    print("⚙️ Creating race day configuration...")
    
    config = """# ATHENA F1 Race Day Configuration
RACE_MODE=true
FAST_STARTUP=true
SIMULATION_COUNT=10000
ELITE_MODE_THRESHOLD=0.8
TRACKSIDE_PORT=8501
ENABLE_VOICE_COMMANDS=false
AUTO_REFRESH_SECONDS=30
DEBUG_MODE=false

# Data Sources
USE_LIVE_TIMING=false
USE_WEATHER_API=false
MANUAL_INPUT_MODE=true

# Performance
PARALLEL_PROCESSING=true
MAX_WORKERS=4
CACHE_STRATEGIES=true
"""
    
    with open('.env', 'w') as f:
        f.write(config)
    
    print("✅ Configuration file created")

def setup_trackside_shortcuts():
    """Create quick access scripts"""
    print("🚀 Creating trackside shortcuts...")
    
    # Windows batch file
    start_script = """@echo off
echo Starting ATHENA F1 Trackside Interface...
streamlit run trackside_interface.py --server.port 8501 --server.headless true
pause
"""
    
    with open('start_trackside.bat', 'w') as f:
        f.write(start_script)
    
    # Emergency backup script
    backup_script = """@echo off
echo ATHENA F1 Emergency Mode
echo Running reduced-complexity analysis...
python src/main.py --mode emergency --fast --output console
pause
"""
    
    with open('emergency_mode.bat', 'w') as f:
        f.write(backup_script)
    
    print("✅ Trackside shortcuts created")
    print("   - start_trackside.bat: Launch main interface")
    print("   - emergency_mode.bat: Backup calculation mode")

def run_system_test():
    """Quick system test"""
    print("🧪 Running system test...")
    
    try:
        # Test imports
        sys.path.append('src')
        from algorithms.monte_carlo_strategy import WorldClassMonteCarloStrategy
        from data.models import TireCompound, StrategyType
        
        # Quick initialization test
        mc = WorldClassMonteCarloStrategy()
        print("✅ Monte Carlo system: OK")
        
        print("✅ System test passed - Ready for race day!")
        return True
        
    except Exception as e:
        print(f"❌ System test failed: {e}")
        return False

def display_launch_instructions():
    """Show how to launch for race day"""
    instructions = """
    ╔══════════════════════════════════════════════════════════╗
    ║                   🏁 READY FOR RACE DAY! 🏁              ║
    ╠══════════════════════════════════════════════════════════╣
    ║                                                          ║
    ║  Quick Launch Options:                                   ║
    ║                                                          ║
    ║  1. 🖥️  MAIN INTERFACE (Recommended):                   ║
    ║     Double-click: start_trackside.bat                   ║
    ║     Or run: streamlit run trackside_interface.py        ║
    ║                                                          ║
    ║  2. ⚡ EMERGENCY MODE (Backup):                          ║
    ║     Double-click: emergency_mode.bat                    ║
    ║     Or run: python src/main.py --mode emergency         ║
    ║                                                          ║
    ║  3. 🌐 WEB ACCESS:                                       ║
    ║     Open browser to: http://localhost:8501              ║
    ║                                                          ║
    ║  📱 Mobile Access:                                       ║
    ║     Find your IP address, access from phone/tablet      ║
    ║     http://YOUR_IP:8501                                 ║
    ║                                                          ║
    ╠══════════════════════════════════════════════════════════╣
    ║  🏎️ RACE DAY CHECKLIST:                                 ║
    ║  □ Charge laptop battery (6+ hours)                     ║
    ║  □ Test internet connection                              ║
    ║  □ Have backup power supply                              ║
    ║  □ Know track-specific tire strategies                   ║
    ║  □ Practice using interface before race                  ║
    ╚══════════════════════════════════════════════════════════╝
    
    💡 PRO TIPS:
    - Pre-load common strategies before race start
    - Use ELITE MODE for critical late-race decisions
    - Have backup manual calculations ready
    - Test voice commands if using mobile
    
    🚀 ATHENA F1 is ready to give you the competitive edge!
    """
    print(instructions)

def main():
    """Main setup process"""
    print_banner()
    time.sleep(2)
    
    # System checks
    if not check_python():
        print("\n❌ Setup failed: Python requirements not met")
        input("Press Enter to exit...")
        return False
    
    if not verify_athena_structure():
        print("\n❌ Setup failed: Missing ATHENA F1 files")
        input("Press Enter to exit...")
        return False
    
    # Installation
    if not install_dependencies():
        print("\n❌ Setup failed: Could not install dependencies")
        input("Press Enter to exit...")
        return False
    
    # Configuration
    create_config_file()
    setup_trackside_shortcuts()
    
    # Testing
    if not run_system_test():
        print("\n⚠️ Warning: System test failed, but you can still try to run")
    
    # Success!
    print("\n🏆 ATHENA F1 Trackside Setup Complete!")
    time.sleep(1)
    display_launch_instructions()
    
    # Auto-launch option
    launch = input("\n🚀 Launch trackside interface now? (y/n): ").lower().strip()
    if launch in ['y', 'yes']:
        print("Starting ATHENA F1 Trackside Interface...")
        try:
            subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'trackside_interface.py', 
                          '--server.port', '8501'])
        except KeyboardInterrupt:
            print("\n👋 ATHENA F1 interface stopped. Ready for next race!")
    
    return True

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Setup cancelled. Run again when ready for race day!")
    except Exception as e:
        print(f"\n💥 Unexpected error during setup: {e}")
        print("Please check your system and try again.")
    finally:
        input("\nPress Enter to exit...")
