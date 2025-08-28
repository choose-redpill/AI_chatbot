#!/usr/bin/env python3
"""
Setup script for Rural Renewable Energy Chatbot
This script handles installation and initial configuration
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def install_system_dependencies():
    """Install system-level dependencies for audio processing"""
    system = platform.system().lower()
    
    print("🔧 Installing system dependencies...")
    
    try:
        if system == "linux":
            # Ubuntu/Debian
            subprocess.run([
                "sudo", "apt-get", "update"
            ], check=True)
            subprocess.run([
                "sudo", "apt-get", "install", "-y",
                "portaudio19-dev", "python3-pyaudio", "espeak", "espeak-data"
            ], check=True)
            print("✅ Linux dependencies installed")
            
        elif system == "darwin":  # macOS
            # Check if Homebrew is installed
            try:
                subprocess.run(["brew", "--version"], check=True, capture_output=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("❌ Homebrew not found. Please install Homebrew first:")
                print("   /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
                return False
            
            subprocess.run(["brew", "install", "portaudio"], check=True)
            print("✅ macOS dependencies installed")
            
        elif system == "windows":
            print("⚠️  Windows detected. Please manually install:")
            print("   1. Microsoft Visual C++ 14.0 or greater")
            print("   2. PyAudio wheel from https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio")
            
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing system dependencies: {e}")
        return False

def install_python_dependencies():
    """Install Python packages"""
    print("📦 Installing Python dependencies...")
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "--upgrade", "pip"
        ], check=True)
        
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True)
        
        print("✅ Python dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing Python dependencies: {e}")
        return False

def setup_ollama():
    """Setup Ollama and download required models"""
    print("🤖 Setting up Ollama...")
    
    try:
        # Check if Ollama is installed
        try:
            result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
            print(f"✅ Ollama found: {result.stdout.strip()}")
        except FileNotFoundError:
            print("❌ Ollama not found. Please install Ollama:")
            print("   Visit: https://ollama.ai/download")
            return False
        
        # Download Mistral model
        print("📥 Downloading Mistral model (this may take a while)...")
        result = subprocess.run([
            "ollama", "pull", "mistral"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Mistral model downloaded successfully")
        else:
            print(f"⚠️  Warning: Could not download Mistral model: {result.stderr}")
            print("   The chatbot will use fallback mode")
        
        return True
    except Exception as e:
        print(f"⚠️  Ollama setup warning: {e}")
        print("   The chatbot will use fallback mode")
        return True

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    directories = ["docs", "vector_db", "logs"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"   Created: {directory}/")
    
    print("✅ Directories created")

def create_config_file():
    """Create configuration file"""
    print("⚙️  Creating configuration file...")
    
    config_content = """# Renewable Energy Chatbot Configuration

# LLM Settings
LLM_MODEL=mistral
LLM_TEMPERATURE=0.3
MAX_TOKENS=512

# RAG Settings
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K_RETRIEVAL=3

# Voice Settings
VOICE_ENABLED=true
TTS_RATE=150
TTS_VOLUME=0.8

# UI Settings
THEME=dark
LANGUAGE=english
ENABLE_VOICE_UI=true

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/chatbot.log

# Vector Database
VECTOR_DB_PATH=./vector_db
COLLECTION_NAME=renewable_energy_enhanced
"""
    
    with open(".env", "w") as f:
        f.write(config_content)
    
    print("✅ Configuration file created (.env)")

def run_initial_setup():
    """Run initial setup and test"""
    print("🧪 Running initial setup test...")
    
    try:
        # Test import of main modules
        import streamlit
        import chromadb
        import langchain
        print("✅ Core modules imported successfully")
        
        # Create sample documents
        print("📄 Creating sample documents...")
        from chatbot import RenewableEnergyChatbot
        chatbot = RenewableEnergyChatbot()
        chatbot.create_comprehensive_dataset()
        print("✅ Sample documents created")
        
        print("🎉 Setup completed successfully!")
        print("\n📚 Next steps:")
        print("   1. Run: streamlit run chatbot.py")
        print("   2. Open your browser to the displayed URL")
        print("   3. Start asking questions about renewable energy!")
        
        return True
    except Exception as e:
        print(f"❌ Setup test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🌱 Rural Renewable Energy Chatbot Setup")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version.split()[0]} detected")
    
    # Setup steps
    steps = [
        ("Creating directories", create_directories),
        ("Installing system dependencies", install_system_dependencies),
        ("Installing Python dependencies", install_python_dependencies),
        ("Setting up Ollama", setup_ollama),
        ("Creating configuration", create_config_file),
        ("Running initial setup", run_initial_setup)
    ]
    
    for step_name, step_func in steps:
        print(f"\n🔄 {step_name}...")
        if not step_func():
            print(f"❌ Failed: {step_name}")
            print("   Please check the error messages above and try again.")
            sys.exit(1)
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\n🚀 To start the chatbot, run:")
    print("   streamlit run chatbot.py")

if __name__ == "__main__":
    main()
