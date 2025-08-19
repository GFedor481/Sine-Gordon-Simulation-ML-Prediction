# Project Structure and Organization

This document provides a detailed breakdown of the project structure and explains the purpose of each component.

## 📁 File Organization

### Core Project Files
- **`sine_gordon_project.ipynb`** - Main project notebook containing the complete implementation
- **`sine_gordon_project copy 2 (1).ipynb`** - Alternative/backup version of the main notebook
- **`sine_Gordon_localization_prediction (3).pdf`** - Published research paper

### Data and Models
- **`sine_gordon_dataset_100k.h5`** - Large dataset containing 100,000 simulation trajectories (3GB)
- **`sine_gordon_model.h5`** - Trained machine learning model (303MB)
- **`sine_gordon_dataset.log`** - Log file recording dataset generation process

### Visualizations and Results
- **`error_histogram.png`** - Visualization of model prediction errors
- **`sine_gordon_dataset.log`** - Dataset generation logs

### Configuration and Documentation
- **`README.md`** - Main project documentation
- **`requirements.txt`** - Python package dependencies
- **`.gitignore`** - Git ignore rules
- **`PROJECT_STRUCTURE.md`** - This file

## 🔬 Code Organization

### 1. Simulation Functions
```python
def f(x, t, g, h):
    # Core Sine-Gordon equation implementation
    # Returns derivatives for coupled pendulum system
```

### 2. Localization Detection
```python
def findLocalization(entotal, energy):
    # Detects energy localization events
    # Returns time and site of localization
```

### 3. Data Processing
```python
def boxcar(energy, snap, kernel_size=10):
    # Applies boxcar smoothing to reduce noise
    # Returns smoothed energy distribution
```

## 📊 Data Flow

1. **Simulation Generation**
   - Set parameters (g, h, A) for different instability types
   - Run ODE integration for coupled pendulum system
   - Calculate energy distributions over time

2. **Localization Detection**
   - Apply energy threshold criteria
   - Identify localization events
   - Record timing and spatial information

3. **Machine Learning Pipeline**
   - Generate training datasets from simulations
   - Train models to predict localization
   - Validate and analyze performance

## 🎯 Key Parameters

### Instability Types
- **Type I Instability**: h = -0.15, A = 1.25
- **Type II Instability**: h = 0, A = 1.2
- **Type III Instability**: h = +0.5, A = 1.75

### Simulation Parameters
- **Coupling Strength (g)**: 0.75
- **Time Final (tf)**: 150
- **Time Step (dt)**: 0.5
- **Number of Pendulums**: 100

## 🔄 Workflow

1. **Setup**: Install dependencies from `requirements.txt`
2. **Explore**: Open main notebook in Jupyter
3. **Simulate**: Run simulations with different parameters
4. **Analyze**: Examine localization patterns and ML predictions
5. **Visualize**: Generate plots and performance metrics

## 📝 Notes for Contributors

- Large data files (`.h5`) are excluded from git due to size
- The main notebook contains the complete implementation
- Backup notebook provides alternative approaches
- Log files help track dataset generation progress
- Visualizations demonstrate key results and model performance

## 🚨 Important Considerations

- **Data Size**: The full dataset is 3GB and not tracked by git
- **Model Size**: Trained models are 300MB+ and not tracked by git
- **Regeneration**: Large files may need to be regenerated locally
- **Dependencies**: Ensure all packages from `requirements.txt` are installed
