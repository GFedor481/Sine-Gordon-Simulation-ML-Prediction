# Sine-Gordon Simulation and Machine Learning Localization Prediction

This repository contains the implementation and research for predicting energy localization in the Sine-Gordon equation using machine learning techniques. The work led to a published paper on localization prediction in nonlinear wave systems.

## 📖 Research Overview

The Sine-Gordon equation describes the dynamics of coupled pendulums and exhibits rich nonlinear phenomena including energy localization. This project develops machine learning models to predict when and where energy localization occurs in these systems, which has applications in understanding nonlinear wave dynamics, soliton behavior, and energy transport in physical systems.

## 🎯 Project Goals

- **Simulate** the Sine-Gordon equation with different instability types
- **Generate** large datasets of simulation trajectories
- **Train** machine learning models to predict localization events
- **Analyze** the effectiveness of different ML approaches for nonlinear dynamics prediction

## 📁 Repository Structure

```
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── .gitignore                         # Git ignore rules
├── sine_gordon_project.ipynb          # Main project notebook
├── sine_gordon_project copy 2 (1).ipynb # Alternative/backup notebook
├── sine_gordon_dataset_100k.h5        # Large dataset (3GB - not tracked by git)
├── sine_gordon_model.h5               # Trained ML model (not tracked by git)
├── sine_gordon_dataset.log            # Dataset generation log
├── error_histogram.png                # Model performance visualization
└── sine_Gordon_localization_prediction (3).pdf  # Published paper
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- Jupyter Notebook or JupyterLab
- Required packages (see `requirements.txt`)

### Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd Sine-Gordon-Simulation-ML-Prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Launch Jupyter:
```bash
jupyter notebook
```

4. Open `sine_gordon_project.ipynb` to explore the main project

## 🔬 Key Components

### 1. Sine-Gordon Simulation
The core simulation implements the coupled pendulum system:
- **Parameters**: Coupling strength (g), external field (h), amplitude (A)
- **Instability Types**:
  - Type I: h = -0.15, A = 1.25
  - Type II: h = 0, A = 1.2  
  - Type III: h = +0.5, A = 1.75

### 2. Localization Detection
Functions to identify energy localization events:
- Energy threshold-based detection
- Time and site localization prediction
- Boxcar smoothing for noise reduction

### 3. Machine Learning Pipeline
- Dataset generation from simulations
- Model training and validation
- Performance analysis and visualization

## 📊 Results and Visualizations

The project includes:
- Error histograms showing model performance
- Energy distribution plots
- Localization prediction accuracy metrics

## 📚 Published Paper

The research findings are documented in: `sine_Gordon_localization_prediction (3).pdf`

## 🤝 Contributing

This is a research project, but contributions are welcome:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request


## 👥 Authors

Adrian Mittal, Fedya Grishanov, Anne Pham, Noah Lape, L.Q. English

## 🙏 Acknowledgments

Hana Zwick, Dr. Robert Malkin, Prof. Lulu Wang

## 📞 Contact

grishanovf@gmail.com

---

**Note**: Large data files (`.h5` files) are not tracked by git due to size constraints. You may need to regenerate these or download them separately for full project functionality.
