# Quick Start Guide

Get up and running with the Sine-Gordon ML project in under 10 minutes!

## ⚡ Quick Setup

### 1. Clone and Navigate
```bash
git clone <your-repo-url>
cd Sine-Gordon-Simulation-ML-Prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch Jupyter
```bash
jupyter notebook
```

### 4. Open the Main Notebook
Open `sine_gordon_project.ipynb` and start exploring!

## 🚀 First Steps

### Run a Basic Simulation
1. Navigate to the simulation section in the notebook
2. Set your parameters:
   ```python
   g = 0.75      # Coupling strength
   h = 0.5       # External field (Type III instability)
   A = 1.75      # Amplitude
   ```
3. Execute the simulation cells
4. View the energy distribution plots

### Test Localization Detection
1. Find the `findLocalization` function
2. Apply it to your simulation results
3. Check when and where energy localizes

## 📊 What You'll See

- **Energy Plots**: Visualize how energy distributes across the pendulum chain
- **Localization Events**: Identify when energy concentrates at specific sites
- **ML Predictions**: See how well the model predicts localization timing

## 🔧 Common Issues

### Missing Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Large Files Missing
The large `.h5` files are not in git. You can:
- Regenerate them by running the full simulation pipeline
- Use smaller test datasets for initial exploration
- Contact the authors for data access

### Jupyter Issues
```bash
pip install jupyter --upgrade
jupyter notebook --generate-config
```

## 🎯 Next Steps

1. **Experiment with Parameters**: Try different g, h, A values
2. **Analyze Results**: Examine localization patterns
3. **Train Models**: Use the ML pipeline to predict localization
4. **Read the Paper**: Check `sine_Gordon_localization_prediction (3).pdf`

## 📞 Need Help?

- Check the main `README.md` for detailed documentation
- Review `PROJECT_STRUCTURE.md` for project organization
- Look at the notebook comments for code explanations
- Examine the published paper for theoretical background

---

**Happy Exploring!** 🚀
