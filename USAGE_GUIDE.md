# 🧪 PyE+ Usage Guide

This guide complements the Setup Guide and focuses on how to **run simulations, train surrogates, and optimize using PyE+** through its GUI.

---

## 🚀 Quick Launch

If you completed the setup successfully, launch the app by:
- Running `Graphics.py` manually **OR**
- Double-clicking the **PyE+ desktop shortcut** (linked to `launch_graphics.bat`)

The GUI will open and display multiple tabs, each corresponding to a stage of simulation or optimization.

---

## 🖥️ Main Page – Baseline Setup

This tab allows you to define the **baseline parameters and optimization bounds**.

### Steps:
1. Click **“Browse”** to select your `Scripts/` folder.
2. Use the provided input fields to review or modify baseline values.
3. Adjust **Min/Max bounds** for each parameter (used by NSGA-II).
4. Optional: click **Default Values** to auto-fill known defaults.
5. Enable **“Run Baseline Simulation?”** to ensure `energyplus_baseline.json` is created.

Then click **Start** to run the baseline.

---

## 🤖 NSGA-II Tab – Evolutionary Optimization

This tab handles true multi-objective optimization.

### Parameters:
- **Comfort bounds**: Min/Max temperatures and ideal temp (e.g., 18–28°C, ideal = 23°C)
- **Generations**: Number of optimization rounds (e.g., 10)
- **Population size**: Number of candidates per generation
- **Max Estimated Runs** = Population + (Generations × Population)

### Actions:
1. Enable **“Use NSGA-II for Optimization”**
2. Click **Reset Run Counter** to start fresh.
3. Click **“Run NSGA-II”** to begin.
4. Watch live Pareto Front evolve in the 3D plot.

> ℹ️ All simulations run in parallel with EnergyPlus in the background.

---

## 🧠 Machine Learning Tab – Surrogate-Assisted Optimization

This enables **surrogate modeling** to accelerate NSGA-II.

### Workflow:
1. Enable **“Use Surrogate Modeling”**
2. Set parameters:
   - Initial DOE samples (e.g., 100)
   - Infill Fraction (e.g., 10%)
   - Retrain interval (e.g., every 5 generations)
3. Select a surrogate type: `RandomForest`, `GaussianProcess`, or `MLPRegressor`
4. Click **Generate LHC Sample** to seed the training data.
5. Click **Train Surrogate** to build the model.

Once ready:
- Click **“Start SA-NSGA-II”** to launch surrogate-assisted optimization
- Watch the **RMSE vs. Generations** plot for surrogate quality

---

## 📊 Results

Key output files include:
- `pareto_front.json` – stores non-dominated solutions
- `energyplus_baseline.json` – baseline reference
- `training_data.json` – DOE archive used for surrogate
- `rmse_log.json` – RMSE history

Knee-point and final Pareto plots are saved in the working directory.

---

## 💡 Tips

- If `run_counter.txt` or `energyplus_baseline.json` are missing, re-run baseline.
- Surrogates work best with at least 30–50 true simulations.
- The LED icon on the NSGA-II tab indicates whether surrogate mode is active.
- Use **“Stop”** button on Main tab to interrupt simulation.

---

## 🧠 Want More?

Refer to:
- `SETUP_GUIDE.md` for environment and file prep
- `NSGA_II_Algorithm.py` and `Pilot_Interface.py` for algorithm logic
- `SixSigma.py` for EnergyPlusPlugin script structure