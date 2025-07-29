# PyE+

**A Python–EnergyPlus Co-simulation and Optimization Framework with Expedited Computation using ML-based Surrogate Modeling**

---

## 🔍 Overview

**PyE+** is a flexible and extensible framework designed to integrate **EnergyPlus** with **Python** to enable co-simulation, multi-objective optimization (via NSGA-II), and machine learning–based surrogate modeling for expedited evaluation of building energy performance and occupant comfort.

---

## ⚙️ Features

- ✅ Co-simulation interface between **Python and EnergyPlus**
- ✅ Parametric control of **U-values**, **lighting/plug loads**, **infiltration**, and **HVAC setpoints**
- ✅ Integrated **NSGA-II** optimization (with optional surrogate acceleration)
- ✅ GUI built with **Tkinter** for user-friendly execution
- ✅ Live plotting of Pareto front and RMSE trends
- ✅ Citation-aware licensing and `CITATION.cff` metadata

---

## 🧪 Dependencies

- Python ≥ 3.8
- EnergyPlus ≥ v9.5
- [eppy](https://github.com/santoshphilip/eppy)
- [DEAP](https://github.com/DEAP/deap)
- scikit-learn
- matplotlib, numpy, pandas
- tkinter (standard in Python)

---

## 🚀 Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/MubashirWani/PyE-Plus.git
   cd PyE-Plus
2. Set your EnergyPlus path and ensure Energy+.idd is accessible.

3. Activate your virtual environment and install dependencies:

   - pip install -r requirements.txt
     
4. Run the GUI:

   - python Graphics.py

---

## 📂 Included IDF Files

Two EnergyPlus Input Data Files (IDFs) are included in this package:

- `Baseline_IDF_Modified.idf`
- `RealBuilding_Test_MoreParameters_Wellington_Python.idf`

These files provide reference building models for baseline and measure simulations, respectively.

> 🧠 **Tip:** Users are encouraged to explore the second IDF file—especially the Python Plugin and related object definitions—to better understand how PyE+ leverages the EnergyPlus Python API for real-time control and optimization.

---

📄 License

> This project is licensed under the MIT License.

---

🧠 Acknowledgements
   - Developed as part of research on building energy optimization and smart grid integration.

   - EnergyPlus® is a product of the U.S. Department of Energy.

---

## 📚 Citation

**Academic Use and Citation Notice**  
If you use this software, in whole or in part, for academic research, publication, or derivative work, **you are kindly requested to cite the original creator**:

> Mubashir Hussain Wani. _"PyE+: A Python-EnergyPlus Co-simulation and Optimization Framework with Expedited Computation using ML-based Surrogate Modeling"_, 2025. Available: https://doi.org/10.5281/zenodo.15877753

This is not a legal requirement but a scholarly courtesy.
