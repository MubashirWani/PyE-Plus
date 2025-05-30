# 🚀 PyE+ Setup & Usage Guide

This guide complements the README by focusing on **system setup, configuration paths**, and **how to properly run the PyE+ GUI-based simulation and optimization environment**.

---

## 🧰 System Requirements

- **EnergyPlus**: v23.2.0 (with Python plugin enabled)
- **Python**: v3.12 (use a virtual environment)
- Use the following to install required Python packages:
  ```bash
  pip install -r requirements.txt
  ```

---

## 📁 File & Folder Layout

Place all core files (scripts, IDFs, icons, JSONs) inside a single directory — e.g., `Scripts/` inside `.venv/`. This ensures file references within the code resolve correctly.

> Example path: `C:\Users\yourname\PyCharmProjects\PyEPlus\.venv\Scripts\`

---

## 🛠️ Configuration Checklist

### 1. ✅ `Pilot_Interface.py`

Update the following absolute paths to match your system:

```python
baseline_idf_path = '.../Baseline_IDF_Modified.idf'
weather_file = '.../NZL_Wellington.Wellington.934360_IWEC.epw'
idf_path = '.../RealBuilding_Test_MoreParameters_Wellington_Python.idf'
```

### 2. ✅ `Graphics.py`

Find and update:

```python
folder_path_var = tk.StringVar(value="C:\path\to\your\Scripts")
```

### 3. ✅ `launch_graphics.bat`

Update the working directory:

```bat
cd C:\path\to\your\Scripts
```

### 4. ✅ (Optional) `activate.bat`

Make sure the virtual environment path is correct:

```bat
set "VIRTUAL_ENV=C:\path\to\.venv"
```

---

## 🖥️ Desktop Shortcut (Optional)

To launch with a double-click:

1. Right-click the `launch_graphics.bat` → Create shortcut.
2. Move it to Desktop.
3. Set **Target** and **Start in** paths to your `Scripts/` folder.
4. Click **Change Icon** and point to `pye__logo_rX8_icon.ico`.

---

## 🧪 Running the App

1. Launch `Graphics.py` (via shortcut or manually).
2. Browse to your `Scripts` folder.
3. Set the **number of parameter vectors** (default is 3).
4. Tick **“Run Baseline Simulation?”** on first run — generates the baseline `.json` file.
5. Hit **Start** and monitor progress via command window or GUI.

---

## 🧩 Model Files: Advanced Tip

### Included:
- `Baseline_IDF_Modified.idf`
- `RealBuilding_Test_MoreParameters_Wellington_Python.idf`

> 💡 The second file demonstrates how to use **EnergyPlus's Python Plugin** to control zone temperatures, plug loads, lighting, and infiltration — driven by logic in `SixSigma.py` and `DataXC.py`.

Explore the plugin-defined objects inside this IDF for deeper integration insights.

---

## 🧠 Environment Variables (Recommended)

Add to PATH (System Environment Variables):

- `...\.venv\Scripts`
- `C:\EnergyPlusV23-2-0\pyenergyplus`
- `C:\EnergyPlusV23-2-0`

---

## 🔚 Final Note

Ensure that your project folder structure mirrors the paths used inside the code. This ensures reproducibility across machines and proper integration between Python, EnergyPlus, and the GUI.