#Graphics.py
import json
import traceback
import os
import subprocess
import time
import threading
import tkinter as tk
import Pilot_Interface
from Pilot_Interface import co_simulate
from Pilot_Interface import run_baseline_simulation
from tkinter import filedialog, ttk, messagebox
from NSGA_II_Algorithm import nsga_ii
from NSGA_II_Algorithm import THETA_MIN, THETA_MAX, nsga_ii
from ml_surrogates import make_surrogate
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Initialize the root window
root = tk.Tk()
root.title("Co-simulation Setup")
root.geometry("1060x900")
root.rowconfigure(0, weight=1)
root.columnconfigure(0, weight=1)

# training archive
X_train = []
Y_train = []
surrogate = None
true_runs_count = 0

# Globals for RMSE Plot on ML Page
rmse_vs_gen_log = []
rmse_flag = None
rmse_ax = None
rmse_canvas = None

# Placeholder for the Train-Surrogate button:
train_btn = None
# Decide initial enable/disable
_initial_train_state = 'normal' if true_runs_count > 0 else 'disabled'

surrogate_error_var = tk.StringVar(value="Surrogate RMSE: N/A")

def on_generate_lhc_clicked():
    """Run generate_lhc in a daemon thread, disabling the button until it’s done."""
    generate_lhc_btn.config(state="disabled")
    def task():
        try:
            generate_lhc()
        finally:
            generate_lhc_btn.config(state="normal")
    threading.Thread(target=task, daemon=True).start()

def generate_lhc():
    """
    Draw an LHC across [THETA_MIN, THETA_MAX], run true sims to seed X_train/Y_train,
    and persist each sample to training_data.json.
    """
    global X_train, Y_train, true_runs_count
    # Number of DOE samples
    n = initial_sample_var.get()
    dims = len(THETA_MIN)
    # Latin-Hypercube Sampling
    cut = np.linspace(0, 1, n + 1)
    u = np.random.rand(n, dims)
    lhs = np.zeros_like(u)
    for j in range(dims):
        idx = np.random.permutation(n)
        lhs[:, j] = cut[:-1] + u[idx, j] * (1.0 / n)
    # Scale to parameter bounds
    samples = THETA_MIN + lhs * (THETA_MAX - THETA_MIN)

    # Load or initialize persistent archive
    try:
        with open("training_data.json", "r") as f:
            archive = json.load(f)
    except FileNotFoundError:
        archive = []

    # Fetch comfort settings from UI
    T_min = float(comfort_Tmin_var.get())
    T_max = float(comfort_Tmax_var.get())
    T_ideal = float(comfort_Tideal_var.get())

    # Run true simulations for each sample
    for x in samples:
        y = co_simulate(
            x,
            comfort_bounds=(T_min, T_max),
            T_ideal=T_ideal
        )
        X_train.append(x)
        Y_train.append(y)
        true_runs_count += 1
        # Append to disk archive
        archive.append({
            "params": x.tolist(),
            "objectives": list(y)
        })

    # Write updated archive back to disk
    with open("training_data.json", "w") as f:
        json.dump(archive, f, indent=4)

    # Update live counter in UI
    true_runs_var.set(f"True Runs Used: {true_runs_count}")

    # Re-enable Train Surrogate button
    train_btn.config(state='normal')

# Update RMSE automatically on GUI (ML Page)
# def update_surrogate_rmse(X, Y, surrogate):
#     if X is not None and Y is not None and surrogate is not None:
#         pred = surrogate.predict(X)
#         rmse = np.sqrt(np.mean((pred - Y) ** 2))
#         def thread_safe_update():
#             surrogate_error_var.set(f"Surrogate RMSE: {rmse: .3f}")
#         root.after(0, thread_safe_update)
#     else:
#         root.after(0, lambda: surrogate_error_var.set("Surrogate RMSE: N/A"))


update_rmse_callback = None # Placeholder for the callback registration

def update_surrogate_rmse(X, Y, surrogate, rmse=None, gen=None):
    if X is not None and Y is not None and surrogate is not None:
        final_rmse = rmse
        if final_rmse is None:
            pred = surrogate.predict(X)
            rmse = np.sqrt(np.mean((pred - Y) ** 2))

        # Log RMSE to the list if generation is provided
        if gen is not None:
            rmse_vs_gen_log.append({"generation": gen, "rmse": rmse})

        def thread_safe_update():
            surrogate_error_var.set(f"Surrogate RMSE: {rmse: .3f}")
        root.after(0, thread_safe_update)
    else:
        root.after(0, lambda: surrogate_error_var.set("Surrogate RMSE: N/A"))

def reset_rmse_plot():
    rmse_values.clear()
    rmse_gens.clear()
    rmse_vs_gen_log.clear()

    # Clear the JSON file
    with open("rmse_log.json", "w") as f:
        json.dump([], f)

    # Clear the plot
    rmse_ax.clear()
    rmse_ax.set_xlabel("Generation")
    rmse_ax.set_ylabel("RMSE")
    rmse_ax.grid(True)
    rmse_canvas.draw()


def train_surrogate():
    """
    Fit the chosen ML surrogate on all data in training_data.json and report RMSE to UI.
    """
    global surrogate, X_train, Y_train
    # Load persistent archive
    try:
        with open("training_data.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        messagebox.showerror(
            "Training Error",
            "No training data found. Please generate DOE samples first."
        )
        return

    # Extract input/output arrays
    X = np.array([entry["params"] for entry in data])
    Y = np.array([entry["objectives"] for entry in data])

    # Clear RMSE vs. Generation plot
    reset_rmse_plot() ######################################## Check this!

    # Instantiate and fit surrogate
    kind = surrogate_type_var.get()
    surrogate = make_surrogate(kind)
    surrogate.fit(X, Y)

    # Sync in-memory lists for any downstream infill use
    X_train.clear()
    Y_train.clear()
    X_train.extend(X)
    Y_train.extend(Y)

    # Compute RMSE and update UI
    # pred = surrogate.predict(X)
    # rmse = np.sqrt(np.mean((pred - Y) ** 2))
    # surrogate_error_var.set(f"Surrogate RMSE: {rmse:.3f}")

    update_surrogate_rmse(X_train, Y_train, surrogate)

def start_surrogate_nsga():
    """Launch a surrogate-assisted NSGA-II run in a background thread."""
    global surrogate, X_train, Y_train

    def _worker():

        # Clear previous RMSE vs. GEN log
        with open("rmse_log.json", "w") as f:
            json.dump([], f)

        # Clear out any leftovers from previous runs
        for fname in ["pareto_update_trigger.json", "pareto_front.json", "knee_point.json"]:
            try:
                if fname == "pareto_update_trigger.json":
                    with open(fname, "w") as f:
                        f.truncate(0)
                else:
                    os.remove(fname)
            except FileNotFoundError:
                # if it didn’t exist, no problem
                pass

        # Clear the Matplotlib canvas immediately
        pareto_ax.clear()
        pareto_canvas.draw()

        # Safety checks before starting
        if not use_surrogate_var.get():
            messagebox.showwarning(
                "Surrogate Mode Not Enabled",
                "Please enable surrogate modeling by ticking the checkbox."
            )
            return

        if surrogate is None:
            messagebox.showerror(
                "Surrogate Not Trained",
                "No trained surrogate model found. Please train one first."
            )
            return

        if len(X_train) == 0 or len(Y_train) == 0:
            messagebox.showerror(
                "No Training Data",
                "Surrogate model has no training data. Please generate LHC samples and train the model first."
            )
            return

        # Pull settings
        gens = int(nsga_generations_var.get())
        pop_size = int(nsga_population_var.get())
        Tmin = float(comfort_Tmin_var.get())
        Tmax = float(comfort_Tmax_var.get())
        cbounds = (Tmin, Tmax)
        Tideal = float(comfort_Tideal_var.get())
        infill = float(infill_frac_var.get()) / 100.0
        retrain_it = int(retrain_var.get())

        print("[INFO] Starting surrogate-assisted NSGA-II optimization")
        print(f"[DEBUG] Generations: {gens}, Population: {pop_size}, Infill Fraction: {infill}, Retrain Interval: {retrain_it}")
        print(f"[DEBUG] Using surrogate: {type(surrogate)} with {len(X_train)} training samples")

        # # Call NSGA-II with surrogate enabled
        # pareto_thread = threading.Thread(target=live_pareto_updater, daemon=True)
        # pareto_thread.start()

        # Start simulation state tracking
        nsga_simulation_running.set(True)

        global update_rmse_callback
        update_rmse_callback = update_surrogate_rmse

        final_hof = nsga_ii(
            theta_min=THETA_MIN.tolist(),
            theta_max=THETA_MAX.tolist(),
            num_generations=gens,
            population_size=pop_size,
            comfort_bounds=cbounds,
            T_ideal=Tideal,
            use_surrogate=True,
            surrogate=surrogate,
            X_train=X_train,
            Y_train=Y_train,
            infill_frac=infill,
            retrain_interval=retrain_it,
            update_rmse_callback=update_surrogate_rmse,
            update_rmse_plot=update_rmse_plot
        )

        # Final update of Pareto plot
        final_pts = [ind.fitness.values for ind in final_hof]
        update_pareto_plot(final_pts)

        # Mark as done
        nsga_simulation_running.set(False)

        messagebox.showinfo("Success", "Surrogate-assisted NSGA-II completed.")

    # Set running flag before starting updater!
    nsga_simulation_running.set(True)

    # Start updater first
    threading.Thread(target=live_pareto_updater, daemon=True).start()

    # Then run the worker
    threading.Thread(target=_worker, daemon=True).start()

# Variables and Defaults
default_values = {
    "Thermal Absorptance": 0.85,
    "Solar Absorptance": 0.65,
    "Thickness": 0.2,
    "Conductivity": 1.5,
    "Initial Heating Setpoint (Guestroom)": 20,
    "Initial Cooling Setpoint (Guestroom)": 24,
    "Initial Heating Setpoint (Corridor)": 20,
    "Initial Cooling Setpoint (Corridor)": 24,
    "Lighting Power Density (Guestroom)": 10,
    "Lighting Power Density (Corridor)": 5,
    "Plug Load Power Density (Guestroom)": 5,
    "Plug Load Power Density (Corridor)": 2,
    "Infiltration Rate (Guestrooms)": 0.5,
    "Infiltration Rate (Corridors)": 0.3,
}

theta_min = np.array([0.75, 0.55, 0.1, 1.0, 15, 20.5, 15, 20.5, 0.1, 0.01, 0.01, 0.01, 0.01, 0.01])
theta_max = np.array([0.95, 0.75, 0.3, 2.5, 20, 25, 20, 25, 15, 7, 7, 5, 1, 1])

folder_path_var = tk.StringVar(value="C:\\Users\\wanimh\\PycharmProjects\\pythonProject\\.venv\\Scripts")
num_vectors_var = tk.StringVar(value="1")
baseline_simulation_var = tk.BooleanVar(value=False)
progress_bar_label = tk.StringVar(value="")
elapsed_time_display = tk.StringVar(value="Elapsed Time: 000h 00m 00s")
simulation_running = tk.BooleanVar(value=False)
use_nsga_ii_var = tk.BooleanVar(value=False)  # Checkbox for NSGA-II
use_surrogate_var = tk.BooleanVar(value=False) # For both NSGA-II and ML Page

# Run Counter
estimate_runs_var   = tk.IntVar(value=0)
run_counter_var     = tk.IntVar(value=0)

# NSGA‑II progress UI state
nsga_progress_label        = tk.StringVar(value="")
nsga_elapsed_time_display  = tk.StringVar(value="Elapsed Time: 000h 00m 00s")
nsga_simulation_running    = tk.BooleanVar(value=False)
nsga_start_time            = 0
# For the NSGA-II Parameters
comfort_Tmin_var = tk.StringVar(value="18.0")
comfort_Tmax_var = tk.StringVar(value="28.0")
comfort_Tideal_var = tk.StringVar(value="23.0")

nsga_generations_var = tk.StringVar(value="10")
nsga_population_var  = tk.StringVar(value="15")

# === PARETO PLOT AREA ===
pareto_fig = plt.figure(figsize=(5, 3))
pareto_ax = pareto_fig.add_subplot(111, projection='3d')

# To update this dynamically:
rmse_values = []
rmse_gens = []

# Live RMSE Plot on ML Page
def update_rmse_plot(gen, rmse):
    rmse_values.append(rmse)
    rmse_gens.append(gen)

    # Append to log
    rmse_vs_gen_log.append({"generation": gen, "rmse": rmse})

    # Save log to JSON file
    with open("rmse_log.json", "w") as f:
        json.dump(rmse_vs_gen_log, f, indent=2)

    # Plotting logic
    rmse_values.append(rmse)
    rmse_gens.append(gen)
    rmse_ax.clear()
    # rmse_ax.set_title("RMSE vs. Generation")
    rmse_ax.set_xlabel("Generation")
    rmse_ax.set_ylabel("RMSE")
    rmse_ax.grid(True)
    rmse_ax.plot(rmse_gens, rmse_values, marker='o', color='blue')
    rmse_canvas.draw()

# Function to update the plot
def update_pareto_plot(data_points):
    pareto_ax.clear()
    pareto_ax.set_title("Pareto Front (Live)")
    pareto_ax.set_xlabel("Energy Savings")
    pareto_ax.set_ylabel("GWP Reduction")
    pareto_ax.set_zlabel("Comfort Index")
    pareto_ax.set_xlim([-1, 1])
    pareto_ax.set_ylim([-1, 1])
    pareto_ax.set_zlim([-1, 1])
    if data_points:
        arr = np.array(data_points)
        pareto_ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2], c='r', marker='o',
                                       label = 'Non-dominated')

        # draw knee-point if available
        try:
            with open("knee_point.json", "r") as f:
                    kp = json.load(f)
                    x, y, z = kp["objectives"]
            pareto_ax.scatter(x, y, z,
                                c = 'b', marker = '^', s = 100,
                                label = 'Knee Point')
            pareto_ax.legend()
        except (FileNotFoundError, KeyError):
         pass

    pareto_canvas.draw()

# Timer variables
start_time = 0

def reset_run_counter():
    try:
        with open("run_counter.txt", "w") as f:
            f.write("0")
        messagebox.showinfo("Counter Reset", "EnergyPlus run counter reset to 0.")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to reset counter: {e}")

def activate_nsga_ii():
        theta_min_values = [float(entry.get()) for entry in min_entries]
        theta_max_values = [float(entry.get()) for entry in max_entries]
        print("NSGA-II selected for Optimization")

        # Temeprature Comfort Bounds
        T_min = float(comfort_Tmin_var.get())
        T_max = float(comfort_Tmax_var.get())
        T_ideal = float(comfort_Tideal_var.get())

        # start NSGA‑II progress UI
        nsga_progress_bar.start(50)
        nsga_progress_label.set("Running NSGA‑II…")
        global nsga_start_time
        nsga_start_time = time.time()
        nsga_simulation_running.set(True)
        threading.Thread(target=update_nsga_elapsed_time, daemon=True).start()

        threading.Thread(target=live_pareto_updater, daemon=True).start()

        num_generations = int(nsga_generations_var.get())
        population_size = int(nsga_population_var.get())

        # compute estimate
        estimate = int(nsga_population_var.get()) + (int(nsga_generations_var.get()) * int(nsga_population_var.get()))
        estimate_runs_var.set(estimate)

        # live read of run_counter.txt
        def _run_counter_updater():
            while nsga_simulation_running.get():
                try:
                    with open("run_counter.txt", "r") as f:
                        run_counter_var.set(int(f.read()))
                except:
                    run_counter_var.set(0)
                time.sleep(0.5)

        threading.Thread(target=_run_counter_updater, daemon=True).start()

        # Dispatch the long‐running work to a thread:
        def _worker():
            # Clear out any leftovers from previous runs
            for fname in ["pareto_update_trigger.json", "pareto_front.json", "knee_point.json"]:
                try:
                    os.remove(fname)
                except FileNotFoundError:
                    # if it didn’t exist, no problem
                    pass

            # Clear the Matplotlib canvas immediately
            pareto_ax.clear()
            pareto_canvas.draw()

            try:
                optimized_population = nsga_ii(
                    np.array(theta_min_values),
                    np.array(theta_max_values),
                    num_generations=num_generations,
                    population_size=population_size,
                    comfort_bounds=(T_min, T_max),
                    T_ideal=T_ideal
                )
                print(f"Optimized Population: {optimized_population}")

                # Load objectives from the saved Pareto front and update the live plot
                if os.path.exists("pareto_front.json"):
                    with open("pareto_front.json", "r") as f:
                        data = json.load(f)
                        update_pareto_plot([d["objectives"] for d in data])

                messagebox.showinfo("Success", "NSGA-II Optimization completed successfully!")
            except Exception as e:
                traceback.print_exc()
                messagebox.showerror("Error", f"An error occurred during NSGA-II execution:\n{e}")
            finally:
                nsga_progress_bar.stop()
                nsga_simulation_running.set(False)
                nsga_progress_label.set("NSGA‑II Complete!")

        threading.Thread(target=_worker, daemon=True).start()

def run_nsga_check():
    if not use_nsga_ii_var.get():
        messagebox.showwarning(
            "NSGA-II Required",
            "Please hit the 'Reset Run Counter' and tick\n"
            "the 'Use NSGA-II for Optimization' checkbox\n"
            "before running NSGA-II."
        )
        return
    activate_nsga_ii()

# Functions
def update_elapsed_time():
    while simulation_running.get():
        elapsed_seconds = int(time.time() - start_time)
        hours, remainder = divmod(elapsed_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        elapsed_time_display.set(f"Elapsed Time: {hours:03}h {minutes:02}m {seconds:02}s")
        time.sleep(1)

def update_nsga_elapsed_time():
     while nsga_simulation_running.get():
         elapsed = int(time.time() - nsga_start_time)
         h, rem = divmod(elapsed, 3600)
         m, s  = divmod(rem, 60)
         nsga_elapsed_time_display.set(f"Elapsed Time: {h:03}h {m:02}m {s:02}s")
         time.sleep(1)

def live_pareto_updater():
    while nsga_simulation_running.get():
        try:
            if os.path.exists("pareto_update_trigger.json"):
                # Wait until file is non-empty
                if os.path.getsize("pareto_update_trigger.json") > 10:
                    with open("pareto_update_trigger.json", "r") as f:
                        content = f.read().strip()
                        if content:
                            data_points = json.loads(content)
                            points = [d["objectives"] for d in data_points]
                            update_pareto_plot(points)
        except Exception as e:
            print(f"[LIVE PARETO ERROR] {e}")
        time.sleep(1)

# In activate_nsga_ii(), start thread:
threading.Thread(target=live_pareto_updater, daemon=True).start()

def update_progress_bar(total_time):
    while simulation_running.get():
        elapsed_seconds = int(time.time() - start_time)
        progress_bar["value"] = min(elapsed_seconds, total_time)
        root.update_idletasks()
        time.sleep(1)

# Navigation Functions
def show_page(page):
    for frame in pages.values():
        frame.grid_remove()
    pages[page].grid(row=0, column=0, sticky="nsew")

def start_simulation():
    # Prevent starting unless baseline box is checked
    if not baseline_simulation_var.get():
        messagebox.showwarning(
            "Baseline Required",
            "Please tick the 'Run Baseline Simulation? checkbox before starting."
        )
        return
    global start_time, process
    folder_path = folder_path_var.get()

    # # Validate num_vectors
    # try:
    #     num_vectors = int(num_vectors_var.get())
    #     if num_vectors <= 0:
    #         raise ValueError
    # except ValueError:
    #     messagebox.showerror("Invalid Input", "Please enter a positive integer for the number of parameter vectors.")
    #     return

    # Validate and save theta_baseline values
    try:
        theta_baseline = [float(entry.get()) for entry in baseline_entries]
        print(f"Retrieved theta_baseline values: {theta_baseline}")  # Debugging: Print the retrieved values

        # Save theta_baseline to a JSON file
        with open('theta_baseline.json', 'w') as f:
            json.dump(theta_baseline, f)
            print("theta_baseline saved to theta_baseline.json")
            print(f"Current working directory: {os.getcwd()}")  # Print the current working directory
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter valid numbers for the baseline parameters.")
        return  # Exit function if theta_baseline values are invalid

    ####### baseline_simulation = baseline_simulation_var.get()

    # Validate and save theta_min and theta_max values
    try:
        theta_min_values = [float(entry.get()) for entry in min_entries]
        theta_max_values = [float(entry.get()) for entry in max_entries]
        print(f"Retrieved theta_min: {theta_min_values}, theta_max: {theta_max_values}")  # Debugging

        # Save to JSON for debugging or verification
        with open('theta_min_max.json', 'w') as f:
            json.dump({"theta_min": theta_min_values, "theta_max": theta_max_values}, f)
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter valid numbers for Min and Max parameter values.")
        return

    # Pass these values in the subprocess call:

    theta_min_json = json.dumps(theta_min_values)
    theta_max_json = json.dumps(theta_max_values)

    if baseline_simulation_var.get():
        def _baseline_thread():
            Pilot_Interface.overwrite = "True"
            run_baseline_simulation(theta_baseline)
            progress_bar.stop()
            simulation_running.set(False)
            progress_bar_label.set("Baseline Simulation Complete!")
        progress_bar.configure(mode="indeterminate")
        progress_bar.start(50)
        progress_bar_label.set("Running Baseline...")
        simulation_running.set(True)
        threading.Thread(target=_baseline_thread, daemon=True).start()

    # Check if NSGA-II is selected
    if use_nsga_ii_var.get():
        activate_nsga_ii(theta_min_values,theta_max_values)  # Run the NSGA-II script
        # Wait for NSGA-II to finish (or ensure it writes outputs correctly)
        time.sleep(2)  # Adjust as per script execution time or mechanism


    start_time = time.time()
    threading.Thread(target=update_elapsed_time, daemon=True).start()

def run_simulation_process(folder_path, baseline_simulation, num_vectors, theta_baseline_json, theta_min_json, theta_max_json):
    global process
    venv_activate = os.path.join(folder_path, 'activate.bat')
    run_this = os.path.join(folder_path, 'Pilot_Interface.py')
    overwrite_flag = 'True' if baseline_simulation else 'False'

    # Properly enclose the JSON string in quotes
    theta_baseline_json = f'"{theta_baseline_json}"'
    theta_min_json = f'"{theta_min_json}"'
    theta_max_json = f'"{theta_max_json}"'

    try:
        process = subprocess.Popen(
            f'cmd /c "{venv_activate} && python {run_this} --Overwrite={overwrite_flag} --NumVectors={num_vectors} --ThetaBaseline={theta_baseline_json} --ThetaMin={theta_min_json} --ThetaMax={theta_max_json}"',
            shell=True
        )

        process.wait()

    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running Pilot_Interface.py: {e}")

    finally:
        simulation_running.set(False)
        progress_bar_label.set("Simulation Complete!")
        progress_bar["value"] = progress_bar["maximum"]
        progress_bar_label_widget.config(foreground="blue", font=("Helvetica", 10, "bold"))

def stop_simulation():
    simulation_running.set(False)
    if "process" in globals() and process.poll() is None:
        subprocess.Popen(f"taskkill /F /T /PID {process.pid}", shell=True)
        progress_bar_label.set("Simulation stopped.")
        print("Simulation stopped.")

# Reset all fields
def reset_fields():
    folder_path_var.set("")
    num_vectors_var.set("0")
    baseline_simulation_var.set(False)
    for entry in baseline_entries:
        entry.delete(0, tk.END)
        entry.insert(0, "0")
    for entry in min_entries + max_entries:
        entry.delete(0, tk.END)
        entry.insert(0, "0")
    progress_bar["value"] = 0
    progress_bar_label.set("")
    elapsed_time_display.set("Elapsed Time: 000h 00m 00s")

# Set default values for all fields
def set_default_values():
    for entry, value in zip(baseline_entries, default_values.values()):
        entry.delete(0, tk.END)
        entry.insert(0, str(value))
        num_vectors_var.set("1")
    for entry, value in zip(min_entries, theta_min):
        entry.delete(0, tk.END)
        entry.insert(0, str(value))
    for entry, value in zip(max_entries, theta_max):
        entry.delete(0, tk.END)
        entry.insert(0, str(value))
    folder_path_var.set("C:\\Users\\wanimh\\PycharmProjects\\pythonProject\\.venv\\Scripts")
    baseline_simulation_var.set(False)

# Browse folder dialog
def browse_folder():
    folder_selected = filedialog.askdirectory()
    folder_path_var.set(folder_selected)

# Pages Dictionary
pages = {}

# Helper Function to Add Scrollbars to Any Page
def create_scrollable_page():
    frame = tk.Frame(root)
    canvas = tk.Canvas(frame, highlightthickness=0)
    scrollable_frame = tk.Frame(canvas)
    scrollbar_vertical = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
    scrollbar_horizontal = ttk.Scrollbar(frame, orient="horizontal", command=canvas.xview)
    canvas.configure(yscrollcommand=scrollbar_vertical.set, xscrollcommand=scrollbar_horizontal.set)

    canvas.grid(row=0, column=0, sticky="nsew")
    scrollbar_vertical.grid(row=0, column=1, sticky="ns")
    scrollbar_horizontal.grid(row=1, column=0, sticky="ew")
    frame.rowconfigure(0, weight=1)
    frame.columnconfigure(0, weight=1)
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

    scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    return frame, scrollable_frame

# Main Page (Baseline Parameters Page)
main_page, scrollable_main = create_scrollable_page()
pages["Main"] = main_page

# Main Page Content
logo_image = Image.open("EnergyPlus_Py_Logo.png")
logo_resized = logo_image.resize((400, 450), Image.Resampling.LANCZOS)
logo_tk = ImageTk.PhotoImage(logo_resized)
logo_label = tk.Label(scrollable_main, image=logo_tk)
logo_label.grid(row=0, column=0, padx=10, pady=10, sticky="n")

# Control Buttons
button_frame = tk.Frame(scrollable_main)
button_frame.grid(row=1, column=0, padx=10, pady=10, sticky="n")
tk.Button(button_frame, text="Start", bg="green", fg="white", font=("Helvetica", 10), command=start_simulation).pack(fill="x", pady=5)
tk.Button(button_frame, text="Stop", bg="pink", fg="white", font=("Helvetica", 10), command=stop_simulation).pack(fill="x", pady=5)
tk.Button(button_frame, text="Reset Fields", bg="orange", fg="white", font=("Helvetica", 10), command=reset_fields).pack(fill="x", pady=5)
tk.Button(button_frame, text="Default Values", bg="blue", fg="white", font=("Helvetica", 10), command=set_default_values).pack(fill="x", pady=5)
tk.Button(button_frame, text="Next >>", bg="brown", fg="white", font=("Helvetica", 10), command=lambda: show_page("EA/AI")).pack(fill="x", pady=5)

# Parameter Section
parameter_frame = tk.LabelFrame(scrollable_main, text="Parameters", font=("Helvetica", 10, "bold"), padx=10, pady=10)
parameter_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

baseline_entries = []
min_entries = []
max_entries = []

# Header Row
tk.Label(parameter_frame, text="Baseline Parameters", font=("Helvetica", 10, "bold")).grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
tk.Label(parameter_frame, text="Optimization Bound Values", font=("Helvetica", 10, "bold"), fg="darkblue").grid(
    row=0, column=2, columnspan=2, pady=(0, 10), sticky="nsew"
)

# Headers
tk.Label(parameter_frame, text="Baseline Parameters", font=("Helvetica", 10, "bold")).grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
tk.Label(parameter_frame, text="Min Values", font=("Helvetica", 10, "bold")).grid(row=1, column=2, padx=5, pady=5, sticky="nsew")
tk.Label(parameter_frame, text="Max Values", font=("Helvetica", 10, "bold")).grid(row=1, column=3, padx=5, pady=5, sticky="nsew")

# Populate Parameters
for idx, (param, default) in enumerate(default_values.items()):
    tk.Label(parameter_frame, text=f"{param}:", font=("Helvetica", 9)).grid(row=idx + 2, column=0, padx=5, pady=2, sticky="w")
    baseline_entry = tk.Entry(parameter_frame, width=12, font=("Helvetica", 9))
    baseline_entry.insert(0, str(default))
    baseline_entry.grid(row=idx + 2, column=1, padx=5, pady=2, sticky="nsew")
    baseline_entries.append(baseline_entry)

    min_entry = tk.Entry(parameter_frame, width=12, font=("Helvetica", 9))
    min_entry.insert(0, str(theta_min[idx]))
    min_entry.grid(row=idx + 2, column=2, padx=5, pady=2, sticky="nsew")
    min_entries.append(min_entry)

    max_entry = tk.Entry(parameter_frame, width=12, font=("Helvetica", 9))
    max_entry.insert(0, str(theta_max[idx]))
    max_entry.grid(row=idx + 2, column=3, padx=5, pady=2, sticky="nsew")
    max_entries.append(max_entry)

# Create a combined frame that will contain both the Folder Path section and the description.
combined_frame = tk.Frame(scrollable_main)
combined_frame.grid(row=1, column=1, padx=10, pady=10, sticky="ew")

# Configure the combined_frame to have two columns that expand properly.
combined_frame.columnconfigure(0, weight=1)
combined_frame.columnconfigure(1, weight=1)

# --- Folder Path Part ---
folder_frame = tk.Frame(combined_frame)
folder_frame.grid(row=0, column=0, padx=(0,10), sticky="ew")
tk.Label(folder_frame, text="Folder Path:", font=("Helvetica", 9)).pack(side="left", padx=5)
tk.Entry(folder_frame, textvariable=folder_path_var, width=50).pack(side="left", fill="x", expand=True, padx=5)
tk.Button(folder_frame, text="Browse", command=browse_folder).pack(side="left", padx=5)

# --- Description/Instruction Part ---
# Instead of using a Label, you could use tk.Message for better wrapping.
description_text = (
    "Baseline Simulation: The simulation will run once using the baseline parameters "
    "provided in the 'Baseline Parameters' section. For optimization, candidate parameter "
    "vectors will be generated automatically by the Evolutionary Algorithm (EA) you choose " 
    "based on the 'Min Values' and 'Max Values' specified."
)
# --- Instruction Box ---
instr_frame = tk.LabelFrame(
    combined_frame,
    text="Info",
    font=("Helvetica", 9, "bold"),
    bd=2,
    relief="groove",
    padx=10,
    pady=10
)
instr_frame.grid(row=1, column=0, padx=(10,0), pady=(15,0), sticky="ew")

tk.Message(
    instr_frame,
    text=description_text,
    font=("Helvetica", 9),
    width=400,
    justify="center"
).pack(fill="both", expand=True)

# Baseline Simulation Checkbox
tk.Checkbutton(scrollable_main, text="Run Baseline Simulation?", variable=baseline_simulation_var).grid(
    row=3, column=1, padx=10, pady=5, sticky="w"
)

# Progress Bar
progress_bar_frame = tk.Frame(scrollable_main)
progress_bar_frame.grid(row=4, column=1, padx=10, pady=10, sticky="ew")
progress_bar = ttk.Progressbar(progress_bar_frame, orient="horizontal", length=400, mode="determinate")
progress_bar.pack(fill="x", expand=True)
progress_bar_label_widget = tk.Label(scrollable_main, textvariable=progress_bar_label, font=("Helvetica", 9))
progress_bar_label_widget.grid(row=5, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
elapsed_time_label = tk.Label(scrollable_main, textvariable=elapsed_time_display, font=("Helvetica", 9))
elapsed_time_label.grid(row=6, column=0, columnspan=2, padx=10, pady=5, sticky="ew")

# Footer
footer_frame = tk.Frame(scrollable_main)
footer_frame.grid(row=7, column=0, columnspan=2, pady=10, sticky="ew")
footer_label = tk.Label(footer_frame, text="© 2024-2025: Dr. Mubashir Wani. All Rights Reserved.", font=("Helvetica", 9))
footer_label.pack(fill="x")

# EA/AI Page
ea_ai_page, scrollable_ea_ai = create_scrollable_page()
pages["EA/AI"] = ea_ai_page

tk.Label(scrollable_ea_ai, text="Evolutionary Algorithm (EA) and Artificial Intelligence (AI)", font=("Helvetica", 14, "bold"), fg="darkblue").pack(pady=20)
tk.Button(scrollable_ea_ai, text="PSO", font=("Helvetica", 12), bg="lightblue", width=20, height=2, command=lambda: show_page("PSO")).pack(pady=10)
tk.Button(scrollable_ea_ai, text="NSGA-II", font=("Helvetica", 12), bg="lightgreen", width=20, height=2, command=lambda: show_page("NSGA-II")).pack(pady=10)
tk.Button(scrollable_ea_ai, text="CLPSO", font=("Helvetica", 12), bg="lightcoral", width=20, height=2, command=lambda: show_page("CLPSO")).pack(pady=10)
tk.Button(scrollable_ea_ai, text="Machine Learning", font=("Helvetica", 12), bg="lightskyblue", width=20, height=2, command=lambda: show_page("Machine Learning")).pack(pady=10)
tk.Button(scrollable_ea_ai, text="Home", font=("Helvetica", 10), bg="brown", fg="white", command=lambda: show_page("Main")).pack(pady=20)

# Algorithm Pages
for algo in ["PSO", "NSGA-II", "CLPSO", "Machine Learning"]:
    page, scrollable_page = create_scrollable_page()
    pages[algo] = page

    # Machine Learning / Surrogate Frame
    if algo == "Machine Learning":
        ml_frame = tk.LabelFrame(scrollable_page,
                                 text="Surrogate-Assisted Infill Settings",
                                 font=("Helvetica", 10, "bold"),
                                 padx=10, pady=10)
        ml_frame.pack(fill="x", padx=20, pady=10)

        # 1) Toggle surrogate on/off
        #use_surrogate_var = tk.BooleanVar(value=False)

        tk.Checkbutton(ml_frame,
                       text="Enable Surrogate Modeling",
                       variable=use_surrogate_var,
                       font=("Helvetica", 9)).grid(row=0, column=0, columnspan=2, sticky="w")

        # 2) Initial sample size
        tk.Label(ml_frame, text="Initial DOE Sample:", font=("Helvetica", 9)) \
            .grid(row=1, column=0, sticky="e", pady=2)
        initial_sample_var = tk.IntVar(value=100)
        tk.Entry(ml_frame, textvariable=initial_sample_var, width=6) \
            .grid(row=1, column=1, sticky="w", pady=2)

        # 3) Infill fraction
        tk.Label(ml_frame, text="Infill Fraction (%):", font=("Helvetica", 9)) \
            .grid(row=2, column=0, sticky="e", pady=2)
        infill_frac_var = tk.DoubleVar(value=10.0)
        tk.Entry(ml_frame, textvariable=infill_frac_var, width=6) \
            .grid(row=2, column=1, sticky="w", pady=2)

        # 4) Retrain interval
        tk.Label(ml_frame, text="Retrain Every (gens):", font=("Helvetica", 9)) \
            .grid(row=3, column=0, sticky="e", pady=2)
        retrain_var = tk.IntVar(value=5)
        tk.Entry(ml_frame, textvariable=retrain_var, width=6) \
            .grid(row=3, column=1, sticky="w", pady=2)

        # 5) Surrogate type selector
        tk.Label(ml_frame, text="Surrogate Model:", font=("Helvetica", 9)) \
            .grid(row=4, column=0, sticky="e", pady=2)
        surrogate_type_var = tk.StringVar(value="RandomForest")
        tk.OptionMenu(ml_frame, surrogate_type_var,
                      "RandomForest", "GaussianProcess", "MLPRegressor") \
            .grid(row=4, column=1, sticky="w", pady=2)

        # 6) Action buttons
        btn_frame = tk.Frame(ml_frame)
        btn_frame.grid(row=5, column=0, columnspan=2, pady=(10, 0))
        # tk.Button(btn_frame, text="Generate LHC Sample",
        #           command=generate_lhc).pack(side="left", padx=5)
        generate_lhc_btn = tk.Button(btn_frame, text="Generate LHC Sample",
                                     command=on_generate_lhc_clicked, bg="#ffcccc")
        generate_lhc_btn.pack(side="left", padx=5)
        # tk.Button(btn_frame, text="Train Surrogate",
        #           command=train_surrogate).pack(side="left", padx=5)

        # global train_btn
        train_btn = tk.Button(
            btn_frame,
            text="Train Surrogate",
            command=train_surrogate,
            bg="white",
            state=_initial_train_state
        )
        train_btn.pack(side="left", padx=5)

        # tk.Button(btn_frame, text="Start SA-NSGA-II",
        #           command=start_surrogate_nsga).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Start SA-NSGA-II", bg="#cceeff",
                  command=lambda: threading.Thread(target=start_surrogate_nsga, daemon=True).start()).pack(side="left", padx=5)

        # 7) Live status labels
        status_frame = tk.Frame(ml_frame)
        status_frame.grid(row=6, column=0, columnspan=2, pady=(10, 0))
        # surrogate_error_var = tk.StringVar(value="Initial Surrogate RMSE: N/A")
        surrogate_error_var.set("Initial Surrogate RMSE: N/A")
        true_runs_var = tk.StringVar(value="True LHC Runs Used: 0")
        tk.Label(status_frame, textvariable=surrogate_error_var,
                 font=("Helvetica", 9)).pack(anchor="w")
        tk.Label(status_frame, textvariable=true_runs_var,
                 font=("Helvetica", 9)).pack(anchor="w")

        # ── On startup, load any existing DOE archive and populate the counter ──
        ARCHIVE_PATH = "training_data.json"  # or wherever you write your LHC samples

        if os.path.exists(ARCHIVE_PATH):
            with open(ARCHIVE_PATH, "r") as f:
                archive = json.load(f)
            # rebuild X_train/Y_train from the file
            for entry in archive:
                X_train.append(np.array(entry["params"], dtype=float))
                Y_train.append(tuple(entry["objectives"]))
            # set the counter label
            true_runs_count = len(X_train)
            true_runs_var.set(f"True LHC Runs Used: {true_runs_count}")

            # If we have some true-runs, enable the "Train Surrogate" button
            if true_runs_count > 0:
                train_btn.config(state="normal")

        # ── RMSE vs. Generations Frame ──
        rmse_plot_labelframe = tk.LabelFrame(scrollable_page,
                                             text="RMSE vs. Generations",
                                             font=("Helvetica", 10, "bold"),
                                             padx=10, pady=5)
        rmse_plot_labelframe.pack(fill="x", padx=10, pady=5)

        # Matplotlib figure and canvas
        rmse_fig, rmse_ax = plt.subplots(figsize=(5, 2.5))
        # rmse_ax.set_title("RMSE vs. Generation")
        rmse_ax.set_xlabel("Generation")
        rmse_ax.set_ylabel("RMSE")
        rmse_ax.grid(True)

        rmse_canvas = FigureCanvasTkAgg(rmse_fig, master=rmse_plot_labelframe)
        rmse_canvas.draw()
        rmse_canvas.get_tk_widget().pack(fill="both", expand=True)

        rmse_canvas.get_tk_widget().config(height=385)

        # Now callbacks (generate_lhc, train_surrogate, start_surrogate_nsga)
        # can read/write these vars to drive the ML workflow.

    # Add a specific content for NSGA-II
    if algo == "NSGA-II":

        # --- Layout frame to align controls nicely ---
        nsga_layout_frame = tk.Frame(scrollable_page)
        nsga_layout_frame.pack(fill="both", expand=True, padx=20, pady=10)
        nsga_layout_frame.columnconfigure(0, weight=1)
        nsga_layout_frame.columnconfigure(1, weight=1)

        # --- Left Column: Buttons & Progress ---
        left_nsga_frame = tk.Frame(nsga_layout_frame)
        left_nsga_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        tk.Button(left_nsga_frame, text="Run NSGA-II", font=("Helvetica", 12), bg="lightgreen",
                  command=run_nsga_check).pack(pady=10, fill="x")
        tk.Checkbutton(left_nsga_frame, text="Use NSGA-II for Optimization", variable=use_nsga_ii_var,
                       font=("Helvetica", 10)).pack(pady=5, fill="x")
        tk.Button(left_nsga_frame, text="Reset Run Counter", font=("Helvetica", 10), bg="tomato",
                  command=reset_run_counter).pack(pady=10, fill="x")

        # Progress Bar
        nsga_progress_bar = ttk.Progressbar(left_nsga_frame, orient="horizontal", length=300, mode="indeterminate")
        nsga_progress_bar.pack(pady=5, fill="x")
        tk.Label(left_nsga_frame, textvariable=nsga_progress_label, font=("Helvetica", 9)).pack()
        tk.Label(left_nsga_frame, textvariable=nsga_elapsed_time_display, font=("Helvetica", 9)).pack()

        # --- Right Column: Optimization Settings ---
        right_nsga_frame = tk.LabelFrame(nsga_layout_frame, text="Optimization Settings", padx=10, pady=10,
                                         font=("Helvetica", 10, "bold"))
        right_nsga_frame.grid(row=0, column=1, sticky="nsew")

        # Occupant Comfort Settings Frame

        comfort_bounds_min = tk.DoubleVar(value=18.0)
        comfort_bounds_max = tk.DoubleVar(value=28.0)
        T_ideal_value = tk.DoubleVar(value=23.0)

        comfort_box = tk.LabelFrame(
            right_nsga_frame,
            text="Occupant Comfort Settings",
            font=("Helvetica", 10, "bold"),
            padx=8, pady=4
        )
        comfort_box.grid(row=0, column=0, sticky="ew", padx=2, pady=(2, 8), columnspan=2)

        tk.Label(comfort_box, text="Min Comfort Temp (°C):").grid(row=0, column=0, sticky="w")
        tk.Entry(comfort_box, textvariable=comfort_bounds_min, width=10).grid(row=0, column=1, sticky="w")

        tk.Label(comfort_box, text="Max Comfort Temp (°C):").grid(row=1, column=0, sticky="w")
        tk.Entry(comfort_box, textvariable=comfort_bounds_max, width=10).grid(row=1, column=1, sticky="w")

        tk.Label(comfort_box, text="Ideal Temperature (°C):").grid(row=2, column=0, sticky="w")
        tk.Entry(comfort_box, textvariable=T_ideal_value, width=10).grid(row=2, column=1, sticky="w")

        # SA-NSGA-II Frame

        sa_nsga_frame = tk.LabelFrame(
            right_nsga_frame,
            text="SA-NSGA-II",
            font=("Helvetica", 9, "bold"),
            padx=5, pady=3
        )
        sa_nsga_frame.grid(row=1, column=0, sticky="ew", padx=2, pady=(2, 2), columnspan=2)

        tk.Button(
            sa_nsga_frame,
            text="Expedite Computation (Machine Learning)",
            font=("Helvetica", 9),
            bg="pink",
            command=lambda: show_page("Machine Learning")
        ).grid(row=0, column=0, sticky="w", padx=2, pady=(0, 2))

        tk.Label(sa_nsga_frame, text="Surrogate Modeling: ").grid(row=1, column=0, sticky="w", padx=(0, 2))

        surrogate_status_label = tk.Label(sa_nsga_frame, text='Inactive', font=("Helvetica", 9, "bold"), fg="red")
        surrogate_status_label.grid(row=1, column=2, sticky="w", padx=(6,0))

        surrogate_led = tk.Canvas(sa_nsga_frame, width=20, height=20, highlightthickness=0)
        surrogate_led.grid(row=1, column=1, sticky="w")
        surrogate_led.create_oval(2, 2, 18, 18, fill="red", tags="led")

        # # --- Place Surrogate Mode Checkbox (for user control) ---
        # surrogate_checkbox = tk.Checkbutton(
        #     right_nsga_frame, text="Use Surrogate Model (SA-NSGA-II)",
        #     variable=use_surrogate_var
        # )
        # surrogate_checkbox.grid(row=3, column=0, columnspan=2, sticky="w")

        # tk.Button(
        #     right_nsga_frame,
        #     text="Go to Surrogate Modeling (Machine Learning)",
        #     font=("Helvetica", 9),
        #     command=lambda: show_page("Machine Learning")
        # ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(0, 2))

        # LED Set-up
        def update_surrogate_led(*args):
            if use_surrogate_var.get():
                surrogate_led.itemconfig("led", fill="green")
                surrogate_status_label.config(text="Active", fg="green")
            else:
                surrogate_led.itemconfig("led", fill="red")
                surrogate_status_label.config(text="Inactive", fg="red")

        use_surrogate_var.trace_add("write", update_surrogate_led)
        update_surrogate_led()


        # In your NSGA-II page layout (after declaring use_surrogate_var):
        # led_frame = tk.Frame(right_nsga_frame)
        # led_frame.grid(row=4, column=0, columnspan=2, pady=(10, 0))
        # tk.Label(led_frame, text="Surrogate Modeling: ").pack(side="left")
        # surrogate_led = tk.Canvas(led_frame, width=20, height=20, highlightthickness=0)
        # surrogate_led.pack(side="left")
        # surrogate_led.create_oval(2, 2, 18, 18, fill="red", tags="led")  # starts as red

        # Bind variable (this triggers when box is toggled)
        use_surrogate_var.trace_add("write", lambda *args: update_surrogate_led())

        # comfort_bounds_min = tk.DoubleVar(value=18.0)
        # comfort_bounds_max = tk.DoubleVar(value=28.0)
        # T_ideal_value = tk.DoubleVar(value=23.0)

        # tk.Label(right_nsga_frame, text="Min Comfort Temp (°C):").grid(row=0, column=0, sticky="w")
        # tk.Entry(right_nsga_frame, textvariable=comfort_bounds_min, width=10).grid(row=0, column=1, sticky="w")
        #
        # tk.Label(right_nsga_frame, text="Max Comfort Temp (°C):").grid(row=1, column=0, sticky="w")
        # tk.Entry(right_nsga_frame, textvariable=comfort_bounds_max, width=10).grid(row=1, column=1, sticky="w")
        #
        # tk.Label(right_nsga_frame, text="Ideal Temperature (°C):").grid(row=2, column=0, sticky="w")
        # tk.Entry(right_nsga_frame, textvariable=T_ideal_value, width=10).grid(row=2, column=1, sticky="w")

        # --- NEW COLUMN: NSGA-II Parameters ---
        params_nsga_frame = tk.LabelFrame(nsga_layout_frame, text="NSGA-II Parameters", padx=10, pady=10,
                                          font=("Helvetica", 10, "bold"))
        params_nsga_frame.grid(row=0, column=2, sticky="nsew")

        # NSGA-II Input Fields
        tk.Label(params_nsga_frame, text="Number of Generations:").grid(row=0, column=0, sticky="w")
        tk.Entry(params_nsga_frame, textvariable=nsga_generations_var, width=10).grid(row=0, column=1, sticky="w")

        tk.Label(params_nsga_frame, text="Population Size:").grid(row=1, column=0, sticky="w")
        tk.Entry(params_nsga_frame, textvariable=nsga_population_var, width=10).grid(row=1, column=1, sticky="w")

        # ── ESTIMATE & LIVE COUNTER ──
        tk.Label(params_nsga_frame, text="Max. Estimated Runs:").grid(row=2, column=0, sticky="w")
        tk.Label(params_nsga_frame, textvariable=estimate_runs_var).grid(row=2, column=1, sticky="w")

        tk.Label(params_nsga_frame, text="Current Run:").grid(row=3, column=0, sticky="w")
        tk.Label(params_nsga_frame, textvariable=run_counter_var).grid(row=3, column=1, sticky="w")

        # NSGA-II Page Pareto Front Frame
        pareto_frame = tk.LabelFrame(scrollable_page, text="Live Pareto Front", padx=10, pady=10, font=("Helvetica", 10, "bold"))
        pareto_frame.pack(padx=10, pady=10, fill="both", expand=True)
        pareto_canvas = FigureCanvasTkAgg(pareto_fig, master=pareto_frame)
        pareto_canvas.get_tk_widget().pack(fill='both', expand=True)

        # Instruction Box below the layout
        nsga_info_frame = tk.LabelFrame(scrollable_page, text="Info", font=("Helvetica", 9, "bold"), bd=2,
                                        relief="groove", padx=10, pady=10)
        nsga_info_frame.pack(padx=10, pady=10, fill="x")

        tk.Message(
            nsga_info_frame,
            text="NSGA-II Optimization: This multi-objective evolutionary algorithm "
                 "searches for optimal trade-offs between energy savings, GWP reduction, "
                 "and occupant comfort. You can configure comfort bounds on the top-middle, "
                 "and run the optimization using the buttons on the top-left. \n\n"
                 "Max. Estimated Runs = Population Size + (Population Size x Generations). Watch the live Pareto Front " 
                 "and knee-point markers update in real time as solutions emerge",

            font=("Helvetica", 7),
            width=600,
            justify="center"
        ).pack(fill="both", expand=True)

        # # — NSGA‑II Progress Bar & Elapsed Time —
        # nsga_progress_frame = tk.Frame(scrollable_page)
        # nsga_progress_frame.pack(pady=10, fill="x")
        # nsga_progress_bar = ttk.Progressbar(
        #     nsga_progress_frame,
        #     orient="horizontal", length=300, mode="indeterminate"
        # )
        # nsga_progress_bar.pack(fill="x", expand=True)
        # tk.Label(
        #     nsga_progress_frame,
        #     textvariable=nsga_progress_label,
        #     font=("Helvetica", 9)
        # ).pack(padx=5)
        # tk.Label(
        #     scrollable_page,
        #     textvariable=nsga_elapsed_time_display,
        #     font=("Helvetica", 9)
        # ).pack(pady=(0, 10))

    tk.Label(scrollable_page, text=f"{algo} Page", font=("Helvetica", 14, "bold"), fg="darkblue").pack(pady=20)
    tk.Button(scrollable_page, text="Back", font=("Helvetica", 10), bg="blue", fg="white", command=lambda: show_page("EA/AI")).pack(pady=10)
    tk.Button(scrollable_page, text="Home", font=("Helvetica", 10), bg="brown", fg="white", command=lambda: show_page("Main")).pack(pady=20)

# Show Main Page initially
show_page("Main")

root.mainloop()


