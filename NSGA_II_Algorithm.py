# NSGA_II_Algorithm.py

# PyE+ - A Python-EnergyPlus Optimization Framework
# Copyright (c) 2025 Dr. Mubashir Hussain Wani
# Licensed under the MIT License. See LICENSE file in the project root for full license text.

import sys
import numpy as np
from deap import base, creator, tools, algorithms
from Pilot_Interface import co_simulate
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import subprocess
import json
import os
import matplotlib.pyplot as plt

# ==== Surrogate Training and Validation
def compute_validation_rmse(model, X, Y):
    # Split into 80% train, 20% validation
    X_train_sub, X_val, Y_train_sub, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Train surrogate on 80%
    model.fit(X_train_sub, Y_train_sub)

    # Predict on 20% validation
    Y_pred = model.predict(X_val)

    # Compute RMSE per-objective (vector RMSE)
    errors = np.array(Y_val) - np.array(Y_pred)
    rmse = np.sqrt(np.mean(errors**2))

    return rmse

# ==== SAFE EVALUATE ====
def safe_evaluate(theta_vect, comfort_bounds, T_ideal):
    try:
        return co_simulate(theta_vect, comfort_bounds=comfort_bounds, T_ideal=T_ideal)
    except Exception as e:
        print(f"[ERROR] Simulation failed for parameters {theta_vect}: {e}")
        # Return the lowest-possible fitness (penalize the individual)
        # return (0.0, 0.0, 0.0)
        objectives = [0.0, 0.0, 0.0]
        print("Objectives (penalized for invalid output):", json.dumps(objectives))
        return objectives

# ==== CONFIGURATION ====
FOLDER_PATH = "C:/Users/wanimh/PycharmProjects/pythonProject/.venv/Scripts"
PILOT_SCRIPT = os.path.join(FOLDER_PATH, "Pilot_Interface.py")
THETA_MIN = np.array([0.85, 0.65, 0.2, 1.5, 15, 20.5, 15, 20.5, 0.1, 0.01, 0.01, 0.01, 0.01, 0.01])
THETA_MAX = np.array([0.95, 0.75, 0.3, 2.5, 20, 23, 20, 23, 2.0, 0.2, 0.2, 0.2, 0.2, 0.2])

# ==== DEAP SETUP ====
creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, 1.0))  # Maximize, Maximize, Maximize
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
#toolbox.register("attr_float", lambda i: np.random.uniform(THETA_MIN[i], THETA_MAX[i]))

def random_theta(i):
    return np.random.uniform(THETA_MIN[i], THETA_MAX[i])
toolbox.register("attr_float", random_theta)

# toolbox.register("individual", tools.initCycle, creator.Individual,
#                  (lambda: [toolbox.attr_float(i) for i in range(len(THETA_MIN))],), n=1)

toolbox.register("individual", tools.initIterate, creator.Individual, lambda:[toolbox.attr_float(i) for i in range(len(THETA_MIN))])

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Evaluation function that wraps the co_simulation from Pilot_Interface.py
# def evaluate(individual):
#     """
#     Directly call the co_simulate() function for this candidate vector.
#     Baseline will never be re‑run here.
#     """
#     # individual is a DEAP Individual (list‑like), so we can pass it straight through.
#     try:
#         values = co_simulate(individual)
#         # co_simulate prints detailed debug out, so you still get visibility.
#         return values
#     except Exception as e:
#         print(f"[NSGA-II ERROR] co_simulate failed: {e}")
#         # worst‑case fallback so GA keeps running
#         return 1e6, 1e6, 1e6

# toolbox.register("evaluate", evaluate)
# toolbox.unregister("evaluate")
# toolbox.register("evaluate", lambda ind: safe_evaluate(ind, comfort_bounds, T_ideal))
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selNSGA2)


# === For clipping the individuals to their bounds after crossover and mutation ===
def clip_individual(individual, bounds):
    for i, (low, high) in enumerate(bounds):
        individual[i] = max(min(individual[i], high), low)
    return individual
# =================================================================================

global ParetoFront_Path
ParetoFront_Path = "C:/Users/wanimh/PycharmProjects/pythonProject/.venv/Scripts/pareto_front.json"

# ==== KNEE POINT IDENTIFICATION ====
def identify_knee_point_mmd(data):
    ideal_point = np.array([1.0, 1.0, 1.0])
    # Filter out penalized individuals
    filtered = [d for d in data if any(v > 1e-5 for v in d["objectives"])]
    if not filtered:
        raise ValueError("No valid individuals in the Pareto Front for knee-point identification")
    objectives = np.array([d["objectives"] for d in filtered])
    distances = np.sum(np.abs(objectives - ideal_point), axis=1)
    knee_index = np.argmin(distances)
    return knee_index, objectives[knee_index], filtered[knee_index]["params"]

def plot_pareto_front(ParetoFront_Path):
    # load the Pareto JSON
    try:
        with open(ParetoFront_Path, "r") as f:
            content = f.read().strip()
            if not content:
                print("[PLOT] pareto_front.json is empty.")
                return
            data = json.loads(content)
    except Exception as e:
        print(f"[PLOT ERROR] Failed to load Pareto front: {e}")
        return

    # if there were no non‐dominated solutions, just skip
    if not data:
        print("[PLOT] No Pareto points to plot.")
        return

    # Build objectives array
    objs = np.array([d["objectives"] for d in data])

    # If only one solution, force 2D shape
    if objs.ndim == 1:
        objs = objs[np.newaxis, :]

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(objs[:, 0], objs[:, 1], objs[:, 2], c='r', label='Pareto points')

    # try annotating knee point, but don’t crash if none
    try:
        knee_index, knee_point, knee_params = identify_knee_point_mmd(data)
        ax.scatter(
            knee_point[0], knee_point[1], knee_point[2],
            c='b', marker='^', s=100, label='Knee Point'
        )
        print(f"Knee Point (MMD): Index {knee_index}, Objectives {knee_point}")
        print("Corresponding Parameters:", knee_params)
        with open("knee_point.json", "w") as f:
            json.dump({
                "index": int(knee_index),
                "objectives": knee_point.tolist(),
                "params": knee_params
            }, f, indent=4)
    except ValueError:
        print("[plot_pareto_front] no valid knee point — skipping annotation")

    ax.set_xlabel('Energy Savings (kWh)')
    ax.set_ylabel('GWP Reduction (kg CO2e)')
    ax.set_zlabel('Occupant Comfort Index')
    ax.set_title('Pareto Front')
    ax.legend()
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    plt.legend()
    plt.tight_layout()
    plt.savefig("pareto_front.png")
    # plt.show()

    # only print/write knee JSON if we found one
    if 'knee_index' in locals():
        print(f"Knee Point (MMD): Index {knee_index}, Objectives {knee_point}")
        print("Corresponding Parameters:", knee_params)
        with open("knee_point.json", "w") as f:
            json.dump({
                "index": int(knee_index),
                "objectives": knee_point.tolist(),
                "params": knee_params
            }, f, indent=4)


def nsga_ii(theta_min, theta_max, num_generations=3, population_size=5, comfort_bounds=(18.0, 28.0), T_ideal=23.0,
            use_surrogate=False, surrogate=None, X_train=None, Y_train=None, infill_frac=0.1, retrain_interval=5, update_rmse_callback=None, update_rmse_plot=None):
    global THETA_MIN, THETA_MAX
    THETA_MIN = np.array(theta_min)
    THETA_MAX = np.array(theta_max)

    def attr_float(i): return np.random.uniform(THETA_MIN[i], THETA_MAX[i])

    toolbox.unregister("attr_float")
    toolbox.register("attr_float", lambda i: attr_float(i))
    toolbox.unregister("individual")
    toolbox.register("individual", tools.initIterate, creator.Individual,
                     lambda: [toolbox.attr_float(i) for i in range(len(THETA_MIN))])
    toolbox.unregister("population")
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # toolbox.register("evaluate", lambda ind: safe_evaluate(ind, comfort_bounds, T_ideal))

    if use_surrogate and surrogate is not None:
        print("[INFO] Registering surrogate evaluation in toolbox.")

        def surrogate_eval(ind):
            vect = np.array(ind).reshape(1, -1)
            try:
                y_pred = surrogate.predict(vect)[0]
                return tuple(y_pred)
            except Exception as e:
                print(f"[SURROGATE ERROR] {e} — falling back to true simulation")
                return safe_evaluate(ind, comfort_bounds, T_ideal)

        toolbox.register("evaluate", surrogate_eval)
    else:
        print("[INFO] Registering true simulation in toolbox.")
        toolbox.register("evaluate", lambda ind: safe_evaluate(ind, comfort_bounds, T_ideal))


    # pop = toolbox.population(n=population_size)

    pop = toolbox.population(n=population_size)

    # --- NEW: evaluate initial pop so HOF isn't empty ---
    hof = tools.ParetoFront()
    fits = list(map(toolbox.evaluate, pop))

    for ind, fit in zip(pop, fits):
        ind.fitness.values = fit
    hof.update(pop)


    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)


    # algorithms.eaMuPlusLambda(pop, toolbox, mu=population_size, lambda_=population_size * 2,
    #                           cxpb=0.7, mutpb=0.2, ngen=num_generations,
    #                           stats=stats, halloffame=hof, verbose=True)
    #
    # with open("pareto_front.json", "w") as f:
    #     json.dump([{"params": ind, "objectives": ind.fitness.values} for ind in hof], f, indent=4)
    #
    # # Call the function to plot the Pareto front
    # print("\nPareto Front saved to pareto_front.json")
    # plot_pareto_front(ParetoFront_Path)
    # # return hof

    for gen in range(num_generations):
        # 1) variation
        offspring = algorithms.varAnd(pop, toolbox, cxpb=0.7, mutpb=0.2)

        # === Force the bounds within specified limits ===
        bounds = list(zip(THETA_MIN, THETA_MAX))
        for ind in offspring:
            clip_individual(ind, bounds)
        # ================================================

        # 2) surrogate prediction for all
        if use_surrogate and surrogate and len(X_train) > 0:
            # build input array
            X_off = np.array([ind for ind in offspring])
            preds = surrogate.predict(X_off)

            # 3) select k infill individuals
            k = max(1, int(len(offspring) * infill_frac))
            # simple acquisition: pick highest sum-of-objectives
            scores = [sum(p) if p is not None else -np.inf for p in preds]
            top_idxs = sorted(range(len(offspring)), key=lambda i: scores[i])[-k:]

            print(f"[INFO] Using true simulation for individuals: {top_idxs}")

        else:
            preds = [None] * len(offspring)

        # 4) assign fitness: true sims for top k, predictions for the rest
        for i, ind in enumerate(offspring):
            if use_surrogate and surrogate:
                if i in top_idxs:
                    # real evaluation + archive
                    y_true = safe_evaluate(ind, comfort_bounds, T_ideal)
                    ind.fitness.values = y_true
                    X_train.append(np.array(ind))
                    Y_train.append(y_true)
                else:
                    ind.fitness.values = tuple(preds[i])
            else:
                # pure NSGA-II path
                ind.fitness.values = toolbox.evaluate(ind)

        # 5) next pop & update HOF
        pop = toolbox.select(pop + offspring, k=population_size)
        hof.update(pop)

        # 6) retrain surrogate if needed
        if use_surrogate and surrogate and ((gen + 1) % retrain_interval == 0):
            # X_arr = np.vstack(X_train)
            # Y_arr = np.vstack(Y_train)
            # surrogate.fit(X_arr, Y_arr)
            # Y_pred = surrogate.predict(X_arr)
            # rmse = np.sqrt(np.mean((Y_arr - Y_pred) ** 2))

            # For RMSE vs. GEN plot
            rmse = compute_validation_rmse(surrogate, X_train, Y_train)
            update_rmse_plot((gen + 1), rmse)

            print(f"[Surrogate] retrained at gen {gen + 1}, RMSE={rmse:.3f}")

            # To update RMSE dynamically on GUI (ML Page)
            if update_rmse_callback:
                update_rmse_callback(X_train, Y_train, surrogate, rmse)

        # # 7) update RMSE on GUI (ML Page)
        # if use_surrogate and (gen+1) % retrain_interval == 0:
        #     surrogate.fit(X_train, Y_train)
        #     if update_rmse_callback is not None:
        #         update_rmse_callback(X_train, Y_train, surrogate)

    # for gen in range(num_generations):
    #     offspring = algorithms.varAnd(pop, toolbox, cxpb=0.7, mutpb=0.2)
    #     fits = toolbox.map(toolbox.evaluate, offspring)
    #     for fit, ind in zip(fits, offspring):
    #         ind.fitness.values = fit
    #     pop = toolbox.select(pop + offspring, k=population_size)
    #     hof.update(pop)

        # Save and update Pareto data each generation
        with open(ParetoFront_Path, "w") as f:
            json.dump(
                [{"params": ind, "objectives": ind.fitness.values}
                 for ind in hof if len(ind.fitness.values) == 3],
                 f, indent=4
            )

        # NEW: Save to temp file to trigger UI update
        with open("pareto_update_trigger.json", "w") as f:
            json.dump(
                [{"objectives": list(ind.fitness.values)} for ind in hof if any(v > 1e-5 for v in ind.fitness.values)],
                f,
                indent=4
            )

    print("\nPareto Front saved to pareto_front.json")
    plot_pareto_front(ParetoFront_Path)

    # Save updated training data
    from utils import save_updated_training_data
    save_updated_training_data(X_train, Y_train)

    # --- NEW: return the Pareto front so Graphics.py gets something ---
    return list(hof)

if __name__ == "__main__":
    nsga_ii(THETA_MIN.tolist(), THETA_MAX.tolist(), num_generations=3, population_size=5)
