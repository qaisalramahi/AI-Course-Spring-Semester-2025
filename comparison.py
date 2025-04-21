# comparison.py  – clean timing for Q‑learning
import importlib, statistics, time, textwrap
import matplotlib.pyplot as plt
from environment import ComplicatedRaceTrackEnvPygame

plt.style.use("ggplot")

ALGORITHMS = {
    "BFS":      "breadth-first-search",
    "DFS":      "depth-first-search",
    "A*":       "a_star_algorithm",
    "Q‑Learn":  "q-learning",          # cached q_table.npy expected
}

RUNS      = 100
GRID_SIZE = (15, 15)


# ---------- warm‑up + measure training ----------

train_times = {}
for name, mod_name in ALGORITHMS.items():
    env  = ComplicatedRaceTrackEnvPygame(grid_size=GRID_SIZE)
    mod  = importlib.import_module(mod_name)
    solve = getattr(mod, "solve")

    t0 = time.perf_counter()
    solve(env)                    # trains if necessary, else just loads
    train_times[name] = time.perf_counter() - t0


# ---------- warm‑up pass (training / caching only) ----------
for mod_name in ALGORITHMS.values():
    env = ComplicatedRaceTrackEnvPygame(grid_size=GRID_SIZE)
    importlib.import_module(mod_name).solve(env)   # discard result

# ---------- timed benchmark ----------
results = {name: {"steps": [], "seconds": []} for name in ALGORITHMS}

for name, mod_name in ALGORITHMS.items():
    mod   = importlib.import_module(mod_name)
    solve = getattr(mod, "solve")
    for _ in range(RUNS):
        env = ComplicatedRaceTrackEnvPygame(grid_size=GRID_SIZE)
        t0  = time.perf_counter()
        path = solve(env)
        dt  = time.perf_counter() - t0
        results[name]["seconds"].append(dt)
        results[name]["steps"].append(len(path))
        
        
# ---------- table + capture lines ----------
print(f"{'Alg':8s}  train(s)  exec(ms)  steps(avg)")
table_lines = []
for name in ALGORITHMS:
    train_s = train_times[name]
    exec_ms = statistics.mean(results[name]["seconds"]) * 1000
    steps   = statistics.mean(results[name]["steps"])
    line    = f"{name:8s}  {train_s:8.2f}  {exec_ms:8.2f}  {steps:9.2f}"
    print(line)
    table_lines.append(line)


# ------------ single grouped bar chart (modern styling) ------------
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "figure.dpi": 110,
    "axes.edgecolor": "#444",
    "axes.grid"    : True,
    "grid.color"   : "#ddd",
    "grid.linestyle": "--",
})

labels  = list(ALGORITHMS.keys())
x       = np.arange(len(labels))
width   = 0.24                       # bar width

# data
train_ms = [train_times[k]*1000 for k in labels]                     # s→ms
exec_ms  = [statistics.mean(results[k]["seconds"])*1000 for k in labels]
steps    = [statistics.mean(results[k]["steps"])          for k in labels]

# colours (flat UI palette)
c_exec, c_train, c_steps = "#268bd2", "#f39c12", "#2ecc71"

fig, ax1 = plt.subplots(figsize=(9, 5))
ax2 = ax1.twinx()                  # secondary axis for steps

# bars
b_exec  = ax1.bar(x - width, exec_ms,  width, label="Exec ms",  color=c_exec)
b_train = ax1.bar(x, train_ms, width, label="Train ms", color=c_train)
b_steps = ax2.bar(x + width,  steps,    width, label="Steps",    color=c_steps)

# value‑labels
for bars, ax in ((b_exec, ax1), (b_train, ax1), (b_steps, ax2)):
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)

# axes styling
ax1.set_xlabel("Algorithm")
ax1.set_xticks(x, labels)
ax1.set_ylabel("Time (ms)")
ax2.set_ylabel("Path length (steps)")
ax1.set_ylim(0, max(train_ms + exec_ms)*1.1)
ax2.set_ylim(0, max(steps)*1.15)

# legend (combine both axes’ handles)
handles = [b_exec[0], b_train[0], b_steps[0]]
labelsL = ["Execution (ms)", "Training (ms)", "Steps"]
ax1.legend(handles, labelsL, frameon=False, loc="upper left")

plt.title(f"Training vs Execution vs Steps ({RUNS} runs)")
plt.tight_layout()
plt.show()
