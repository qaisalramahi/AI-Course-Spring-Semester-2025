# comparison.py  ── pretty version
import importlib, statistics, time, textwrap
import matplotlib.pyplot as plt
from environment import ComplicatedRaceTrackEnvPygame

plt.style.use("ggplot")          # modern look without extra libs

ALGORITHMS = {
    "BFS": "breadth-first-search",
    "DFS": "depth-first-search",
    "A*":  "a_star_algorithm",
    # "Q‑Learn": "q_learning",
}

RUNS       = 100
GRID_SIZE  = (10, 10)

results = {name: {"steps": [], "seconds": []} for name in ALGORITHMS}

for name, module_name in ALGORITHMS.items():
    mod   = importlib.import_module(module_name)
    solve = getattr(mod, "solve")

    for _ in range(RUNS):
        env = ComplicatedRaceTrackEnvPygame(grid_size=GRID_SIZE)
        t0  = time.perf_counter()
        path = solve(env)
        dt   = time.perf_counter() - t0

        results[name]["steps"].append(len(path))
        results[name]["seconds"].append(dt)

# ---------- summary table (stdout) ----------
print(f"{'Alg':8s}  steps(avg)  time(ms)")
table_lines = []
for name in ALGORITHMS:
    stp = statistics.mean(results[name]["steps"])
    tms = statistics.mean(results[name]["seconds"]) * 1000
    line = f"{name:8s}  {stp:9.2f}  {tms:7.2f}"
    print(line)
    table_lines.append(line)

# concatenate for figure textbox
table_text = "```\n" + "\n".join(table_lines) + "\n```"

# ---------- plotting ----------
labels = list(ALGORITHMS.keys())
avg_steps = [statistics.mean(results[k]["steps"])    for k in labels]
avg_time  = [statistics.mean(results[k]["seconds"])*1000 for k in labels]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 8.5))

# --- steps bar ---
bars1 = ax1.bar(labels, avg_steps)
ax1.set_title("Average Path Length (steps)")
ax1.set_ylabel("steps")
ax1.bar_label(bars1, fmt="%.1f", padding=3)

# --- time bar ---
bars2 = ax2.bar(labels, avg_time)
ax2.set_title(f"Average Wall‑clock Time over {RUNS} runs")
ax2.set_ylabel("milliseconds")
ax2.bar_label(bars2, fmt="%.2f ms", padding=3)

# --- add table as a textbox on the right ---
fig.text(0.70, 0.50, textwrap.dedent(table_text),
         family="monospace", fontsize=9,
         bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.85))

plt.tight_layout(rect=[0,0,0.68,1])   # leave space for table
plt.show()
