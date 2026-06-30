import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

df = pd.read_csv("timing_results.csv")
sysc = df[df.truth == "Systolic"]
both = df[df.truth == "Both"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Timing localization via segment-pooled murmur-band energy (40-patient CirCor sample)",
             fontsize=13, fontweight="bold")

# --- Panel 1: systolic_fraction distribution ---
rng = np.random.default_rng(0)
def strip(sub, color, label, marker="o"):
    y = rng.uniform(-0.18, 0.18, len(sub))
    ax1.scatter(sub.sys_frac, y, c=color, s=70, alpha=0.8, edgecolor="black",
                linewidth=0.5, label=label, marker=marker, zorder=3)

corr = sysc[sysc.pred == "Systolic"]
miss = sysc[sysc.pred == "Diastolic"]
strip(corr, "#2ca02c", f"Systolic – correct (n={len(corr)})")
strip(miss, "#d62728", f"Systolic – miss (n={len(miss)})")
strip(both, "#ff7f0e", f"Both (n={len(both)})", marker="D")

ax1.axvline(0.5, color="gray", ls="--", lw=1.5)
ax1.text(0.5, 0.27, "decision boundary", ha="center", color="gray", fontsize=9)
ax1.axvline(sysc.sys_frac.median(), color="#2ca02c", ls=":", lw=1.5)
ax1.text(sysc.sys_frac.median(), -0.30, f"systolic median\n{sysc.sys_frac.median():.2f}",
         ha="center", color="#2ca02c", fontsize=9)
ax1.set_xlim(0, 1); ax1.set_ylim(-0.4, 0.4)
ax1.set_yticks([])
ax1.set_xlabel("systolic fraction  =  E_Sys / (E_Sys + E_Dia)")
ax1.set_title("Energy lands in SYSTOLE for systolic murmurs →")
ax1.legend(loc="upper left", fontsize=9)
ax1.grid(alpha=0.3, axis="x")
ax1.fill_betweenx([-0.4, 0.4], 0.5, 1.0, color="#2ca02c", alpha=0.05)
ax1.fill_betweenx([-0.4, 0.4], 0.0, 0.5, color="#d62728", alpha=0.05)

# --- Panel 2: E_Sys vs E_Dia scatter (log-log) ---
for sub, color, label, marker in [
    (corr, "#2ca02c", "Systolic – correct", "o"),
    (miss, "#d62728", "Systolic – miss", "o"),
    (both, "#ff7f0e", "Both", "D"),
]:
    ax2.scatter(sub.E_Dia, sub.E_Sys, c=color, s=70, alpha=0.8,
                edgecolor="black", linewidth=0.5, label=label, marker=marker, zorder=3)
lims = [1e-4, 1]
ax2.plot(lims, lims, "k--", lw=1, alpha=0.6)
ax2.text(2e-2, 4e-2, "E_Sys = E_Dia", rotation=45, color="gray", fontsize=9)
ax2.set_xscale("log"); ax2.set_yscale("log")
ax2.set_xlim(lims); ax2.set_ylim(lims)
ax2.set_xlabel("Diastolic median energy (E_Dia)")
ax2.set_ylabel("Systolic median energy (E_Sys)")
ax2.set_title("Above the line = systole louder (correct for systolic murmur)")
ax2.legend(loc="lower right", fontsize=9)
ax2.grid(alpha=0.3, which="both")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("timing_localization.png", dpi=130)
print("saved timing_localization.png")
print(f"systolic correct: {len(corr)}/{len(sysc)} = {100*len(corr)/len(sysc):.1f}%")
print(f"systolic sys_frac: mean={sysc.sys_frac.mean():.3f} median={sysc.sys_frac.median():.3f}")
print(f"misses sys_frac: {sorted(miss.sys_frac.round(3).tolist())}")
