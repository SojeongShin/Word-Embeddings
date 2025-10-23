import re, pandas as pd, matplotlib.pyplot as plt

pattern = re.compile(r"epoch\s+(\d+)\s+step\s+(\d+)\s+loss\s+([0-9.]+)")
rows = []
with open("log.txt", "r") as f:
    for line in f:
        m = pattern.search(line)
        if m:
            epoch, step, loss = int(m.group(1)), int(m.group(2)), float(m.group(3))
            rows.append({"epoch": epoch, "step": step, "loss": loss})
df = pd.DataFrame(rows).sort_values(["epoch","step"])
df["global_step"] = range(1, len(df)+1)

# CSV로 저장
df.to_csv("learning_curve.csv", index=False)

# 그래프
plt.figure()
plt.plot(df["global_step"], df["loss"])
plt.xlabel("Global Step"); plt.ylabel("Loss"); plt.title("Learning Curve")
plt.tight_layout(); plt.savefig("learning_curve.png", dpi=150)
