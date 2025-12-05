import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("sandbox_runs.csv")

# IR 越小越好，用 1 - ir_gt_1 做纵轴
df["ir_good"] = 1.0 - df["ir_gt_1"]

# 综合评分：SLO 高 + 高干扰少（你也可以改成其它权重）
df["score"] = df["slo_rate"] + df["ir_good"]

topk = 10          # 高亮几个
top_annotate = 5   # 标注几个 ID

best = df.nlargest(topk, "score")

print("Top configs:")
print(best[[
    "label", "limit_threshold", "bandwidth_refill",
    "compute_ceiling", "max_boost", "decay", "adjust_interval",
    "slo_rate", "ir_gt_1", "score"
]])

# 根据数据范围缩放坐标轴
pad = 0.06
xmin = df["slo_rate"].min()  - pad
xmax = df["slo_rate"].max()  + pad
ymin = df["ir_good"].min()   - pad
ymax = df["ir_good"].max()   + pad

plt.figure(figsize=(6, 4.5))

# 所有组合（浅灰小点）
plt.scatter(
    df["slo_rate"],
    df["ir_good"],
    alpha=0.25,
    s=18,
    color="lightgray",
    label="All configs",
)

# Top-k 组合（蓝色稍大点）
plt.scatter(
    best["slo_rate"],
    best["ir_good"],
    color="tab:blue",
    s=30,
    label=f"Top {topk} configs",
)

# 只给前 top_annotate 个加一个小号 label，不加大框，避免遮挡
for _, row in best.head(top_annotate).iterrows():
    plt.text(
        row["slo_rate"] + 0.0005,   # 稍微右移一点点
        row["ir_good"] + 0.0005,    # 稍微上移一点点
        row["label"],
        fontsize=7,
        ha="left",
        va="bottom"
    )

plt.xlabel("SLO hit rate")
plt.ylabel("1 - fraction of requests with IR > 1")
plt.title("Sandbox parameter combinations")
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig("sandbox_param_scatter_ir1_clean.png", dpi=300)
plt.show()