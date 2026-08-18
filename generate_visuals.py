import pandas as pd
from src.visuals import plot_throughput

df = pd.read_csv("results/throughput.csv", sep="\t")
model_names = df["model name"].unique().tolist()
colors = {"BART_4-4-404_66m": "red"}

plot_throughput(
    df,
    model_names,
    colors,
    save_path="results/throughput_bars.png",
)