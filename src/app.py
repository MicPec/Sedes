import streamlit as st
from charts import (
    LineChart,
    BarChart,
    Histogram,
    ScatterChart,
    PieChart,
    BoxPlot,
    ViolinPlot,
    Heatmap,
    AreaChart,
    FunnelChart,
)
import pandas as pd
from components import TextComponent, ChartComponent

# Set page config
st.set_page_config(layout="wide")

st.title(":toilet: SEDES")
st.write("**S**imple & **E**legant **D**ata **E**xplorer **S**ystem")

st.sidebar.title("Settings")

comp1 = TextComponent(text="testujemy TEXT")
comp1.draw()

# Load data
df = pd.read_csv("src/iris.csv")
# st.dataframe(df)

hist = Histogram(
    df,
    name="test",
    x_column="sepal length",
    color="class",
    bins=10,
    title="test",
    params={"barmode": "group"},
)
hist_fig = hist.plot()

comp2 = ChartComponent(chart=hist_fig)
comp2.draw()

comp3 = ChartComponent(chart=hist_fig)
comp3.draw()
