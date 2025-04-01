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

# Set page config
st.set_page_config(layout="wide")

st.title(":toilet: SEDES")
st.write("**S**imple & **E**legant **D**ata **E**xplorer **S**ystem")

st.sidebar.title("Settings")


# Load data
df = pd.read_csv("src/iris.csv")
st.dataframe(df)

# Create and plot line chart
linechart = LineChart(
    name="test",
    id="1",
    chart_type="line",
    x_column="sepal length",
    y_column="petal width",
    hue_column="class",
    title="test",
)
line_fig = linechart.plot(df)

# Create and plot bar chart
barchart = BarChart(
    name="test",
    id="2",
    chart_type="bar",
    x_column="sepal length",
    y_column="sepal width",
    group_by="class",
    title="test",
)
bar_fig = barchart.plot(df)


histchart = Histogram(
    name="test",
    id="3",
    chart_type="histogram",
    x_column="sepal width",
    bins=6,
    title="test",
)
hist_fig = histchart.plot(df)

# Create and plot scatter chart
scatterchart = ScatterChart(
    name="test",
    id="4",
    chart_type="scatter",
    x_column="sepal length",
    y_column="sepal width",
    hue_column="class",
    size_column="petal length",
    title="test",
)
scatter_fig = scatterchart.plot(df)

# Create and plot pie chart
piechart = PieChart(
    name="test",
    id="5",
    chart_type="pie",
    group_by="class",
    title="test",
)
pie_fig = piechart.plot(df)

# Create and plot box plot
boxchart = BoxPlot(
    name="test",
    id="6",
    chart_type="boxplot",
    x_column="sepal length",
    y_column="class",
    color="class",
    title="test",
)
box_fig = boxchart.plot(df)

# Create and plot violin plot
violinchart = ViolinPlot(
    name="test",
    id="7",
    chart_type="violin",
    x_column="class",
    y_column="sepal width",
    color="class",
    title="test",
)
violin_fig = violinchart.plot(df)

# Create and plot heatmap
heatmapchart = Heatmap(
    name="test",
    id="8",
    chart_type="heatmap",
    x_column="sepal length",
    y_column="sepal width",
    # z_column="petal length",
    title="test",
)
heatmap_fig = heatmapchart.plot(df)

# Create and plot area chart
areachart = AreaChart(
    name="test",
    id="9",
    chart_type="area",
    x_column="sepal width",
    y_column="sepal length",
    color="class",
    title="test",
)
area_fig = areachart.plot(df)


# Create and plot funnel chart
funnelchart = FunnelChart(
    name="test",
    id="11",
    chart_type="funnel",
    x_column="sepal width",
    y_column="class",
    color="class",
    title="test",
)
funnel_fig = funnelchart.plot(df)

st.plotly_chart(funnel_fig, use_container_width=True)

# st.plotly_chart(line_fig, use_container_width=True)
# st.plotly_chart(bar_fig, use_container_width=True)
# st.plotly_chart(hist_fig, use_container_width=True)
# st.plotly_chart(scatter_fig, use_container_width=True)
# st.plotly_chart(pie_fig, use_container_width=True)
# st.plotly_chart(box_fig, use_container_width=True)
# st.plotly_chart(violin_fig, use_container_width=True)
st.plotly_chart(heatmap_fig, use_container_width=True)
st.plotly_chart(area_fig, use_container_width=True)
