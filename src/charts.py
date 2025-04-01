import pandas as pd
import plotly.express as px
import plotly.graph_objects as go  # Keep for Figure type hints
from typing import Dict, List, Any, Protocol
from dataclasses import dataclass, asdict


class Chart(Protocol):
    """Protocol for chart objects"""

    def plot(self, df: pd.DataFrame): ...

    def to_dict(self) -> Dict[str, Any]: ...


@dataclass
class BaseChart:
    name: str
    id: str
    chart_type: str
    x_column: str = ""
    y_column: str = ""
    title: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseChart":
        return cls(**data)


@dataclass
class LineChart(BaseChart):
    name: str
    chart_type: str = "line"
    hue_column: str = ""

    def plot(self, df: pd.DataFrame) -> px.line:
        fig = px.line(
            df,
            x=self.x_column,
            y=self.y_column,
            color=self.hue_column if self.hue_column else None,
            title=self.title,
            markers=True,
        )
        return fig


@dataclass
class BarChart(BaseChart):
    chart_type: str = "bar"
    group_by: str = ""

    def plot(self, df: pd.DataFrame, fig=None) -> px.bar:
        if self.group_by:
            grouped = df.groupby([self.x_column, self.group_by])[self.y_column].mean().reset_index()
        else:
            grouped = df.groupby([self.x_column])[self.y_column].mean().reset_index()

        fig = px.bar(
            grouped,
            x=self.x_column,
            y=self.y_column,
            color=self.group_by if self.group_by else None,
            barmode="group",
            title=self.title,
        )
        return fig


@dataclass
class Histogram(BaseChart):
    chart_type: str = "histogram"
    bins: int = 10

    def plot(self, df: pd.DataFrame, fig=None) -> px.histogram:
        if fig is None:
            fig = px.histogram(
                df,
                x=self.x_column,
                nbins=self.bins,
                title=self.title,
            )
        return fig


@dataclass
class ScatterChart(BaseChart):
    chart_type: str = "scatter"
    hue_column: str = ""
    size_column: str = ""

    def plot(self, df: pd.DataFrame) -> px.scatter:
        fig = px.scatter(
            df,
            x=self.x_column,
            y=self.y_column,
            color=self.hue_column if self.hue_column else None,
            size=self.size_column if self.size_column else None,
            title=self.title,
        )
        return fig


@dataclass
class PieChart(BaseChart):
    chart_type: str = "pie"
    group_by: str = ""

    def plot(self, df: pd.DataFrame) -> px.pie:
        fig = px.pie(
            df,
            names=self.group_by,
            title=self.title,
        )
        return fig


@dataclass
class BoxPlot(BaseChart):
    chart_type: str = "boxplot"
    color: str = ""

    def plot(self, df: pd.DataFrame) -> px.box:
        fig = px.box(
            df,
            x=self.x_column,
            y=self.y_column,
            color=self.color if self.color else None,
            title=self.title,
        )
        return fig


@dataclass
class ViolinPlot(BaseChart):
    chart_type: str = "violin"
    color: str = ""

    def plot(self, df: pd.DataFrame) -> px.violin:
        fig = px.violin(
            df,
            x=self.x_column,
            y=self.y_column,
            color=self.color if self.color else None,
            title=self.title,
        )
        return fig


@dataclass
class Heatmap(BaseChart):
    chart_type: str = "heatmap"
    z_column: str = ""

    def plot(self, df: pd.DataFrame) -> px.density_heatmap:
        fig = px.density_heatmap(
            df,
            x=self.x_column,
            y=self.y_column,
            z=self.z_column if self.z_column else None,
            title=self.title,
        )
        return fig


@dataclass
class AreaChart(BaseChart):
    chart_type: str = "area"
    color: str = ""

    def plot(self, df: pd.DataFrame) -> px.area:
        fig = px.area(
            df,
            x=self.x_column,
            y=self.y_column,
            color=self.color if self.color else None,
            title=self.title,
        )
        return fig


@dataclass
class SunburstChart(BaseChart):
    chart_type: str = "sunburst"
    color: str = ""

    def plot(self, df: pd.DataFrame) -> px.sunburst:
        fig = px.sunburst(
            df,
            path=[self.x_column, self.y_column],
            color=self.color if self.color else None,
            title=self.title,
        )
        return fig


@dataclass
class FunnelChart(BaseChart):
    chart_type: str = "funnel"
    color: str = ""

    def plot(self, df: pd.DataFrame) -> px.funnel:
        fig = px.funnel(
            df,
            x=self.x_column,
            y=self.y_column,
            color=self.color if self.color else None,
            title=self.title,
        )
        return fig
