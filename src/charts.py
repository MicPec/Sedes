import pandas as pd
import plotly.express as px
import plotly.graph_objects as go  # Keep for Figure type hints
from typing import Dict, List, Any, Protocol
from dataclasses import dataclass, asdict, field
from uuid import uuid4


class Chart(Protocol):
    """Protocol for chart objects"""

    def plot(self, df: pd.DataFrame): ...

    def to_dict(self) -> Dict[str, Any]: ...

    @classmethod
    def from_dict(cls, data: Dict[str, Any], df: pd.DataFrame = None) -> "Chart":
        """
        Create a Chart object from a serialized dictionary.

        Args:
            data (Dict[str, Any]): The serialized chart data.
            df (pd.DataFrame, optional): The DataFrame to use with the chart. Defaults to None.

        Returns:
            Chart: The constructed Chart object.
        """
        ...


@dataclass
class BaseChart:
    df: pd.DataFrame
    id: str = uuid4().hex
    chart_type: str = ""
    name: str = ""
    x_column: str = ""
    y_column: str = ""
    title: str = ""
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def to_dict(self) -> Dict[str, Any]:
        # Only serialize the configuration, not the DataFrame
        data = asdict(self)
        data.pop("df", None)  # Remove DataFrame from serialization
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any], df: pd.DataFrame = None) -> "BaseChart":
        chart_map = {
            "line": LineChart,
            "bar": BarChart,
            "scatter": ScatterChart,
            "histogram": Histogram,
            "box": BoxPlot,
            "violin": ViolinPlot,
            "heatmap": Heatmap,
            "area": AreaChart,
            "funnel": FunnelChart,
        }
        chart_type = data.pop("chart_type", "")
        if df is not None:
            data["df"] = df
        if chart_type in chart_map:
            return chart_map[chart_type](**data)
        return cls(**data)

    @classmethod
    def _get_name(cls) -> str:
        return cls.__name__


@dataclass
class LineChart(BaseChart):
    chart_type: str = "line"
    hue_column: str = ""

    def plot(self) -> px.line:
        fig = px.line(
            self.df,
            x=self.x_column,
            y=self.y_column,
            color=self.hue_column if self.hue_column else None,
            title=self.title,
            **self.params,
        )
        return fig


@dataclass
class BarChart(BaseChart):
    chart_type: str = "bar"
    group_by: str = ""

    def plot(self) -> px.bar:
        if self.group_by:
            grouped = self.df.groupby([self.x_column, self.group_by])[self.y_column].mean().reset_index()
        else:
            grouped = self.df.groupby([self.x_column])[self.y_column].mean().reset_index()

        fig = px.bar(
            grouped,
            x=self.x_column,
            y=self.y_column,
            color=self.group_by if self.group_by else None,
            # barmode="group",
            title=self.title,
            **self.params,
        )
        return fig


@dataclass
class Histogram(BaseChart):
    chart_type: str = "histogram"
    color: str = ""
    bins: int = 10

    def plot(self) -> px.histogram:
        fig = px.histogram(
            self.df,
            x=self.x_column,
            nbins=self.bins,
            title=self.title,
            color=self.color if self.color else None,
            # barmode="group",
            **self.params,
        )
        return fig


@dataclass
class ScatterChart(BaseChart):
    chart_type: str = "scatter"
    hue_column: str = ""
    size_column: str = ""

    def plot(self) -> px.scatter:
        fig = px.scatter(
            self.df,
            x=self.x_column,
            y=self.y_column,
            color=self.hue_column if self.hue_column else None,
            size=self.size_column if self.size_column else None,
            title=self.title,
            **self.params,
        )
        return fig


@dataclass
class PieChart(BaseChart):
    chart_type: str = "pie"
    group_by: str = ""

    def plot(self) -> px.pie:
        fig = px.pie(self.df, names=self.group_by, title=self.title, **self.params)
        return fig


@dataclass
class BoxPlot(BaseChart):
    chart_type: str = "boxplot"
    color: str = ""

    def plot(self) -> px.box:
        fig = px.box(
            self.df,
            x=self.x_column,
            y=self.y_column,
            color=self.color if self.color else None,
            title=self.title,
            **self.params,
        )
        return fig


@dataclass
class ViolinPlot(BaseChart):
    chart_type: str = "violin"
    color: str = ""

    def plot(self) -> px.violin:
        fig = px.violin(
            self.df,
            x=self.x_column,
            y=self.y_column,
            color=self.color if self.color else None,
            title=self.title,
            **self.params,
        )
        return fig


@dataclass
class Heatmap(BaseChart):
    chart_type: str = "heatmap"
    z_column: str = ""

    def plot(self) -> px.density_heatmap:
        fig = px.density_heatmap(
            self.df,
            x=self.x_column,
            y=self.y_column,
            z=self.z_column if self.z_column else None,
            title=self.title,
            **self.params,
        )
        return fig


@dataclass
class AreaChart(BaseChart):
    chart_type: str = "area"
    color: str = ""

    def plot(self) -> px.area:
        fig = px.area(
            self.df,
            x=self.x_column,
            y=self.y_column,
            color=self.color if self.color else None,
            title=self.title,
            **self.params,
        )
        return fig


@dataclass
class SunburstChart(BaseChart):
    chart_type: str = "sunburst"
    color: str = ""

    def plot(self) -> px.sunburst:
        fig = px.sunburst(
            self.df,
            path=[self.x_column, self.y_column],
            color=self.color if self.color else None,
            title=self.title,
            **self.params,
        )
        return fig


@dataclass
class FunnelChart(BaseChart):
    chart_type: str = "funnel"
    color: str = ""

    def plot(self) -> px.funnel:
        fig = px.funnel(
            self.df,
            x=self.x_column,
            y=self.y_column,
            color=self.color if self.color else None,
            title=self.title,
            **self.params,
        )
        return fig
