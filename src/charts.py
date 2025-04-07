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
    # name: str = ""
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
        # Build parameters dict with only non-empty values
        params = {
            "x": self.x_column,
            "y": self.y_column,
            "title": self.title if self.title else None,
        }

        # Only add color if it's set
        if self.hue_column:
            params["color"] = self.hue_column

        # Add any extra parameters
        params.update(self.params)

        fig = px.line(self.df, **params)
        return fig


@dataclass
class BarChart(BaseChart):
    chart_type: str = "bar"
    group_by: str = ""

    def plot(self) -> px.bar:
        # Group the data appropriately
        if self.group_by:
            grouped = self.df.groupby([self.x_column, self.group_by])[self.y_column].mean().reset_index()
        else:
            grouped = self.df.groupby([self.x_column])[self.y_column].mean().reset_index()

        # Build parameters dict with only non-empty values
        params = {
            "x": self.x_column,
            "y": self.y_column,
            "title": self.title if self.title else None,
        }

        # Only add color if group_by is set
        if self.group_by:
            params["color"] = self.group_by

        # Add any extra parameters
        params.update(self.params)

        fig = px.bar(grouped, **params)
        return fig


@dataclass
class Histogram(BaseChart):
    chart_type: str = "histogram"
    color: str = ""
    bins: int = 10

    def plot(self) -> px.histogram:
        # Build parameters dict with only non-empty values
        params = {
            "x": self.x_column,
            "nbins": self.bins,
            "title": self.title if self.title else None,
        }

        # Only add color if it's set
        if self.color:
            params["color"] = self.color

        # Add any extra parameters
        params.update(self.params)

        fig = px.histogram(self.df, **params)
        return fig


@dataclass
class ScatterChart(BaseChart):
    chart_type: str = "scatter"
    hue_column: str = ""
    size_column: str = ""

    def plot(self) -> px.scatter:
        # Build parameters dict with only non-empty values
        params = {
            "x": self.x_column,
            "y": self.y_column,
            "title": self.title if self.title else None,
        }

        # Only add color and size if they're set
        if self.hue_column:
            params["color"] = self.hue_column
        if self.size_column:
            params["size"] = self.size_column

        # Add any extra parameters
        params.update(self.params)

        fig = px.scatter(self.df, **params)
        return fig


@dataclass
class PieChart(BaseChart):
    chart_type: str = "pie"
    group_by: str = ""
    values_column: str = ""

    def plot(self) -> px.pie:
        # Build parameters dict with only non-empty values
        params = {
            "names": self.group_by,
            "title": self.title if self.title else None,
        }
        
        # Add values column if specified
        if self.values_column:
            params["values"] = self.values_column

        # Add any extra parameters
        params.update(self.params)

        fig = px.pie(self.df, **params)
        return fig


@dataclass
class BoxPlot(BaseChart):
    chart_type: str = "boxplot"
    color: str = ""

    def plot(self) -> px.box:
        # Build parameters dict with only non-empty values
        params = {
            "x": self.x_column,
            "y": self.y_column,
            "title": self.title if self.title else None,
        }

        # Only add color if it's set
        if self.color:
            params["color"] = self.color

        # Add any extra parameters
        params.update(self.params)

        fig = px.box(self.df, **params)
        return fig


@dataclass
class ViolinPlot(BaseChart):
    chart_type: str = "violin"
    color: str = ""

    def plot(self) -> px.violin:
        # Build parameters dict with only non-empty values
        params = {
            "x": self.x_column,
            "y": self.y_column,
            "title": self.title if self.title else None,
        }

        # Only add color if it's set
        if self.color:
            params["color"] = self.color

        # Add any extra parameters
        params.update(self.params)

        fig = px.violin(self.df, **params)
        return fig


@dataclass
class Heatmap(BaseChart):
    chart_type: str = "heatmap"
    z_column: str = ""

    def plot(self) -> px.density_heatmap:
        # Build parameters dict with only non-empty values
        params = {
            "x": self.x_column,
            "y": self.y_column,
            "title": self.title if self.title else None,
        }

        # Only add z if it's set
        if self.z_column:
            params["z"] = self.z_column

        # Add any extra parameters
        params.update(self.params)

        fig = px.density_heatmap(self.df, **params)
        return fig


@dataclass
class AreaChart(BaseChart):
    chart_type: str = "area"
    color: str = ""

    def plot(self) -> px.area:
        # Build parameters dict with only non-empty values
        params = {
            "x": self.x_column,
            "y": self.y_column,
            "title": self.title if self.title else None,
        }

        # Only add color if it's set
        if self.color:
            params["color"] = self.color

        # Add any extra parameters
        params.update(self.params)

        fig = px.area(self.df, **params)
        return fig


@dataclass
class SunburstChart(BaseChart):
    chart_type: str = "sunburst"
    color: str = ""

    def plot(self) -> px.sunburst:
        # Build parameters dict with only non-empty values
        params = {
            "path": [self.x_column, self.y_column],
            "title": self.title if self.title else None,
        }

        # Only add color if it's set
        if self.color:
            params["color"] = self.color

        # Add any extra parameters
        params.update(self.params)

        fig = px.sunburst(self.df, **params)
        return fig


@dataclass
class FunnelChart(BaseChart):
    chart_type: str = "funnel"
    color: str = ""

    def plot(self) -> px.funnel:
        # Build parameters dict with only non-empty values
        params = {
            "x": self.x_column,
            "y": self.y_column,
            "title": self.title if self.title else None,
        }

        # Only add color if it's set
        if self.color:
            params["color"] = self.color

        # Add any extra parameters
        params.update(self.params)

        fig = px.funnel(self.df, **params)
        return fig


chart_types = {
    "Line Chart": LineChart,
    "Bar Chart": BarChart,
    "Histogram": Histogram,
    "Scatter Chart": ScatterChart,
    "Pie Chart": PieChart,
    "Box Plot": BoxPlot,
    "Violin Plot": ViolinPlot,
    "Heatmap": Heatmap,
    "Area Chart": AreaChart,
    "Funnel Chart": FunnelChart,
}
