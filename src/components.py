from dataclasses import dataclass
from typing import Protocol, Any
from enum import Enum
import streamlit as st
from charts import Chart
from dataclasses import asdict
from uuid import uuid4
from string import Template
import pandas as pd


ComponentType = Enum("ComponentType", ["CHART", "OPERATION", "DATA", "TEXT"])


class Component(Protocol):
    component_type: ComponentType
    id: str
    name: str

    def draw(self) -> Any: ...


@dataclass
class BaseComponent:
    component_type: ComponentType
    id: str = None
    name: str = None

    def __post_init__(self):
        if self.id is None:
            self.id = uuid4().hex
        if self.name is None:
            self.name = f"Component {self.id[:4]}"

    def draw(self) -> Any:
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        """Convert component to a dictionary for serialization."""
        data = asdict(self)
        # Convert enum to string for JSON serialization
        if "component_type" in data and isinstance(data["component_type"], Enum):
            data["component_type"] = data["component_type"].name
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BaseComponent":
        """Create a component from a dictionary."""
        # Convert component_type from string back to enum if needed
        if "component_type" in data and isinstance(data["component_type"], str):
            try:
                data["component_type"] = ComponentType[data["component_type"]]
            except KeyError:
                # Handle case where enum value doesn't exist
                data["component_type"] = ComponentType.CHART  # Default fallback
        return cls(**data)


@dataclass
class TextComponent(BaseComponent):
    component_type: ComponentType = ComponentType.TEXT
    text: Template = ""

    def __post_init__(self):
        super().__post_init__()
        # Use the first few characters of the text as the name
        if self.text:
            text_str = str(self.text)
            self.name = text_str[:20] + "..." if len(text_str) > 20 else text_str

    def draw(self) -> Any:
        return st.write(self.text, key=uuid4().hex)


@dataclass
class ChartComponent(BaseComponent):
    component_type: ComponentType = ComponentType.CHART
    chart: Chart = None

    def draw(self) -> Any:
        return st.plotly_chart(self.chart, use_container_width=True, key=uuid4().hex)


@dataclass
class DataInfoComponent(BaseComponent):
    component_type: ComponentType = ComponentType.DATA
    source_df_id: str = ""
    info_type: str = "preview"  # Options: preview, shape, statistics, column_types, missing_values, all

    def draw(self) -> Any:
        # Get the dataframe
        if not hasattr(st.session_state, "app_state"):
            return st.error("App state not initialized")

        df = st.session_state.app_state.get_dataframe_by_id(self.source_df_id)
        if df is None:
            return st.error(f"DataFrame not found: {self.source_df_id}")

        # Display the selected information
        if self.info_type == "preview":
            return st.dataframe(df, key=uuid4().hex)

        elif self.info_type == "shape":
            st.write(f"**Shape**: {df.shape[0]} rows × {df.shape[1]} columns")

        elif self.info_type == "statistics":
            st.write("### Numeric Columns Statistics")
            return st.dataframe(df.describe(), key=uuid4().hex)

        elif self.info_type == "column_types":
            st.write("### Column Types")
            dtypes_df = pd.DataFrame({"Data Type": [str(dtype) for dtype in df.dtypes]}, index=df.columns)
            dtypes_df.index.name = "Column"
            return st.dataframe(dtypes_df.reset_index(), key=uuid4().hex)

        elif self.info_type == "missing_values":
            st.write("### Missing Values")
            missing_df = pd.DataFrame(
                {"Missing Values": df.isna().sum(), "Percentage": (df.isna().sum() / len(df) * 100).round(2)}
            )
            missing_df.index.name = "Column"
            return st.dataframe(missing_df.reset_index(), key=uuid4().hex)

        else:
            return st.error(f"Invalid info type: {self.info_type}")
