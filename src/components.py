from dataclasses import dataclass
from typing import Protocol, Any
from enum import Enum
import streamlit as st
from charts import Chart
from dataclasses import asdict
from uuid import uuid4
from string import Template


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
