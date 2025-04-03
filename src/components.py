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
    # id: str

    def draw(self) -> Any: ...


@dataclass
class BaseComponent:
    component_type: ComponentType

    def draw(self) -> Any:
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BaseComponent":
        return cls(**data)


@dataclass
class TextComponent(BaseComponent):
    component_type: ComponentType = ComponentType.TEXT
    text: Template = ""

    def draw(self) -> Any:
        return st.write(self.text, key=uuid4().hex)


@dataclass
class ChartComponent(BaseComponent):
    component_type: ComponentType = ComponentType.CHART
    chart: Chart = None

    def draw(self) -> Any:
        return st.plotly_chart(self.chart, use_container_width=True, key=uuid4().hex)
