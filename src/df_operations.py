import pandas as pd
from typing import Dict, List, Any
from dataclasses import dataclass, field, asdict


@dataclass
class BaseDfOperation:
    name: str
    id: str
    operation_type: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseDfOperation":
        return cls(**data)


@dataclass
class FilterOperation(BaseDfOperation):
    column: str = ""
    filter_type: str = ""  # "equals", "contains", "greater_than", etc.
    filter_value: Any = None
    operation_type: str = "filter"

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.column or not self.filter_type:
            return df

        case_map = {
            "equals": lambda x: x == self.filter_value,
            "contains": lambda x: x.astype(str).str.contains(str(self.filter_value), na=False),
            "greater_than": lambda x: x > self.filter_value,
            "less_than": lambda x: x < self.filter_value,
            "between": lambda x: (x >= self.filter_value[0]) & (x <= self.filter_value[1]),
        }

        if self.filter_type not in case_map:
            return df

        return df[df[self.column].apply(case_map[self.filter_type])]


@dataclass
class AggregateOperation(BaseDfOperation):
    group_by_columns: List[str] = field(default_factory=list)
    agg_functions: Dict[str, str] = field(default_factory=dict)  # column: agg_func
    operation_type: str = "aggregate"

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.group_by_columns or not self.agg_functions:
            return df

        return df.groupby(self.group_by_columns).agg(self.agg_functions).reset_index()


@dataclass
class LoadCsvOperation(BaseDfOperation):
    file_path: str = ""
    operation_type: str = "load_csv"

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.read_csv(self.file_path)


@dataclass
class SaveCsvOperation(BaseDfOperation):
    file_path: str = ""
    operation_type: str = "save_csv"

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        df.to_csv(self.file_path, index=False)
        return df
