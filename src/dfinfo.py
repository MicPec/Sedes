import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, List


@dataclass
class DataFrameInfo:
    df: pd.DataFrame

    def get_info(self) -> Dict[str, Any]:
        if self.df is None or self.df.empty:
            return {"error": "DataFrame is empty or None"}

        info = {
            "shape": self.get_shape(),
            "columns": self.get_columns(),
            "dtypes": self.get_dtypes(),
            "numeric_stats": self.get_numeric_stats(),
            "missing_values": self.get_missing_values(),
            "sample": self.get_sample(),
        }

        return info

    def get_shape(self) -> Dict[str, int]:
        return {"rows": self.df.shape[0], "columns": self.df.shape[1]}

    def get_columns(self) -> List[str]:
        return self.df.columns.tolist()

    def get_dtypes(self) -> Dict[str, str]:
        return {col: str(dtype) for col, dtype in self.df.dtypes.items()}

    def get_numeric_stats(self) -> Dict[str, Dict[str, float]]:
        numeric_df = self.df.select_dtypes(include=["number"])
        if numeric_df.empty:
            return {}

        stats = {}
        for col in numeric_df.columns:
            stats[col] = {
                "min": float(numeric_df[col].min()),
                "max": float(numeric_df[col].max()),
                "mean": float(numeric_df[col].mean()),
                "median": float(numeric_df[col].median()),
                "std": float(numeric_df[col].std()),
            }
        return stats

    def get_missing_values(self) -> Dict[str, Dict[str, Any]]:
        missing = {}
        for col in self.df.columns:
            null_count = self.df[col].isna().sum()
            missing[col] = {"count": int(null_count), "percentage": float(null_count / len(self.df) * 100)}
        return missing
