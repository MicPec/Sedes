import streamlit as st
import nbformat
import pandas as pd
from df_operations import FilterOperation, AggregateOperation, LoadCsvOperation, DataCleanOperation

def generate_notebook_cells():
    """Generate Jupyter notebook cells based on the current operations"""
    cells = []
    
    # Add imports cell
    imports = """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
    """
    cells.append(nbformat.v4.new_code_cell(imports))
    
    # Get operations by type
    load_ops = [op for op in st.session_state.app_state.operations if isinstance(op, LoadCsvOperation)]
    filter_ops = [op for op in st.session_state.app_state.operations if isinstance(op, FilterOperation)]
    clean_ops = [op for op in st.session_state.app_state.operations if isinstance(op, DataCleanOperation)]
    agg_ops = [op for op in st.session_state.app_state.operations if isinstance(op, AggregateOperation)]
    
    # Add load operations
    if load_ops:
        cells.append(nbformat.v4.new_markdown_cell("## Data Loading"))
        
        for op in load_ops:
            code = f"# Load dataset: {op.name}\n"
            code += f"df_{op.id[:8]} = pd.read_csv('{op.file_path}', sep='{op.sep}')\n"
            code += f"print(f'Loaded dataframe shape: {{df_{op.id[:8]}.shape}}')"
            cells.append(nbformat.v4.new_code_cell(code))
    
    # Add filter operations
    if filter_ops:
        cells.append(nbformat.v4.new_markdown_cell("## Data Filtering"))
        
        for op in filter_ops:
            source_df = f"df_{op.source_df_id[:8]}"
            result_df = f"df_{op.id[:8]}"
            
            code = f"# Apply filter: {op.name}\n"
            
            # Handle multi-filter case
            if hasattr(op, 'filters') and op.filters:
                code += f"{result_df} = {source_df}.copy()\n"
                
                for i, filter_item in enumerate(op.filters):
                    col = filter_item.get('column')
                    filter_type = filter_item.get('filter_type')
                    filter_value = filter_item.get('filter_value')
                    
                    if filter_type == "equals":
                        code += f"{result_df} = {result_df}[{result_df}['{col}'] == {repr(filter_value)}]\n"
                    elif filter_type == "not_equals":
                        code += f"{result_df} = {result_df}[{result_df}['{col}'] != {repr(filter_value)}]\n"
                    elif filter_type == "contains":
                        code += f"{result_df} = {result_df}[{result_df}['{col}'].astype(str).str.contains({repr(str(filter_value))})]\n"
                    elif filter_type == "greater_than":
                        code += f"{result_df} = {result_df}[{result_df}['{col}'] > {repr(filter_value)}]\n"
                    elif filter_type == "less_than":
                        code += f"{result_df} = {result_df}[{result_df}['{col}'] < {repr(filter_value)}]\n"
            else:
                # Handle single filter case
                if op.filter_type == "equals":
                    code += f"{result_df} = {source_df}[{source_df}['{op.column}'] == {repr(op.filter_value)}]\n"
                elif op.filter_type == "not_equals":
                    code += f"{result_df} = {source_df}[{source_df}['{op.column}'] != {repr(op.filter_value)}]\n"
                elif op.filter_type == "contains":
                    code += f"{result_df} = {source_df}[{source_df}['{op.column}'].astype(str).str.contains({repr(str(op.filter_value))})]\n"
                elif op.filter_type == "greater_than":
                    code += f"{result_df} = {source_df}[{source_df}['{op.column}'] > {repr(op.filter_value)}]\n"
                elif op.filter_type == "less_than":
                    code += f"{result_df} = {source_df}[{source_df}['{op.column}'] < {repr(op.filter_value)}]\n"
            
            code += f"print(f'Filtered dataframe shape: {{{{result_df}}}}.shape')"
            cells.append(nbformat.v4.new_code_cell(code))
    
    # Add data cleaning operations
    if clean_ops:
        cells.append(nbformat.v4.new_markdown_cell("## Data Cleaning"))
        
        for op in clean_ops:
            source_df = f"df_{op.source_df_id[:8]}"
            result_df = f"df_{op.id[:8]}"
            
            code = f"# Apply data cleaning: {op.name}\n"
            code += f"{result_df} = {source_df}.copy()\n"
            
            if hasattr(op, 'drop_columns') and op.drop_columns:
                code += f"{result_df} = {result_df}.drop(columns={repr(op.drop_columns)})\n"
            
            if hasattr(op, 'drop_na') and op.drop_na:
                code += f"{result_df} = {result_df}.dropna()\n"
                
            if hasattr(op, 'rename_columns') and op.rename_columns:
                renames = {old: new for old, new in op.rename_columns.items() if old != new}
                if renames:
                    code += f"{result_df} = {result_df}.rename(columns={repr(renames)})\n"
            
            code += f"print(f'Cleaned dataframe shape: {{{{result_df}}}}.shape')"
            cells.append(nbformat.v4.new_code_cell(code))
    
    # Add aggregation operations
    if agg_ops:
        cells.append(nbformat.v4.new_markdown_cell("## Data Aggregation"))
        
        for op in agg_ops:
            source_df = f"df_{op.source_df_id[:8]}"
            result_df = f"df_{op.id[:8]}"
            
            code = f"# Apply aggregation: {op.name}\n"
            
            code += f"{result_df} = {source_df}.groupby('{op.group_by}').agg({{\n"
            if hasattr(op, 'aggregations'):
                for col, func in op.aggregations.items():
                    code += f"    '{col}': '{func}',\n"
            elif hasattr(op, 'agg_func'):
                for col, func in op.agg_func.items():
                    code += f"    '{col}': '{func}',\n"
            code += "}).reset_index()\n"
            
            code += f"print(f'Aggregated dataframe shape: {{{{result_df}}}}.shape')"
            cells.append(nbformat.v4.new_code_cell(code))
    
    # Add visualization section
    cells.append(nbformat.v4.new_markdown_cell("## Visualizations"))
    # Get the current dataframe
    current_df_id = st.session_state.app_state.current_df_id
    if current_df_id:
        df_var = f"df_{current_df_id[:8]}"
        
        # Add basic visualizations
        code = f"# Basic visualizations for {df_var}\n\n"
        
        # Histogram for numeric columns
        code += "# Histograms for numeric columns\n"
        code += f"numeric_cols = {df_var}.select_dtypes(include=['number']).columns\n"
        code += "for col in numeric_cols[:5]:  # Limit to first 5 numeric columns\n"
        code += f"    fig = px.histogram({df_var}, x=col, title=f'Distribution of {{{{col}}}}')\n"
        code += "    fig.show()\n\n"
        
        # Bar chart for categorical columns
        code += "# Bar charts for categorical columns\n"
        code += f"cat_cols = {df_var}.select_dtypes(include=['object', 'category']).columns\n"
        code += "for col in cat_cols[:5]:  # Limit to first 5 categorical columns\n"
        code += f"    value_counts = {df_var}[col].value_counts().reset_index()\n"
        code += f"    value_counts.columns = [col, 'count']\n"
        code += f"    fig = px.bar(value_counts, x=col, y='count', title=f'Counts of {{{{col}}}}')\n"
        code += "    fig.show()\n"
        
        cells.append(nbformat.v4.new_code_cell(code))
    
    return cells
