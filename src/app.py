from re import T
import streamlit as st
import pandas as pd
import os
import json
import nbformat
from uuid import uuid4
from state import AppState
from components import TextComponent, ChartComponent, DataInfoComponent
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
from dfinfo import DataFrameInfo
from df_operations import FilterOperation, AggregateOperation, LoadCsvOperation, SaveCsvOperation, DataCleanOperation
from codegen import generate_notebook_cells

# Set page config
st.set_page_config(layout="wide", page_title="Simple & Elegant Data Exploration System (🚽 SEDES)", page_icon="🚽")

# Initialize session state
if "app_state" not in st.session_state:
    st.session_state.app_state = AppState()

if "show_df_info" not in st.session_state:
    st.session_state.show_df_info = False

# Map of chart types
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


# Helper functions
def save_state():
    os.makedirs("data", exist_ok=True)
    st.session_state.app_state.save_state("data/app_state.json")
    st.success("State saved successfully!")


def load_state():
    st.session_state.app_state.load_state("data/app_state.json")
    st.success("State loaded successfully!")
    st.rerun()


def generate_notebook():
    """Generate a Jupyter notebook from the current operations"""
    st.subheader("Generate Jupyter Notebook")

    notebook_path = st.text_input("Notebook Path", value="data_analysis.ipynb")

    if st.button("Generate Notebook"):
        try:
            # Create notebook cells based on operations
            cells = generate_notebook_cells()

            # Create the notebook
            notebook = nbformat.v4.new_notebook()
            notebook.cells = cells

            # Save the notebook
            with open(notebook_path, "w") as f:
                nbformat.write(notebook, f)

            st.success(f"Notebook generated at {notebook_path}")

            # Provide download link
            with open(notebook_path, "r") as f:
                notebook_content = f.read()

            st.download_button(
                label="Download Notebook",
                data=notebook_content,
                file_name=os.path.basename(notebook_path),
                mime="application/x-ipynb+json",
            )
        except Exception as e:
            st.error(f"Error generating notebook: {e}")


# Dialog for adding or editing text component
@st.dialog("Add Text")
def add_text(component_id=None, edit=False):
    dialog_title = "Edit Text Component" if edit else "Add Text Component"
    st.write(dialog_title)

    component = None
    component_text = ""

    if edit and component_id:
        component = next(
            (c for c in st.session_state.app_state.components if getattr(c, "id", id(c)) == component_id), None
        )
        if component:
            component_text = getattr(component, "text", "")

    text = st.text_area("Text Content", value=component_text)

    if st.button("Submit"):
        if edit and component:
            # Update component properties
            component.text = text
            st.session_state.app_state.update_component(component_id, component)
            st.rerun()
        else:
            # Create component
            component = TextComponent(text=text)
            st.session_state.app_state.add_component(component)
            st.rerun()


def edit_text(component_id):
    return add_text(component_id=component_id, edit=True)


# Dialog for adding or editing chart component
@st.dialog("Chart", width="large")
def add_chart(component_id=None, edit=False):
    """
    Dialog for adding or editing a chart component to the application.

    Args:
        component_id (str, optional): ID of the component to edit. Defaults to None.
        edit (bool, optional): Whether this is an edit operation. Defaults to False.
    """
    dialog_title = "Edit Chart Component" if edit else "Add Chart Component"
    st.write(dialog_title)

    component = None
    source_df_id = st.session_state.app_state.current_df_id
    chart_type = "Line Chart"
    existing_params = None

    if edit and component_id:
        # Get the component to edit
        component = get_component_by_id(component_id)
        if not component or not hasattr(component, "chart"):
            st.error("Chart component not found")
            return

        source_df_id = getattr(component, "source_df_id", st.session_state.app_state.current_df_id)

        # Get chart type
        chart_type = getattr(component, "chart_type", "Line Chart")
        if not chart_type:
            chart_type = get_chart_type_from_component(component)

        # Get existing parameters or extract from chart
        existing_params = getattr(component, "chart_params", None)
        if not existing_params:
            existing_params = extract_chart_parameters(component.chart)

    # Get source dataframe
    selected_df_id, df = select_df(source_df_id)
    if df is None:
        return

    # Allow user to change chart type
    if edit:
        chart_type = st.selectbox(
            "Chart Type", list(chart_types.keys()), index=list(chart_types.keys()).index(chart_type)
        )
    else:
        chart_type = st.selectbox("Chart Type", list(chart_types.keys()))

    # Get chart parameters based on type
    chart_params = get_chart_params(chart_type, df, existing_params)

    if st.button("Submit"):
        # Create chart based on type
        chart_fig = create_chart(chart_type, df, chart_params)

        if edit and component:
            # Update component
            component.chart = chart_fig
            setattr(component, "source_df_id", selected_df_id)

            # Update stored chart parameters and type
            setattr(component, "chart_type", chart_type)
            setattr(component, "chart_params", chart_params)

            # Update component in app state
            st.session_state.app_state.update_component(component_id, component)
            st.rerun()
        else:
            # Create component
            component = ChartComponent(chart=chart_fig)

            # Store source dataframe ID and chart metadata for later editing
            setattr(component, "source_df_id", selected_df_id)
            setattr(component, "chart_type", chart_type)
            setattr(component, "chart_params", chart_params)

            # Add component to app state
            st.session_state.app_state.add_component(component)
            st.rerun()


# Alias for edit_chart that calls add_chart with edit=True
def edit_chart(component_id):
    """
    Dialog for editing an existing chart component.

    Args:
        component_id (str): ID of the component to edit
    """
    return add_chart(component_id=component_id, edit=True)


def select_df(source_df_id=None):
    """Helper function to get the source dataframe for a chart"""
    # Get available dataframes
    df_names = st.session_state.app_state.get_dataframe_names()
    if not df_names:
        st.warning("No data available. Please load data first.")
        return None, None

    # Select source dataframe
    df_options = list(df_names.items())

    if source_df_id and source_df_id in [df_id for df_id, _ in df_options]:
        index = [df_id for df_id, _ in df_options].index(source_df_id)
    else:
        index = 0

    selected_df_id = st.selectbox(
        "Select Dataframe",
        options=[df_id for df_id, _ in df_options],
        index=index,
        format_func=lambda x: df_names.get(x, "Unknown"),
    )

    df = st.session_state.app_state.get_dataframe_by_id(selected_df_id)
    if df is None:
        st.warning("Selected dataframe not available.")
        return None, None
    else:
        draw_sample_data(df)

    return selected_df_id, df


def draw_sample_data(df):
    with st.expander("Sample Data"):
        # Get the number of rows to sample (min of 5 or the total number of rows)
        sample_size = min(5, len(df))
        st.dataframe(df.sample(sample_size) if sample_size > 0 else df.head(0))


def get_chart_params(chart_type, df, existing_params=None):
    """
    Get chart parameters based on chart type.

    Args:
        chart_type (str): Type of chart to create
        df (pd.DataFrame): DataFrame to use for the chart
        existing_params (dict, optional): Existing parameters for editing. Defaults to None.

    Returns:
        dict: Parameters for the chart
    """
    columns = df.columns.tolist()
    existing_params = existing_params or {}

    # Common parameters for all charts
    chart_params = {
        "title": st.text_input("Chart Title", value=existing_params.get("title", "")),
    }

    # Add x_column for all chart types except Pie Chart
    if chart_type != "Pie Chart":
        default_x = existing_params.get("x_column", "")
        chart_params["x_column"] = st.selectbox(
            "X Column",
            options=columns,
            index=columns.index(default_x) if default_x in columns else 0,
        )

    # Add y_column for charts that need it
    if chart_type not in ["Histogram", "Pie Chart"]:
        default_y = existing_params.get("y_column", "")
        chart_params["y_column"] = st.selectbox(
            "Y Column",
            options=columns,
            index=columns.index(default_y) if default_y in columns else 0,
        )

    # Add chart-specific parameters
    if chart_type in ["Line Chart", "Scatter Chart"]:
        default_hue = existing_params.get("hue_column", "")
        chart_params["hue_column"] = st.selectbox(
            "Color By",
            options=[""] + columns,
            index=([""] + columns).index(default_hue) if default_hue in [""] + columns else 0,
        )

    if chart_type == "Scatter Chart":
        default_size = existing_params.get("size_column", "")
        chart_params["size_column"] = st.selectbox(
            "Size By",
            options=[""] + columns,
            index=([""] + columns).index(default_size) if default_size in [""] + columns else 0,
        )

    if chart_type in ["Bar Chart", "Pie Chart"]:
        default_group = existing_params.get("group_by", "")
        chart_params["group_by"] = st.selectbox(
            "Group By",
            options=[""] + columns,
            index=([""] + columns).index(default_group) if default_group in [""] + columns else 0,
        )

        # Add values_column for Pie Chart
        if chart_type == "Pie Chart":
            default_values = existing_params.get("values_column", "")
            chart_params["values_column"] = st.selectbox(
                "Values (optional)",
                options=[""] + columns,
                index=([""] + columns).index(default_values) if default_values in [""] + columns else 0,
            )

    if chart_type == "Histogram":
        chart_params["bins"] = st.slider(
            "Number of Bins",
            min_value=5,
            max_value=100,
            value=existing_params.get("bins", 10),
            step=5,
        )
        default_color = existing_params.get("color", "")
        chart_params["color"] = st.selectbox(
            "Color By",
            options=[""] + columns,
            index=([""] + columns).index(default_color) if default_color in [""] + columns else 0,
        )

    if chart_type in ["Box Plot", "Violin Plot"]:
        default_color = existing_params.get("color", "")
        chart_params["color"] = st.selectbox(
            "Color By",
            options=[""] + columns,
            index=([""] + columns).index(default_color) if default_color in [""] + columns else 0,
        )

    # Add extra parameters
    chart_params["extra_params"] = extra_params(existing_params)

    return chart_params


def extra_params(existing_params=None):
    """
    Add extra parameters to charts.

    Args:
        existing_params (dict, optional): Existing parameters for editing. Defaults to None.

    Returns:
        dict: Extra parameters for the chart
    """
    # Initialize session state for extra params if not exists
    if "extra_params_state" not in st.session_state:
        st.session_state.extra_params_state = existing_params.get("extra_params", {}) if existing_params else {}

    # Use the session state to store params
    params = st.session_state.extra_params_state

    with st.expander("Extra Parameters (Advanced, use with caution)"):
        # Display existing parameters
        params_to_delete = []
        params_to_update = {}

        for key, value in params.items():
            col1, col2, col3 = st.columns([2, 4, 1])
            with col1:
                new_key = st.text_input("Parameter", value=key, key=f"param_key_{key}")
            with col2:
                new_value = st.text_input("Value", value=value, key=f"param_value_{key}")
            with col3:
                if st.button("🗑️", key=f"delete_param_{key}"):
                    params_to_delete.append(key)

            # Check if key or value changed
            if new_key != key or new_value != value:
                params_to_update[key] = (new_key, new_value)

        # Handle parameter updates
        for old_key, (new_key, new_value) in params_to_update.items():
            if old_key in params:
                del params[old_key]
            params[new_key] = new_value

        # Handle parameter deletions
        for key in params_to_delete:
            if key in params:
                del params[key]

        # Add new parameter button
        if st.button("➕ Add Parameter"):
            # Generate a unique temporary key
            temp_key = f"new_param_{len(params)}"
            params[temp_key] = ""

    return params


def create_chart(chart_type, df, params):
    """
    Create a chart based on chart type and parameters.

    Args:
        chart_type (str): Type of chart to create
        df (pd.DataFrame): DataFrame to use for the chart
        params (dict): Parameters for the chart

    Returns:
        plotly.graph_objects.Figure: The created chart
    """
    chart_class = chart_types[chart_type]

    # Create base chart parameters
    chart_params = {
        "df": df,
    }

    # Only add parameters that have values
    if params.get("title"):
        chart_params["title"] = params["title"]

    # Add x_column for charts that need it (all except Pie Chart)
    if chart_type != "Pie Chart" and params.get("x_column"):
        chart_params["x_column"] = params["x_column"]

    # Add y_column for charts that need it
    if chart_type not in ["Histogram", "Pie Chart"] and params.get("y_column"):
        chart_params["y_column"] = params["y_column"]

    # Add hue_column only for charts that support it
    if chart_type in ["Line Chart", "Scatter Chart"] and params.get("hue_column"):
        chart_params["hue_column"] = params["hue_column"]

    # Add size_column for scatter charts
    if chart_type == "Scatter Chart" and params.get("size_column"):
        chart_params["size_column"] = params["size_column"]

    # Add group_by for charts that need it
    if chart_type in ["Bar Chart", "Pie Chart"] and params.get("group_by"):
        chart_params["group_by"] = params["group_by"]

    # Add values_column for Pie Chart
    if chart_type == "Pie Chart" and params.get("values_column"):
        chart_params["values_column"] = params["values_column"]

    # Add histogram-specific parameters
    if chart_type == "Histogram":
        if "bins" in params:
            chart_params["bins"] = params["bins"]
        if params.get("color"):
            chart_params["color"] = params["color"]

    # Add color parameter for charts that support it
    if chart_type in ["Box Plot", "Violin Plot", "Area Chart", "Funnel Chart"] and params.get("color"):
        chart_params["color"] = params["color"]

    # Add extra parameters
    if params.get("extra_params"):
        chart_params["params"] = params["extra_params"]

    # Create chart
    chart = chart_class(**chart_params)
    return chart.plot()


def get_component_by_id(component_id):
    """Helper function to get a component by its ID"""
    return next((c for c in st.session_state.app_state.components if getattr(c, "id", id(c)) == component_id), None)


def get_chart_type_from_component(component):
    """Helper function to determine chart type from a component"""
    chart_type = "Line Chart"  # Default
    for name, chart_class in chart_types.items():
        if component.chart.__class__.__name__ == chart_class.__name__:
            chart_type = name
            break
    return chart_type


def extract_chart_parameters(chart):
    """Helper function to extract parameters from an existing chart"""
    params = {}

    # For Plotly figures, we need to extract from the figure data and layout
    try:
        # Extract title from layout
        if hasattr(chart, "layout") and hasattr(chart.layout, "title"):
            if hasattr(chart.layout.title, "text"):
                params["title"] = chart.layout.title.text
            else:
                params["title"] = str(chart.layout.title)

        # Extract x and y columns from the figure data
        if hasattr(chart, "data") and len(chart.data) > 0:
            # Try to get x_column from the first trace
            if hasattr(chart.data[0], "x") and isinstance(chart.data[0].x, pd.Series):
                params["x_column"] = chart.data[0].x.name

            # Try to get y_column from the first trace
            if hasattr(chart.data[0], "y") and isinstance(chart.data[0].y, pd.Series):
                params["y_column"] = chart.data[0].y.name

            # For histograms, the x data is in 'x' not 'y'
            if chart.data[0].type == "histogram" and "y_column" not in params and "x_column" in params:
                params["y_column"] = params["x_column"]

            # Try to extract color/hue information
            if hasattr(chart.data[0], "marker") and hasattr(chart.data[0].marker, "color"):
                if isinstance(chart.data[0].marker.color, pd.Series):
                    params["hue_column"] = chart.data[0].marker.color.name
                elif isinstance(chart.data[0].marker.color, str):
                    params["color"] = chart.data[0].marker.color

            # Try to extract size information for scatter plots
            if chart.data[0].type == "scatter" and hasattr(chart.data[0].marker, "size"):
                if isinstance(chart.data[0].marker.size, pd.Series):
                    params["size_column"] = chart.data[0].marker.size.name

            # For histograms, try to extract bins
            if chart.data[0].type == "histogram" and hasattr(chart.data[0], "nbinsx"):
                params["bins"] = chart.data[0].nbinsx
    except Exception as e:
        st.warning(f"Could not extract all parameters from chart: {e}")

    # Set default values for missing parameters
    params.setdefault("name", "")
    params.setdefault("title", "")
    params.setdefault("x_column", "")
    params.setdefault("y_column", "")
    params.setdefault("hue_column", "")
    params.setdefault("size_column", "")
    params.setdefault("group_by", "")
    params.setdefault("bins", 10)
    params.setdefault("color", "")

    return params


# Dialog for loading CSV
@st.dialog("Load CSV")
def load_csv():
    st.write("Load CSV File")

    name = st.text_input("Operation Name", value=f"Load CSV {uuid4().hex[:4]}")

    # Get file path and separator
    file_path, separator = get_csv_file_parameters()

    if st.button("Submit"):
        if not file_path:
            st.error("Please provide a file path or upload a file")
            return

        # Create operation
        operation = LoadCsvOperation(name=name, id=uuid4().hex, file_path=file_path, sep=separator)
        st.session_state.app_state.add_operation(operation)
        st.rerun()


def get_csv_file_parameters():
    """Helper function to get CSV file parameters"""
    file_path = st.text_input("File Path")

    # File uploader as an alternative
    uploaded_file = st.file_uploader("Or upload a file", type=["csv"])

    # Get separator
    separator = get_csv_separator()

    # Handle file upload
    if uploaded_file is not None:
        # Save the uploaded file
        save_path = os.path.join("data", uploaded_file.name)
        os.makedirs("data", exist_ok=True)

        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_path = save_path

    return file_path, separator


def get_csv_separator():
    """Helper function to get CSV separator"""
    # Separator selection
    separator_options = {"Comma (,)": ",", "Semicolon (;)": ";", "Tab (\t)": "\t", "Pipe (|)": "|", "Space ( )": " "}
    separator_choice = st.selectbox("CSV Separator", options=list(separator_options.keys()), index=0)
    separator = separator_options[separator_choice]

    # Custom separator option
    use_custom_separator = st.checkbox("Use custom separator")
    if use_custom_separator:
        custom_separator = st.text_input("Custom Separator")
        if custom_separator:
            separator = custom_separator

    return separator


# Dialog for adding filter
@st.dialog("Add Filter")
def add_filter(operation_id=None, edit=False):
    dialog_title = "Edit Filter Operation" if edit else "Add Filter"
    st.write(dialog_title)

    # Initialize variables
    operation = None
    source_df_id = None
    operation_name = f"Filter {uuid4().hex[:4]}"
    filter_column = None
    filter_type = None
    filter_value = None

    # If editing, get the operation and its values
    if edit and operation_id:
        operation = next((op for op in st.session_state.app_state.operations if op.id == operation_id), None)
        if not operation or not hasattr(operation, "column") or operation.operation_type != "filter":
            st.error("Filter operation not found")
            return

        source_df_id = operation.source_df_id
        operation_name = operation.name
        filter_column = operation.column
        filter_type = operation.filter_type
        filter_value = operation.filter_value

    # Get source dataframe
    source_df_id, df = get_source_df("Select Source Dataframe" if not edit else "Select Source DataFrame", source_df_id)
    if df is None:
        return

    name = st.text_input("Operation Name", value=operation_name)

    # Get filter parameters
    if edit and filter_column and filter_type:
        # For editing, we need to pre-select the column and filter type
        columns = df.columns.tolist()

        # Set the default column index
        column_index = 0
        if filter_column in columns:
            column_index = columns.index(filter_column)

        column = st.selectbox("Select Column", options=columns, index=column_index)

        # Set the default filter type index
        filter_types = get_filter_types(df[column].dtype)
        filter_type_index = 0
        if filter_type in filter_types:
            filter_type_index = filter_types.index(filter_type)

        filter_type = st.selectbox("Filter Type", options=filter_types, index=filter_type_index)

        # Set the filter value
        if filter_type == "isin":
            # For 'isin', we need to handle a list of values
            if isinstance(filter_value, list):
                filter_value_str = ", ".join(str(val) for val in filter_value)
            else:
                filter_value_str = str(filter_value)
            filter_value_input = st.text_input(
                "Filter Values (comma-separated for multiple values)", value=filter_value_str
            )
        else:
            # For other filter types, it's a single value
            filter_value_input = st.text_input("Filter Value", value=str(filter_value))

        filter_params = {"column": column, "filter_type": filter_type, "filter_value": filter_value_input}
    else:
        # For adding new operations, use the existing helper function
        filter_params = get_filter_parameters(df)

    if st.button("Submit"):
        # Convert filter value based on column data type
        filter_value = convert_filter_value(
            df, filter_params["column"], filter_params["filter_type"], filter_params["filter_value"]
        )

        if edit and operation:
            # Update operation
            operation.name = name
            operation.column = filter_params["column"]
            operation.filter_type = filter_params["filter_type"]
            operation.filter_value = filter_value
            operation.source_df_id = source_df_id

            # Update operation in app state
            st.session_state.app_state.update_operation(operation_id, operation)
        else:
            # Create operation
            operation = FilterOperation(
                name=name,
                id=uuid4().hex,
                column=filter_params["column"],
                filter_type=filter_params["filter_type"],
                filter_value=filter_value,
                source_df_id=source_df_id,
            )
            st.session_state.app_state.add_operation(operation)
        st.rerun()


def edit_filter_operation(operation_id):
    """Dialog for editing a filter operation"""
    return add_filter(operation_id=operation_id, edit=True)


def get_filter_parameters(df):
    """Helper function to get filter parameters"""
    columns = df.columns.tolist()
    column = st.selectbox("Column", options=columns)

    filter_types = get_filter_types(df[column].dtype)
    filter_type = st.selectbox("Filter Type", options=filter_types)

    # Different input based on filter type
    if filter_type == "between":
        col1, col2 = st.columns(2)
        with col1:
            min_value = st.text_input("Min Value")
        with col2:
            max_value = st.text_input("Max Value")
        filter_value = f"{min_value},{max_value}"
    elif filter_type == "isin":
        filter_value = st.text_input("Filter Values (comma-separated for multiple values)")
    else:
        filter_value = st.text_input("Filter Value")

    return {"column": column, "filter_type": filter_type, "filter_value": filter_value}


def get_filter_types(dtype):
    """Helper function to get appropriate filter types based on column data type"""
    # Basic filter types for all data types
    filter_types = ["equals", "not_equals", "isin"]

    # Add numeric-specific filter types for numeric data
    if pd.api.types.is_numeric_dtype(dtype):
        filter_types.extend(["greater_than", "less_than", "between"])

    # Add string-specific filter types for string data
    if pd.api.types.is_string_dtype(dtype):
        filter_types.extend(["contains", "startswith", "endswith"])

    # Add datetime-specific filter types for datetime data
    if pd.api.types.is_datetime64_dtype(dtype):
        filter_types.extend(["before", "after", "between"])

    return filter_types


def convert_filter_value(df, column, filter_type, filter_value):
    """Helper function to convert filter value based on column data type"""
    try:
        # Handle empty values
        if not filter_value:
            if filter_type == "isin":
                return []
            return None

        # Handle list-based filter types
        if filter_type == "isin":
            values = [val.strip() for val in filter_value.split(",")]
            # Convert to appropriate type based on column
            if pd.api.types.is_numeric_dtype(df[column].dtype):
                return [float(val) if "." in val else int(val) for val in values if val]
            elif pd.api.types.is_datetime64_dtype(df[column].dtype):
                return [pd.to_datetime(val) for val in values if val]
            else:
                return values

        # Handle between filter type
        if filter_type == "between":
            if "," in filter_value:
                min_val, max_val = filter_value.split(",", 1)
                min_val = min_val.strip()
                max_val = max_val.strip()

                # Convert to appropriate type
                if pd.api.types.is_numeric_dtype(df[column].dtype):
                    return [
                        float(min_val) if "." in min_val else int(min_val),
                        float(max_val) if "." in max_val else int(max_val),
                    ]
                elif pd.api.types.is_datetime64_dtype(df[column].dtype):
                    return [pd.to_datetime(min_val), pd.to_datetime(max_val)]
                else:
                    return [min_val, max_val]
            else:
                # If no comma, return as single value list
                return [filter_value, filter_value]

        # Handle single value filter types
        if pd.api.types.is_numeric_dtype(df[column].dtype):
            # Convert to float or int based on value
            if "." in filter_value:
                return float(filter_value)
            else:
                return int(filter_value)
        elif pd.api.types.is_datetime64_dtype(df[column].dtype):
            return pd.to_datetime(filter_value)
        else:
            # For string columns, return as is
            return filter_value
    except (ValueError, TypeError):
        # If conversion fails, return as string
        return filter_value


# Dialog for adding aggregation
@st.dialog("Add Aggregation")
def add_aggregation(operation_id=None, edit=False):
    dialog_title = "Edit Aggregation Operation" if edit else "Add Aggregation"
    st.write(dialog_title)

    # Initialize variables
    operation = None
    source_df_id = None
    operation_name = f"Aggregate {uuid4().hex[:4]}"
    group_by = []
    agg_func = {}

    # If editing, get the operation and its values
    if edit and operation_id:
        operation = next((op for op in st.session_state.app_state.operations if op.id == operation_id), None)
        if not operation or not hasattr(operation, "group_by") or operation.operation_type != "aggregate":
            st.error("Aggregation operation not found")
            return

        source_df_id = operation.source_df_id
        operation_name = operation.name
        group_by = operation.group_by
        agg_func = operation.agg_func

    # Get source dataframe
    source_df_id, df = get_source_df("Select Source Dataframe" if not edit else "Select Source DataFrame", source_df_id)
    if df is None:
        return

    name = st.text_input("Operation Name", value=operation_name)

    # Get aggregation parameters
    if edit and operation:
        # For editing, we need to pre-select the group by columns and aggregation functions
        columns = df.columns.tolist()

        # Group by columns with defaults from the operation
        group_by = st.multiselect("Group By Columns", options=columns, default=group_by)

        # Aggregation functions for each numeric column
        st.subheader("Aggregation Functions")
        agg_func_new = {}

        numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
        numeric_columns = [col for col in numeric_columns if col not in group_by]
        if not numeric_columns:
            st.warning("No numeric columns available for aggregation")
        else:
            for col in numeric_columns:
                agg_options = ["none", "mean", "sum", "min", "max", "count", "median"]
                # Set default index based on existing operation
                default_index = 0
                if col in agg_func:
                    try:
                        default_index = agg_options.index(agg_func[col])
                    except ValueError:
                        default_index = 0

                selected_agg = st.selectbox(
                    f"Aggregation for {col}", options=agg_options, index=default_index, key=f"edit_agg_{col}"
                )

                # Only add non-'none' aggregation functions to the dictionary
                if selected_agg != "none":
                    agg_func_new[col] = selected_agg

        agg_params = {"group_by": group_by, "agg_func": agg_func_new}
    else:
        # For adding new operations, use the existing helper function
        agg_params = get_aggregation_parameters(df)

    if st.button("Submit"):
        if not agg_params["group_by"]:
            st.error("Please select at least one column to group by")
            return

        if not agg_params["agg_func"]:
            st.error("Please select at least one aggregation function")
            return

        if edit and operation:
            # Update operation
            operation.name = name
            operation.group_by = agg_params["group_by"]
            operation.agg_func = agg_params["agg_func"]
            operation.source_df_id = source_df_id

            # Update operation in app state
            st.session_state.app_state.update_operation(operation_id, operation)
        else:
            # Create operation
            operation = AggregateOperation(
                name=name,
                id=uuid4().hex,
                group_by=agg_params["group_by"],
                agg_func=agg_params["agg_func"],
                source_df_id=source_df_id,
            )
            st.session_state.app_state.add_operation(operation)
        st.rerun()


def edit_aggregation_operation(operation_id):
    """Dialog for editing an aggregation operation"""
    return add_aggregation(operation_id=operation_id, edit=True)


def get_aggregation_parameters(df):
    """
    Get aggregation parameters for dataframe operations.

    Args:
        df (pd.DataFrame): DataFrame to aggregate

    Returns:
        dict: Aggregation parameters including group_by columns and aggregation functions
    """
    columns = df.columns.tolist()
    group_by = st.multiselect("Group By Columns", options=columns)

    # Aggregation functions for each numeric column
    st.subheader("Aggregation Functions")
    agg_func = {}

    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    numeric_columns = [col for col in numeric_columns if col not in group_by]
    if not numeric_columns:
        st.warning("No numeric columns available for aggregation")
    else:
        for col in numeric_columns:
            agg_options = ["none", "mean", "sum", "min", "max", "count", "median"]
            selected_agg = st.selectbox(f"Aggregation for {col}", options=agg_options, index=0, key=f"agg_{col}")

            # Only add non-'none' aggregation functions to the dictionary
            if selected_agg != "none":
                agg_func[col] = selected_agg

    # Filter out any 'none' values that might have slipped through
    return {"group_by": group_by, "agg_func": {k: v for k, v in agg_func.items() if v != "none"}}


# Dialog for adding or editing data cleaning
@st.dialog("Add Data Cleaning")
def add_data_cleaning(operation_id=None, edit=False):
    dialog_title = "Edit Data Cleaning Operation" if edit else "Add Data Cleaning"
    st.write(dialog_title)

    # Initialize variables
    operation = None
    source_df_id = None
    operation_name = f"Data Cleaning {uuid4().hex[:4]}"
    clean_type = None
    selected_columns = []
    params = {"fill_value": None, "replace_values": {}, "new_column_names": {}}

    # If editing, get the operation and its values
    if edit and operation_id:
        operation = next((op for op in st.session_state.app_state.operations if op.id == operation_id), None)
        if not operation or not hasattr(operation, "clean_type"):
            st.error("Data cleaning operation not found")
            return

        source_df_id = operation.source_df_id
        operation_name = operation.name
        clean_type = operation.clean_type
        selected_columns = operation.columns if hasattr(operation, "columns") else []

        # Set parameters based on cleaning type
        if operation.clean_type == "fillna" and hasattr(operation, "fill_value"):
            params["fill_value"] = operation.fill_value
        elif operation.clean_type == "replace" and hasattr(operation, "replace_values"):
            params["replace_values"] = operation.replace_values
        elif operation.clean_type == "rename" and hasattr(operation, "new_column_names"):
            params["new_column_names"] = operation.new_column_names

    # Get source dataframe
    source_df_id, df = get_source_df("Select Source Dataframe" if not edit else "Select Source DataFrame", source_df_id)
    if df is None:
        return

    # Set operation name
    name = st.text_input("Operation Name", value=operation_name)

    # Select cleaning type
    if edit and clean_type:
        clean_types = ["dropna", "fillna", "drop_duplicates", "replace", "rename"]
        clean_type_index = clean_types.index(clean_type) if clean_type in clean_types else 0
        clean_type = st.selectbox(
            "Cleaning Type" if not edit else "Cleaning Method",
            options=clean_types,
            index=clean_type_index,
            format_func=lambda x: {
                "dropna": "Drop Missing Values",
                "fillna": "Fill Missing Values",
                "drop_duplicates": "Remove Duplicates",
                "replace": "Replace Values",
                "rename": "Rename Columns",
            }.get(x, x),
        )
    else:
        clean_type = select_clean_type()

    # Get columns from dataframe
    columns = df.columns.tolist()

    # Get columns to apply cleaning to
    if edit and selected_columns:
        # For editing, we need to handle the selected columns differently
        if clean_type in ["dropna", "fillna", "drop_duplicates", "replace"]:
            selected_columns = st.multiselect(
                "Select Columns" if clean_type != "rename" else "Select Columns to Rename", options=columns
            )
        else:
            selected_columns = []
    else:
        selected_columns = select_clean_cols(clean_type, columns)

    # Get additional parameters based on type
    if edit:
        if clean_type == "fillna":
            fill_options = ["", "mean", "median", "mode", "ffill", "bfill"]
            fill_value = params["fill_value"]
            fill_index = fill_options.index(fill_value) if fill_value in fill_options else 0
            params["fill_value"] = st.selectbox("Fill Value", options=fill_options, index=fill_index)
            if params["fill_value"] == "":
                params["fill_value"] = st.text_input(
                    "Custom Fill Value", value=str(fill_value) if fill_value not in fill_options else ""
                )
        elif clean_type == "replace":
            st.write("Replace Values")
            replace_values = {}

            # Display existing replace values
            for i, (old_val, new_val) in enumerate(params["replace_values"].items()):
                col1, col2, col3 = st.columns([3, 3, 1])
                with col1:
                    old_value = st.text_input(f"Old Value {i+1}", value=str(old_val))
                with col2:
                    new_value = st.text_input(f"New Value {i+1}", value=str(new_val))
                with col3:
                    if st.button("", key=f"del_replace_{i}"):
                        continue  # Skip this pair in the loop

                replace_values[convert_val_to_type(old_value)] = convert_val_to_type(new_value)

            # Add new replace value
            if st.button("Add Replace Value"):
                col1, col2 = st.columns(2)
                with col1:
                    old_value = st.text_input("Old Value", key="new_old_val")
                with col2:
                    new_value = st.text_input("New Value", key="new_new_val")

                if old_value:
                    replace_values[convert_val_to_type(old_value)] = convert_val_to_type(new_value)

            params["replace_values"] = replace_values
        elif clean_type == "rename":
            st.write("Rename Columns")
            new_column_names = {}

            # Display existing column renames
            for i, (old_name, new_name) in enumerate(params["new_column_names"].items()):
                if old_name in df.columns:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.text(f"Original: {old_name}")
                    with col2:
                        new_column_name = st.text_input(f"New Name for {old_name}", value=new_name, key=f"rename_{i}")
                        new_column_names[old_name] = new_column_name

            # Add options for columns not yet renamed
            for col in df.columns:
                if col not in new_column_names:
                    rename_col = st.checkbox(f"Rename {col}", key=f"check_{col}")
                    if rename_col:
                        new_name = st.text_input(f"New Name for {col}", key=f"new_name_{col}")
                        if new_name:
                            new_column_names[col] = new_name

            params["new_column_names"] = new_column_names
    else:
        # For adding new operations, use the existing helper functions
        cleaning_params = get_cleaning_params(clean_type, df, columns)
        params["fill_value"] = cleaning_params.get("fill_value")
        params["replace_values"] = cleaning_params.get("replace_values", {})
        params["new_column_names"] = cleaning_params.get("new_column_names", {})

    if st.button("Submit"):
        if edit and operation:
            # Update operation
            operation.name = name
            operation.clean_type = clean_type
            operation.columns = selected_columns
            operation.source_df_id = source_df_id

            # Update specific parameters based on cleaning type
            if clean_type == "fillna":
                operation.fill_value = convert_val_to_type(params["fill_value"])
            elif clean_type == "replace":
                operation.replace_values = params["replace_values"]
            elif clean_type == "rename":
                operation.new_column_names = params["new_column_names"]

            # Update operation in app state
            st.session_state.app_state.update_operation(operation_id, operation)
            st.rerun()
        else:
            # Create operation
            operation = DataCleanOperation(
                name=name,
                id=uuid4().hex,
                clean_type=clean_type,
                columns=selected_columns,
                fill_value=params["fill_value"],
                replace_values=params["replace_values"],
                new_column_names=params["new_column_names"],
                source_df_id=source_df_id,
            )
            st.session_state.app_state.add_operation(operation)
            st.rerun()


# Alias for edit_data_cleaning_operation that calls add_data_cleaning with edit=True
def edit_data_cleaning_operation(operation_id):
    """Dialog for editing a data cleaning operation"""
    return add_data_cleaning(operation_id=operation_id, edit=True)


def get_source_df(label="Select Dataframe", source_df_id=None):
    """Helper function to get a source dataframe for operations"""
    # Get available dataframes
    df_names = st.session_state.app_state.get_dataframe_names()
    if not df_names:
        st.warning("No data available. Please load data first.")
        return None, None

    # Select source dataframe
    df_options = list(df_names.items())

    # Set the default index based on the source_df_id if provided
    index = 0
    if source_df_id and source_df_id in [df_id for df_id, _ in df_options]:
        index = [df_id for df_id, _ in df_options].index(source_df_id)

    selected_df_id = st.selectbox(
        label,
        options=[df_id for df_id, _ in df_options],
        index=index,
        format_func=lambda x: df_names.get(x, "Unknown"),
    )

    df = st.session_state.app_state.get_dataframe_by_id(selected_df_id)
    if df is None:
        st.warning("Selected dataframe not available.")
        return None, None

    return selected_df_id, df


def select_clean_type():
    """Helper function to select the cleaning type"""
    clean_types = ["dropna", "fillna", "drop_duplicates", "replace", "rename"]
    return st.selectbox(
        "Cleaning Method",
        options=clean_types,
        format_func=lambda x: {
            "dropna": "Drop Missing Values",
            "fillna": "Fill Missing Values",
            "drop_duplicates": "Remove Duplicates",
            "replace": "Replace Values",
            "rename": "Rename Columns",
        }.get(x, x),
    )


def select_clean_cols(clean_type, columns):
    """Helper function to select columns for cleaning operation"""
    if clean_type in ["dropna", "fillna", "drop_duplicates", "replace"]:
        return st.multiselect(
            "Select Columns" if clean_type != "rename" else "Select Columns to Rename", options=columns
        )
    else:
        return []


def get_cleaning_params(clean_type, df, columns):
    """Helper function to get additional parameters based on cleaning type"""
    params = {"fill_value": None, "replace_values": {}, "new_column_names": {}}

    if clean_type == "fillna":
        params["fill_value"] = get_fillna_params()
    elif clean_type == "replace":
        params["replace_values"] = get_replace_params()
    elif clean_type == "rename":
        params["new_column_names"] = get_rename_params(columns)

    return params


def get_fillna_params():
    """Helper function to get parameters for fillna operation"""
    fill_method = st.selectbox("Fill Method", options=["value", "mean", "median", "mode", "ffill", "bfill"])

    if fill_method == "value":
        fill_value = st.text_input("Fill Value")
        # Try to convert to appropriate type
        try:
            if fill_value.lower() in ["true", "false"]:
                fill_value = fill_value.lower() == "true"
            elif "." in fill_value and fill_value.replace(".", "", 1).isdigit():
                fill_value = float(fill_value)
            elif fill_value.isdigit():
                fill_value = int(fill_value)
        except ValueError:
            # Keep as string if conversion fails
            pass
    else:
        # For other methods, we'll handle them in the apply method
        fill_value = fill_method

    return fill_value


def get_replace_params():
    """Helper function to get parameters for replace operation"""
    replace_values = {}

    st.write("Enter values to replace (one pair per line):")
    st.write("Format: old_value,new_value")
    replace_text = st.text_area("Replace Values")

    if replace_text:
        for line in replace_text.strip().split("\n"):
            if "," in line:
                old_val, new_val = line.split(",", 1)
                old_val = old_val.strip()
                new_val = new_val.strip()

                # Try to convert to appropriate types
                old_val = convert_val_to_type(old_val)
                new_val = convert_val_to_type(new_val)

                replace_values[old_val] = new_val

    return replace_values


def get_rename_params(columns):
    """Helper function to get parameters for rename operation"""
    new_column_names = {}

    st.write("Select columns to rename:")

    for col in columns:
        new_name = st.text_input(f"New name for '{col}'", value=col)
        if new_name != col:
            new_column_names[col] = new_name

    return new_column_names


def convert_val_to_type(val):
    """Helper function to convert a string value to the appropriate type"""
    try:
        if val.lower() in ["true", "false"]:
            return val.lower() == "true"
        elif "." in val and val.replace(".", "", 1).isdigit():
            return float(val)
        elif val.isdigit():
            return int(val)
        return val
    except (ValueError, AttributeError):
        # Keep as is if conversion fails or if it's not a string
        return val


# Dialog for saving CSV
@st.dialog("Save CSV")
def save_csv():
    st.write("Save CSV File")

    # Get available dataframes
    df_names = st.session_state.app_state.get_dataframe_names()
    if not df_names:
        st.warning("No data available. Please load data first.")
        return

    # Select source dataframe
    df_options = list(df_names.items())
    selected_df_id = st.selectbox(
        "Select Dataframe to Save",
        options=[df_id for df_id, _ in df_options],
        format_func=lambda x: df_names.get(x, "Unknown"),
    )

    df = st.session_state.app_state.get_dataframe_by_id(selected_df_id)
    if df is None:
        st.warning("Selected dataframe not available.")
        return

    name = st.text_input("Operation Name", value=f"Save CSV {uuid4().hex[:4]}")

    # Get file path and separator
    file_path = st.text_input("File Path", value=os.path.join("data", f"export_{uuid4().hex[:8]}.csv"))
    separator = get_csv_separator()

    if st.button("Submit"):
        if not file_path:
            st.error("Please provide a file path")
            return

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Create operation
        operation = SaveCsvOperation(
            name=name, id=uuid4().hex, file_path=file_path, sep=separator, source_df_id=selected_df_id
        )

        # Apply the operation directly
        operation.apply(df)
        st.session_state.app_state.add_operation(operation)
        st.success(f"File saved to {file_path}")
        st.rerun()


# Dialog for showing dataframe info
@st.dialog("DataFrame Info")
def show_dataframe_info():
    # Get source dataframe
    selected_df_id, df = get_source_df("Select Dataframe")
    if df is None:
        return

    # Get dataframe info
    df_info = DataFrameInfo(df)
    info_dict = df_info.get_info()

    # Display info components
    display_basic_info(info_dict)
    display_column_info(info_dict)
    display_numeric_stats(info_dict)
    display_sample_data(df)


def display_basic_info(info_dict):
    """Helper function to display basic dataframe information"""
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Rows", info_dict["shape"]["rows"])
    with col2:
        st.metric("Columns", info_dict["shape"]["columns"])


def display_column_info(info_dict):
    """Helper function to display column information"""
    st.subheader("Columns")
    col_data = []
    for col in info_dict["columns"]:
        dtype = info_dict["dtypes"][col]
        missing = info_dict["missing_values"][col]
        col_data.append(
            {
                "Column": col,
                "Type": dtype,
                "Missing": f"{missing['count']} ({missing['percentage']:.2f}%)",
            }
        )

    st.dataframe(pd.DataFrame(col_data))


def display_numeric_stats(info_dict):
    """Helper function to display numeric statistics"""
    if info_dict["numeric_stats"]:
        st.subheader("Numeric Statistics")

        # Create a dataframe for each statistic
        stats_df = {}
        for col, stats in info_dict["numeric_stats"].items():
            for stat_name, stat_value in stats.items():
                if stat_name not in stats_df:
                    stats_df[stat_name] = {}
                stats_df[stat_name][col] = stat_value

        tabs = st.tabs(list(stats_df.keys()))
        for i, (stat_name, stat_data) in enumerate(stats_df.items()):
            with tabs[i]:
                st.dataframe(pd.Series(stat_data))


def display_sample_data(df):
    """Helper function to display sample data"""
    st.subheader("Sample Data")
    st.dataframe(df.head())


def display_eda_tab():
    """Display EDA components in the first tab"""
    # Display components
    st.header("EDA Components")
    if st.session_state.app_state.components:
        for i, component in enumerate(st.session_state.app_state.components):
            component_id = getattr(component, "id", id(component))
            component_name = getattr(component, "name", f"Component {i+1}")

            with st.expander(f"{component_name}", expanded=True):
                # Display source dataframe if available
                source_df_id = getattr(component, "source_df_id", None)
                if source_df_id:
                    source_name = st.session_state.app_state.dataframe_names.get(source_df_id, "Unknown")
                    st.write(f"Source DataFrame: {source_name}")

                # Display component based on type
                if hasattr(component, "text"):
                    st.markdown(component.text)
                elif hasattr(component, "chart"):
                    st.plotly_chart(component.chart, use_container_width=True)
                elif hasattr(component, "info_type"):
                    display_data_info(component.info_type, component.source_df_id)

                # Component actions
                col1, col2 = st.columns(2)
                with col1:
                    if hasattr(component, "text"):
                        if st.button("Edit", key=f"edit_comp_{i}"):
                            edit_text(component_id)
                    elif hasattr(component, "chart"):
                        if st.button("Edit", key=f"edit_comp_{i}"):
                            edit_chart(component_id)
                    elif hasattr(component, "info_type"):
                        if st.button("Edit", key=f"edit_comp_{i}"):
                            edit_data_info_component(component_id)

                with col2:
                    if st.button("Delete", key=f"delete_comp_{i}"):
                        st.session_state.app_state.delete_component(component_id)
                        st.rerun()
    else:
        st.info("No components added yet. Use the sidebar to add components.")


def display_data_operations_tab():
    """Display data operations in the second tab"""
    # Display operations
    st.header("Data Operations")
    if st.session_state.app_state.operations:
        for i, operation in enumerate(st.session_state.app_state.operations):
            with st.expander(f"{operation.name}", expanded=True):
                st.write(f"Type: {operation.__class__.__name__}")

                # Display operation-specific details
                if hasattr(operation, "file_path"):
                    st.write(f"File: {operation.file_path}")
                    st.write(f"Separator: {operation.sep}")
                elif hasattr(operation, "column"):
                    st.write(f"Column: {operation.column}")
                    st.write(f"Filter Type: {operation.filter_type}")
                    st.write(f"Filter Value: {operation.filter_value}")
                    if operation.source_df_id:
                        source_name = st.session_state.app_state.dataframe_names.get(operation.source_df_id, "Unknown")
                        st.write(f"Source DataFrame: {source_name}")
                elif hasattr(operation, "group_by"):
                    st.write(f"Group By: {', '.join(operation.group_by)}")
                    st.write(f"Aggregations: {operation.agg_func}")
                    if operation.source_df_id:
                        source_name = st.session_state.app_state.dataframe_names.get(operation.source_df_id, "Unknown")
                        st.write(f"Source DataFrame: {source_name}")
                elif hasattr(operation, "clean_type"):
                    st.write(f"Clean Type: {operation.clean_type}")
                    if operation.columns:
                        st.write(f"Columns: {', '.join(operation.columns)}")
                    if operation.source_df_id:
                        source_name = st.session_state.app_state.dataframe_names.get(operation.source_df_id, "Unknown")
                        st.write(f"Source DataFrame: {source_name}")
                    if operation.clean_type == "fillna":
                        st.write(f"Fill Value: {operation.fill_value}")
                    elif operation.clean_type == "replace":
                        st.write(f"Replace Values: {operation.replace_values}")
                    elif operation.clean_type == "rename":
                        st.write(f"New Column Names: {operation.new_column_names}")

                # Operation actions
                col1, col2 = st.columns(2)
                with col1:
                    # Add edit button based on operation type
                    if hasattr(operation, "column") and operation.operation_type == "filter":
                        if st.button("Edit", key=f"edit_op_{i}"):
                            edit_filter_operation(operation.id)
                    elif hasattr(operation, "group_by") and operation.operation_type == "aggregate":
                        if st.button("Edit", key=f"edit_op_{i}"):
                            edit_aggregation_operation(operation.id)
                    elif hasattr(operation, "clean_type") and operation.operation_type == "data_clean":
                        if st.button("Edit", key=f"edit_op_{i}"):
                            edit_data_cleaning_operation(operation.id)

                with col2:
                    if st.button("Delete", key=f"delete_op_{i}"):
                        st.session_state.app_state.delete_operation(operation.id)
                        st.rerun()
    else:
        st.info("No operations added yet. Use the sidebar to add operations.")


def display_data_tab():
    """Display dataframe information and preview in the data tab"""
    # Display available dataframes
    st.header("Available DataFrames")
    df_names = st.session_state.app_state.get_dataframe_names()
    if df_names:
        df_table = []
        for df_id, df_name in df_names.items():
            df = st.session_state.app_state.get_dataframe_by_id(df_id)
            if df is not None:
                df_table.append(
                    {
                        "Name": df_name,
                        "ID": df_id,
                        "Rows": df.shape[0],
                        "Columns": df.shape[1],
                        "Current": "✓" if df_id == st.session_state.app_state.current_df_id else "",
                    }
                )

        st.dataframe(pd.DataFrame(df_table))
    else:
        st.info("No dataframes available yet. Load a CSV file to get started.")

    # DataFrame preview with selection
    st.header("DataFrame Preview")
    if df_names:
        # Create a list of dataframe names for the selectbox
        df_options = list(df_names.values())
        df_ids = list(df_names.keys())

        # Add a selectbox to choose which dataframe to preview
        selected_df_name = st.selectbox(
            "Select DataFrame to Preview",
            options=df_options,
            index=df_ids.index(st.session_state.app_state.current_df_id)
            if st.session_state.app_state.current_df_id in df_ids
            else 0,
        )

        # Get the ID of the selected dataframe
        selected_df_id = df_ids[df_options.index(selected_df_name)]

        # Display the selected dataframe
        df = st.session_state.app_state.get_dataframe_by_id(selected_df_id)
        if df is not None:
            st.dataframe(df)

            # Add basic statistics for the selected dataframe
            with st.expander("DataFrame Statistics"):
                st.write("### Basic Statistics")
                st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
                st.write("### Numeric Columns Statistics")
                st.dataframe(df.describe())

                st.write("### Column Types")
                dtypes_df = pd.DataFrame({"Data Type": [str(dtype) for dtype in df.dtypes]}, index=df.columns)
                dtypes_df.index.name = "Column"
                st.dataframe(dtypes_df.reset_index())

                st.write("### Missing Values")
                missing_df = pd.DataFrame(
                    {"Missing Values": df.isna().sum(), "Percentage": (df.isna().sum() / len(df) * 100).round(2)}
                )
                missing_df.index.name = "Column"
                st.dataframe(missing_df.reset_index())
        else:
            st.warning("Selected dataframe is not available.")
    else:
        st.info("No dataframes available yet. Load a CSV file to get started.")


# Dialog for adding data info component
@st.dialog("Data Info")
def add_data_info(component_id=None, edit=False):
    """
    Dialog for adding or editing a data info component to the application.

    Args:
        component_id (str, optional): ID of the component to edit. Defaults to None.
        edit (bool, optional): Whether this is an edit operation. Defaults to False.
    """
    dialog_title = "Edit Data Info Component" if edit else "Add Data Info Component"
    st.write(dialog_title)

    component = None
    source_df_id = st.session_state.app_state.current_df_id
    info_type = "preview"

    if edit and component_id:
        # Get the component to edit
        component = get_component_by_id(component_id)
        if not component or not isinstance(component, DataInfoComponent):
            st.error("Data Info component not found")
            return

        source_df_id = getattr(component, "source_df_id", st.session_state.app_state.current_df_id)
        info_type = getattr(component, "info_type", "preview")

    # Get source dataframe
    selected_df_id, df = select_df(source_df_id)
    if df is None:
        return

    # Select the type of information to display
    info_types = {
        "preview": "DataFrame Preview",
        "shape": "Shape (rows & columns)",
        "stats": "Statistics (describe)",
        "types": "Column Types",
        "missing": "Missing Values",
        "all": "All Information",
    }

    info_type = st.selectbox(
        "Information Type",
        options=list(info_types.keys()),
        format_func=lambda x: info_types[x],
        index=list(info_types.keys()).index(info_type) if info_type in info_types else 0,
    )

    if st.button("Submit"):
        if edit and component:
            # Update component
            component.source_df_id = selected_df_id
            component.info_type = info_type

            # Update component in app state
            st.session_state.app_state.update_component(component_id, component)
            st.rerun()
        else:
            # Create component
            component = DataInfoComponent(
                source_df_id=selected_df_id, info_type=info_type, name=f"Data Info: {info_types[info_type]}"
            )

            # Add component to app state
            st.session_state.app_state.add_component(component)
            st.rerun()


def edit_data_info_component(component_id):
    """Dialog for editing a data info component"""
    return add_data_info(component_id=component_id, edit=True)


def display_data_info(info_type, source_df_id):
    """Helper function to display data info"""
    df = st.session_state.app_state.get_dataframe_by_id(source_df_id)
    if df is None:
        st.warning("Selected dataframe is not available.")
        return

    if info_type == "preview":
        st.dataframe(df.sample(10) if df.shape[0] > 10 else df.head())
    elif info_type == "shape":
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Rows", df.shape[0])
        with c2:
            st.metric("Columns", df.shape[1])
    elif info_type == "stats":
        st.dataframe(df.describe())
    elif info_type == "types":
        dtypes_df = pd.DataFrame({"Data Type": [str(dtype) for dtype in df.dtypes]}, index=df.columns)
        dtypes_df.index.name = "Column"
        st.dataframe(dtypes_df.reset_index())
    elif info_type == "missing":
        missing_df = pd.DataFrame(
            {"Missing Values": df.isna().sum(), "Percentage": (df.isna().sum() / len(df) * 100).round(2)}
        )
        missing_df.index.name = "Column"
        st.dataframe(missing_df.reset_index())
    elif info_type == "all":
        display_dataframe_info(df)


def display_dataframe_info(df):
    """Helper function to display dataframe info"""
    # Display basic info
    st.subheader("Basic Information")
    st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # Display column types
    st.subheader("Column Types")
    dtypes_df = pd.DataFrame({"Data Type": [str(dtype) for dtype in df.dtypes]}, index=df.columns)
    dtypes_df.index.name = "Column"
    st.dataframe(dtypes_df.reset_index())

    # Display missing values
    st.subheader("Missing Values")
    missing_df = pd.DataFrame(
        {"Missing Values": df.isna().sum(), "Percentage": (df.isna().sum() / len(df) * 100).round(2)}
    )
    missing_df.index.name = "Column"
    st.dataframe(missing_df.reset_index())

    # Display numeric statistics
    st.subheader("Numeric Statistics")
    st.dataframe(df.describe())

    # Display sample data
    st.subheader("Sample Data")
    st.dataframe(df.sample(10) if df.shape[0] > 10 else df.head())


def display_state_management_tab():
    """Display state json representation"""
    pass


def main():
    # Main app layout
    st.title("Simple & Elegant Data Exploration System (🚽 SEDES)")

    # Load sample data if no operations exist
    if not st.session_state.app_state.operations:
        # Load sample data for demonstration
        try:
            sample_df = pd.read_csv("src/iris.csv")
            if "sample_data_loaded" not in st.session_state:
                # Create a load operation for the sample data
                operation = LoadCsvOperation(
                    name="Sample Iris Dataset", id=uuid4().hex, file_path="src/iris.csv", sep=","
                )
                st.session_state.app_state.add_operation(operation)
                st.session_state.sample_data_loaded = True
        except Exception as e:
            st.warning(f"Could not load sample data: {e}")

    # Sidebar for operations
    with st.sidebar:
        st.header("Operations")

        # Data operations section
        st.subheader("Data Operations")

        c1, c2, c3 = st.columns(3)

        with c1:
            if st.button("📂", help="Load CSV", key="load_csv_btn"):
                load_csv()
        with c2:
            if st.button("📊", help="DataFrame Info", key="df_info_btn"):
                show_dataframe_info()
        with c3:
            if st.button("💾", help="Save CSV", key="save_csv_btn"):
                save_csv()

        # Data transformations section
        st.subheader("Data Transformations")

        c1, c2, c3 = st.columns(3)

        with c1:
            if st.button("🔍", help="Add Filter", key="add_filter_btn"):
                add_filter()
        with c2:
            if st.button("🔄", help="Add Aggregation", key="add_agg_btn"):
                add_aggregation()
        with c3:
            if st.button("🧹", help="Add Data Cleaning", key="add_clean_btn"):
                add_data_cleaning()

        # Component operations section
        st.subheader("EDA Components")

        c1, c2, c3 = st.columns(3)

        with c1:
            if st.button("📝", help="Add Text", key="add_text_btn"):
                add_text()
        with c2:
            if st.button("📈", help="Add Chart", key="add_chart_btn"):
                add_chart()
        with c3:
            if st.button("📊", help="Add Data Info", key="add_data_info_btn"):
                add_data_info()

        # State management section
        st.subheader("State Management")

        c1, c2, c3 = st.columns(3)

        with c1:
            if st.button("💾", help="Save State", key="save_state_btn", disabled=True):
                save_state()
        with c2:
            if st.button("📂", help="Load State", key="load_state_btn", disabled=True):
                load_state()
        with c3:
            if st.button("📓", help="Generate Notebook", key="gen_notebook_btn", disabled=True):
                generate_notebook()

    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["EDA", "Data Operations", "Data Preview", "State Management"])

    with tab1:
        display_eda_tab()

    with tab2:
        display_data_operations_tab()

    with tab3:
        display_data_tab()

    with tab4:
        display_state_management_tab()


if __name__ == "__main__":
    main()
