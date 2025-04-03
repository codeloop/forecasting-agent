import pandas as pd
import numpy as np
from prophet import Prophet
from langchain_experimental.tools import PythonREPLTool
from tabulate import tabulate
import json
import io
import ast
from contextlib import redirect_stdout
import subprocess
import sys
import warnings
import traceback

# Silence the plotly import warning
warnings.filterwarnings('ignore', 'Importing plotly failed')

def install_package(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing {package}: {e}")
        return False

class ToolsManager:
    """
    Manages code execution, dependencies, and analysis tools.
    
    Provides:
    - Code execution environment
    - Dependency management
    - Analysis tools
    - Output formatting
    
    Attributes:
        code_interpreter: Python REPL tool for code execution
    """

    def __init__(self):
        try:
            self.code_interpreter = PythonREPLTool()
        except Exception as e:
            print(f"Warning: Code interpreter initialization failed: {e}")
            self.code_interpreter = None

    def extract_imports_from_code(self, code_snippet):
        """Extract import statements and module names from code using AST"""
        try:
            tree = ast.parse(code_snippet)
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        imports.append(name.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module.split('.')[0])
            
            return list(set(imports))  # Remove duplicates
        except:
            # If AST parsing fails, use basic string matching
            import_lines = [line.strip() for line in code_snippet.split('\n') 
                          if line.strip().startswith('import ') or line.strip().startswith('from ')]
            imports = []
            for line in import_lines:
                if line.startswith('import '):
                    imports.extend(name.strip().split('.')[0] 
                                 for name in line[7:].split(','))
                else:  # from ... import ...
                    module = line.split('import')[0].replace('from', '').strip()
                    imports.append(module.split('.')[0])
            return list(set(imports))

    def check_and_install_dependencies(self, code_snippet):
        """
        Check for required packages and install if missing.
        
        Args:
            code_snippet (str): Code to analyze for dependencies
            
        Returns:
            bool: True if all dependencies are satisfied
        """
        # Get required imports from code
        required_imports = self.extract_imports_from_code(code_snippet)
        
        # Filter out standard library modules
        standard_libs = set(sys.stdlib_module_names)
        missing_packages = []
        
        for module in required_imports:
            if module not in standard_libs:
                try:
                    __import__(module)
                except ImportError:
                    missing_packages.append(module)
        
        if missing_packages:
            print(f"\nMissing required packages: {', '.join(missing_packages)}")
            user_input = input("Would you like to install them? (yes/no): ").lower()
            
            if user_input == 'yes':
                for package in missing_packages:
                    print(f"\nInstalling {package}...")
                    if install_package(package):
                        print(f"Successfully installed {package}")
                        try:
                            __import__(package)
                        except ImportError as e:
                            print(f"Warning: Package installed but import failed: {e}")
                            return False
                    else:
                        return False
                return True
            return False
        return True

    def format_stats(self, stats_dict):
        return {k: f"{v:,.2f}" if isinstance(v, (int, float)) else v 
                for k, v in stats_dict.items()}
    
    def generate_descriptive_analysis(self, df, target_column, series_id_column):
        analysis = {}
        
        # Overall dataset analysis
        analysis['overall'] = {
            'total_series': df[series_id_column].nunique(),
            'date_range': [df['date'].min(), df['date'].max()],
            'target_stats': self.format_stats(df[target_column].describe().to_dict())
        }
        
        # Per series analysis
        analysis['per_series'] = {}
        for series in df[series_id_column].unique():
            series_df = df[df[series_id_column] == series]
            analysis['per_series'][series] = {
                'length': len(series_df),
                'target_stats': self.format_stats(series_df[target_column].describe().to_dict())
            }
        
        return analysis

    def format_analysis_output(self, analysis):
        output = []
        output.append("\n=== Overall Analysis ===")
        output.append(f"Total Series: {analysis['overall']['total_series']}")
        output.append(f"Date Range: {analysis['overall']['date_range'][0]} to {analysis['overall']['date_range'][1]}")
        
        output.append("\nOverall Target Statistics:")
        stats_table = [[k, v] for k, v in analysis['overall']['target_stats'].items()]
        output.append(tabulate(stats_table, headers=['Metric', 'Value'], tablefmt='grid'))
        
        output.append("\n=== Per Series Analysis ===")
        for series, data in analysis['per_series'].items():
            output.append(f"\nSeries: {series}")
            output.append(f"Number of records: {data['length']:,}")
            stats_table = [[k, v] for k, v in data['target_stats'].items()]
            output.append(tabulate(stats_table, headers=['Metric', 'Value'], tablefmt='grid'))
        
        return "\n".join(output)

    def generate_forecast(self, df, target_column, series_id_column, periods=10):
        """
        Generate forecasts for multiple time series.
        
        Args:
            df (pd.DataFrame): Input dataset
            target_column (str): Column containing target variable
            series_id_column (str): Column identifying different series
            periods (int): Number of periods to forecast
            
        Returns:
            str: Formatted forecast results
        """
        if df is None:
            return "No data available for forecasting. Please analyze a dataset first."
        
        forecasts = {}
        for series in df[series_id_column].unique():
            series_df = df[df[series_id_column] == series].copy()
            series_df['ds'] = pd.to_datetime(series_df['date'])
            series_df['y'] = series_df[target_column]
            
            try:
                model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
                model.fit(series_df[['ds', 'y']])
                
                future = model.make_future_dataframe(periods=periods, freq='H')
                forecast = model.predict(future)
                
                forecasts[series] = {
                    'forecast': forecast.tail(periods)['yhat'].tolist(),
                    'timestamps': forecast.tail(periods)['ds'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist()
                }
            except Exception as e:
                forecasts[series] = f"Error forecasting: {str(e)}"
        
        return self.format_forecast_output(forecasts)
    
    def format_forecast_output(self, forecasts):
        output = ["\n=== Forecasting Results ==="]
        
        for series, data in forecasts.items():
            output.append(f"\nSeries: {series}")
            if isinstance(data, str):  # Error message
                output.append(data)
            else:
                forecast_table = [[ts, f"{val:,.2f}"] for ts, val in zip(data['timestamps'], data['forecast'])]
                output.append(tabulate(forecast_table, headers=['Timestamp', 'Forecast'], tablefmt='grid'))
        
        return "\n".join(output)
    
    def execute_code(self, code_snippet, df=None):
        """
        Execute Python code with proper error handling and context.
        
        Args:
            code_snippet (str): Code to execute
            df (pd.DataFrame, optional): DataFrame to make available to code
            
        Returns:
            dict: Execution results including:
                - output: stdout capture
                - results: returned variables
                - success: execution status
                - debug_info: execution context
        """
        # First check and handle imports
        if "import plotly" in code_snippet:
            code_snippet = "import warnings\nwarnings.filterwarnings('ignore', 'Importing plotly failed')\n" + code_snippet
        
        # Check and install dependencies first
        if not self.check_and_install_dependencies(code_snippet):
            return {
                'output': "Required packages not installed. Cannot execute code.",
                'results': None,
                'success': False
            }
            
        try:
            # Add debug information
            debug_info = {
                'data_info': None,
                'imports_loaded': [],
                'execution_step': 'initializing'
            }
            
            if df is not None:
                debug_info['data_info'] = {
                    'shape': df.shape,
                    'columns': df.columns.tolist(),
                    'dtypes': df.dtypes.to_dict(),
                    'null_counts': df.isnull().sum().to_dict()
                }
            
            # Add imports to local namespace
            local_vars = {'pd': pd, 'np': np}
            if df is not None:
                local_vars['df'] = df
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    debug_info['execution_step'] = 'datetime conversion done'
            
            # Execute the code
            debug_info['execution_step'] = 'executing code'
            f = io.StringIO()
            with redirect_stdout(f):
                exec(code_snippet, globals(), local_vars)
            output = f.getvalue()
            
            # Get results
            debug_info['execution_step'] = 'collecting results'
            results = {k: v for k, v in local_vars.items() 
                      if k not in ['df', 'pd', 'np'] and not k.startswith('__')}
            
            return {
                'output': output,
                'results': results,
                'success': True,
                'debug_info': debug_info
            }
        except Exception as e:
            error_context = {
                'error_type': type(e).__name__,
                'error_msg': str(e),
                'debug_info': debug_info,
                'traceback': traceback.format_exc()
            }
            
            error_msg = f"Error executing code: {str(e)}\n"
            error_msg += f"\nDebug Information:\n"
            error_msg += f"Error Type: {error_context['error_type']}\n"
            error_msg += f"Last Execution Step: {debug_info['execution_step']}\n"
            if debug_info['data_info']:
                error_msg += f"\nDataFrame Info:\n"
                error_msg += f"Shape: {debug_info['data_info']['shape']}\n"
                error_msg += f"Columns: {debug_info['data_info']['columns']}\n"
                error_msg += f"Data Types:\n"
                for col, dtype in debug_info['data_info']['dtypes'].items():
                    error_msg += f"  {col}: {dtype}\n"
                error_msg += f"Null Counts:\n"
                for col, count in debug_info['data_info']['null_counts'].items():
                    if count > 0:
                        error_msg += f"  {col}: {count}\n"
            
            return {
                'output': error_msg,
                'results': None,
                'success': False,
                'error_context': error_context
            }

    def format_code_output(self, execution_result):
        output = ["\n=== Code Execution Results ==="]
        
        if execution_result['success']:
            if execution_result['output']:
                output.append("\nOutput:")
                output.append(execution_result['output'])
            
            if execution_result['results']:
                output.append("\nReturned Variables:")
                for var_name, value in execution_result['results'].items():
                    output.append(f"\n{var_name}:")
                    if isinstance(value, pd.DataFrame):
                        output.append(tabulate(value.head(), headers='keys', tablefmt='grid'))
                    else:
                        output.append(str(value))
        else:
            output.append("\nExecution Failed:")
            output.append(execution_result['output'])
            
        return "\n".join(output)
