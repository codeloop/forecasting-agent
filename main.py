"""
Forecasting Agent CLI Interface

This module provides the main entry point and command-line interface for the
forecasting agent. It handles user interaction, command processing, and
agent initialization.
"""

import pandas as pd
from src.agent import ForecastingAgent
from colorama import init, Fore, Style

# Initialize colorama
init()

def print_help():
    """Display available commands and their usage."""
    print("\nAvailable commands:")
    print("analyze <csv_path> <target_column> <series_id_column> - Analyze a new dataset")
    print("help - Show this help message")
    print("/bye - Exit the program")
    print("You can also ask general questions about the data or request forecasts!")

def main():
    """
    Main program loop.
    
    Initializes the forecasting agent and processes user commands until exit.
    Handles:
    - Command parsing
    - Agent initialization
    - User interaction
    - Error handling
    """
    agent = ForecastingAgent()
    agent.initialize()
    
    print("\nForecast Agent initialized. Type 'help' for available commands.")
    
    while True:
        try:
            command = input(f"\n{Fore.GREEN}Enter command: {Style.RESET_ALL}").strip()
            
            if command == "/bye":
                agent.save_session()
                print("Session saved. Goodbye!")
                break
            elif command == "help":
                print_help()
            elif command.startswith("analyze"):
                parts = command.split()
                if len(parts) != 4:
                    print("Usage: analyze <csv_path> <target_column> <series_id_column>")
                    continue
                    
                _, csv_path, target_column, series_id_column = parts
                df = pd.read_csv(csv_path)  # Read CSV first
                agent.current_data = df     # Store before analysis
                agent.current_context = {
                    'csv_path': csv_path,
                    'target_column': target_column,
                    'series_id_column': series_id_column,
                    'data_info': f"CSV with {len(df)} rows, columns: {', '.join(df.columns)}"
                }
                analysis = agent.process_dataset(csv_path, target_column, series_id_column)
                print(agent.tools_manager.format_analysis_output(analysis))
            else:
                # Handle as general query
                response = agent.process_query(command)
                print("\nAgent Response:")
                print(response)
                
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
