import os
import json
import pandas as pd
from .memory_manager import MemoryManager
from .tools_manager import ToolsManager
from .planner import Planner
from .ollama_manager import get_available_models, initialize_llm

class ForecastingAgent:
    """
    A forecasting agent that combines LLM capabilities with statistical forecasting.
    
    This agent processes natural language queries, generates and executes code,
    and manages the interaction between different components like memory management,
    tools, and the LLM interface.
    
    Attributes:
        memory_manager (MemoryManager): Manages conversation and execution history
        _tools_manager (ToolsManager): Handles code execution and tool management
        planner (Planner): Manages execution planning
        llm: LLM interface for natural language processing
        current_data (pd.DataFrame): Currently loaded dataset
        current_context (dict): Current execution context
        _last_code (str): Last executed code
        _last_error (str): Last error message
        _last_result (str): Last execution result
    """
    def __init__(self):
        self.memory_manager = MemoryManager()
        self._tools_manager = ToolsManager()  # renamed to _tools_manager
        self.planner = Planner()
        self.llm = None
        self.current_data = None
        self.current_context = {}
        self._last_code = None
        self._last_error = None
        self._last_result = None
        
    @property
    def tools_manager(self):
        return self._tools_manager  # return the internal variable
    
    def initialize(self):
        # Get available Ollama models
        models = get_available_models()
        print("Available models:")
        for idx, model in enumerate(models):
            print(f"{idx+1}. {model}")
        
        model_idx = int(input("Select model number: ")) - 1
        self.llm = initialize_llm(models[model_idx])
        
    def process_dataset(self, csv_path, target_column, series_id_column):
        """
        Load and analyze a dataset.
        
        Args:
            csv_path (str): Path to the CSV file
            target_column (str): Name of the target variable column
            series_id_column (str): Name of the series identifier column
            
        Returns:
            dict: Analysis results including statistics and insights
        """
        # Read dataset
        df = pd.read_csv(csv_path)
        
        # Store basic information in memory
        self.memory_manager.store_dataset_info({
            'target_column': target_column,
            'series_id_column': series_id_column,
            'shape': df.shape,
            'columns': df.columns.tolist()
        })
        
        # Generate descriptive analysis
        analysis = self.tools_manager.generate_descriptive_analysis(df, target_column, series_id_column)
        self.memory_manager.store_analysis(analysis)
        
        return analysis

    def process_query(self, query):
        """
        Process a natural language query from the user.
        
        Handles various types of queries including:
        - Fix commands for code improvement
        - Natural language questions
        - Analysis requests
        - Forecasting requests
        
        Args:
            query (str): User's natural language query
            
        Returns:
            str: Response or execution results
        """
        if query.lower().startswith('fix'):
            if not self._last_code or not self._last_result:
                return "No previous code execution to fix. Please run a command first."
            
            fix_instructions = query[3:].strip()
            if not fix_instructions:
                return "Please provide instructions for the fix, e.g., 'fix write results to csv'"
                
            prompt = f"""Previous execution resulted in:

            Code:
            ```python
            {self._last_code}
            ```

            Results:
            {self._last_result}

            User wants to: {fix_instructions}

            Please provide fixed code that:
            1. Keeps the core functionality
            2. {fix_instructions}
            3. Handles all edge cases
            
            Respond with:
            
            ERROR ANALYSIS:
            <explain what needs to be fixed>
            
            PROPOSED FIXES:
            <list specific changes>
            
            CODE:
            ```python
            <fixed code>
            ```
            
            EXPLANATION:
            <explain how the fixes work>
            """
            
            response = self.llm.invoke(prompt)
            return self._plan_and_execute(response, query)
            
        if not self.llm:
            return "Error: LLM not initialized"
            
        try:
            context = self.memory_manager.get_relevant_context()
            
            # Format conversation history
            history_text = ""
            if context.get('conversation_history'):
                history_text = "\nPrevious interactions:\n"
                for interaction in context['conversation_history'][-5:]:  # Last 5 interactions
                    history_text += f"\nUser: {interaction['query']}\n"
                    history_text += f"Assistant: {interaction['response']}\n"
                    if interaction.get('error'):
                        history_text += f"Error: {interaction['error']}\n"
                    if interaction.get('fixes'):
                        history_text += f"Fixes: {interaction['fixes']}\n"
            
            prompt = f"""You are a forecasting assistant. Current context:
            Data available: {self.current_context.get('data_info', 'No data loaded')}
            Data structure:
            - CSV file: {self.current_context.get('csv_path')}
            - Target column: {self.current_context.get('target_column')}
            - Series ID column: {self.current_context.get('series_id_column')}
            Previous analysis: {context.get('last_analysis', 'None')}
            
            {history_text}
            
            User query: {query}
            
            If you need to generate code, respond with:
            ACTION: CODE_GENERATION
            CODE:
            ```python
            <your code here>
            ```
            EXPLANATION: <explain what the code does>
            
            For other actions, respond with:
            ACTION: <DATA_ANALYSIS|FORECAST|GENERAL>
            EXPLANATION: <why this action>
            TOOLS_NEEDED: <list of required tools>
            """
            
            response = self.llm.invoke(prompt)
            # Store the interaction
            self.memory_manager.store_interaction(query, str(response))
            return self._plan_and_execute(response, query)
            
        except Exception as e:
            return f"Error processing query: {str(e)}"

    def _plan_and_execute(self, llm_response, original_query):
        """
        Plan and execute actions based on LLM response.
        
        Args:
            llm_response (str): Response from the LLM
            original_query (str): Original user query
            
        Returns:
            str: Execution results or error message
            
        Handles:
        - Code generation and execution
        - Fix attempts for failed code
        - Different types of actions (FORECAST, DATA_ANALYSIS, etc.)
        """
        try:
            response_text = str(llm_response)
            action = None
            code = None
            explanation = None
            
            # Handle fix command with instructions
            if original_query.lower().startswith('fix'):
                if not hasattr(self, '_last_code') or not hasattr(self, '_last_error'):
                    return "No previous code execution found to fix"
                    
                fix_instructions = original_query[3:].strip()  # Remove 'fix' and get instructions
                fix_prompt = f"""The previous code execution produced empty or incorrect results.
                User instructions: {fix_instructions}

                Previous Code:
                ```python
                {self._last_code}
                ```

                Last Error/Output:
                {self._last_error}

                Data Context:
                - DataFrame 'df' contains {len(self.current_data)} rows
                - Columns: {', '.join(self.current_data.columns)}
                - Target column: '{self.current_context.get('target_column')}'
                - Series ID column: '{self.current_context.get('series_id_column')}'
                - Date column format: {self.current_data['date'].dtype}

                Please fix the code according to the user's instructions.
                
                Respond in this format:
                
                ERROR ANALYSIS:
                <explain what was wrong with the previous output>
                
                PROPOSED FIXES:
                <list the specific changes being made based on user instructions>
                
                CODE:
                ```python
                <corrected code>
                ```
                
                EXPLANATION:
                <explain how the fixes address the issue>
                """
                
                fix_response = self.llm.invoke(fix_prompt)
                return self._plan_and_execute(fix_response, original_query)

            # Parse the response
            if "ACTION: CODE_GENERATION" in response_text:
                action = "CODE_GENERATION"
                # Extract code between ```python and ``` markers
                start = response_text.find("```python\n") + 10
                end = response_text.find("```", start)
                if start > 9 and end > start:
                    code = response_text[start:end].strip()
                    
                # Extract explanation
                if "EXPLANATION:" in response_text:
                    explanation = response_text.split("EXPLANATION:", 1)[1].strip()
            else:
                for line in response_text.split('\n'):
                    if line.startswith('ACTION:'):
                        action = line.split(':', 1)[1].strip().upper()
                    elif line.startswith('EXPLANATION:'):
                        explanation = line.split(':', 1)[1].strip()
            
            if not action:
                return "Couldn't determine appropriate action"
                
            if action == "CODE_GENERATION":
                if not code:
                    return "No code was generated"

                attempt = 0
                attempt_history = []
                while True:  # Unlimited retries
                    attempt += 1
                    print(f"\nAttempt {attempt}")
                    print("\nGenerated Code:")
                    print("```python")
                    print(code)
                    print("```")
                    print("\nCode Explanation:", explanation)
                    
                    user_input = input("\nWould you like to execute this code? (yes/no/quit): ").lower()
                    if user_input == 'quit':
                        return "Code execution cancelled by user"
                    if user_input != 'yes':
                        continue  # Ask for execution again
                    
                    result = self.tools_manager.execute_code(code, df=self.current_data)
                    
                    # Store last code and result for potential fixes
                    self._last_code = code
                    self._last_error = result['output'] if not result['success'] else str(result['results'])
                    self._last_result = self.tools_manager.format_code_output(result)
                    
                    if result['success']:
                        return self._last_result
                    
                    # Store attempt history
                    attempt_history.append({
                        'attempt': attempt,
                        'code': code,
                        'error': result['output'],
                        'debug_info': result.get('debug_info', {}),
                        'error_context': result.get('error_context', {})
                    })
                    
                    # If execution failed, ask LLM to fix the code
                    print(f"\nExecution failed (Attempt {attempt})")
                    print("Error details:")
                    print(result['output'])
                    print("\nAsking LLM to fix the code...")
                    
                    fix_prompt = f"""The code execution failed. Here's the context:

                    Error Message:
                    {result['output']}
                    
                    Failed Code:
                    ```python
                    {code}
                    ```
                    
                    Data Context:
                    - DataFrame 'df' contains {len(self.current_data)} rows
                    - Columns: {', '.join(self.current_data.columns)}
                    - Target column: '{self.current_context.get('target_column')}'
                    - Series ID column: '{self.current_context.get('series_id_column')}'
                    - Date column format: {self.current_data['date'].dtype}
                    
                    Previous Attempts Summary:
                    {json.dumps([{'attempt': a['attempt'], 'error': a['error']} 
                              for a in attempt_history], indent=2)}
                    
                    Please:
                    1. Analyze the error message and explain what's wrong
                    2. Provide a detailed explanation of the fixes needed
                    3. Provide the corrected code
                    4. Add error handling for similar issues
                    
                    Respond in this format:
                    
                    ERROR ANALYSIS:
                    <explain what caused the error>
                    
                    PROPOSED FIXES:
                    <list the specific changes being made>
                    
                    CODE:
                    ```python
                    <corrected code>
                    ```
                    
                    EXPLANATION:
                    <explain how the fixes address the error>
                    """
                    
                    fix_response = self.llm.invoke(fix_prompt)
                    try:
                        new_code = str(fix_response)
                        # Extract error analysis
                        if "ERROR ANALYSIS:" in new_code:
                            print("\nError Analysis:")
                            print(new_code.split("ERROR ANALYSIS:", 1)[1].split("PROPOSED FIXES:", 1)[0].strip())
                        
                        # Extract proposed fixes
                        if "PROPOSED FIXES:" in new_code:
                            print("\nProposed Fixes:")
                            print(new_code.split("PROPOSED FIXES:", 1)[1].split("CODE:", 1)[0].strip())
                        
                        # Extract code
                        start = new_code.find("```python\n")
                        end = new_code.find("```", start + 10)
                        if start >= 0 and end > start + 10:
                            code = new_code[start + 10:end].strip()
                            # Extract explanation if available
                            if "EXPLANATION:" in new_code[end:]:
                                explanation = new_code[end:].split("EXPLANATION:", 1)[1].strip()
                            continue
                        else:
                            print("Error: Could not find code in LLM response")
                            print("Would you like to:")
                            print("1. Retry with the same code")
                            print("2. Try a different approach")
                            print("3. Quit")
                            choice = input("Enter choice (1/2/3): ")
                            if choice == '2':
                                # Generate completely new code
                                return self.process_query(original_query)
                            elif choice == '3':
                                return "Code execution cancelled by user"
                            # Otherwise continue with same code
                    except Exception as e:
                        print(f"Error parsing LLM response: {e}")
                        user_input = input("Would you like to try again? (yes/no): ")
                        if user_input.lower() != 'yes':
                            break

                    # Store the error and fixes in memory
                    self.memory_manager.store_interaction(
                        original_query,
                        str(fix_response),
                        code=code,
                        error=result['output'],
                        fixes=attempt_history
                    )

            if action == "FORECAST":
                if self.current_data is None:
                    return "Please analyze a dataset first using the 'analyze' command."
                return self.tools_manager.generate_forecast(
                    self.current_data,
                    self.current_context.get('target_column'),
                    self.current_context.get('series_id_column')
                )
            elif action == "DATA_ANALYSIS":
                return self.process_dataset(
                    self.current_context.get('csv_path'),
                    self.current_context.get('target_column'),
                    self.current_context.get('series_id_column')
                )
            elif action == "CODE_EXECUTION":
                if self.current_data is None:
                    return "Please analyze a dataset first using the 'analyze' command."
                    
                # Ask user for confirmation
                print("\nProposed code to execute:")
                print(explanation)
                user_input = input("\nDo you want to execute this code? (yes/no): ")
                
                if user_input.lower() == 'yes':
                    result = self.tools_manager.execute_code(
                        explanation,
                        df=self.current_data
                    )
                    return self.tools_manager.format_code_output(result)
                else:
                    return "Code execution cancelled."
            else:
                return self.llm.invoke(original_query)
                
        except Exception as e:
            return f"Error executing action: {str(e)}"

    def save_session(self):
        self.memory_manager.save_to_disk()
