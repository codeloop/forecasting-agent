# Forecasting Agent

An intelligent agent for time series analysis and forecasting that combines LLM capabilities with statistical forecasting tools.

## Features

- Interactive command-line interface
- Automatic data analysis and visualization
- Time series forecasting with multiple models
- Dynamic code generation and execution
- Extensible tools system
- Conversation memory and context awareness
- Automatic dependency management

## Getting Started

### Prerequisites

- Python 3.12+
- Ollama (for LLM support)
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/codeloop/forecasting-agent.git
cd forecasting-agent
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install pandas numpy prophet darts tabulate langchain-experimental
```

4. Install Ollama and start the service:
```bash
# Follow Ollama installation instructions from: https://ollama.ai/
ollama serve
```

### Usage

1. Start the agent:
```bash
python main.py
```

2. Select an LLM model when prompted

3. Available commands:
- `analyze <csv_path> <target_column> <series_id_column>` - Load and analyze a dataset
- `help` - Show available commands
- `/bye` - Exit the program

4. Natural language queries:
- "Build a forecasting model for next 10 timesteps"
- "Show me the trend analysis"
- "Generate visualizations for each series"
- "Export forecasts to CSV"

5. Fix command:
- If execution produces empty/incorrect results, use:
- `fix <instructions>` (e.g., "fix write results to csv")

## Command Reference

### Basic Commands
```bash
analyze <csv_path> <target_column> <series_id_column>  # Load and analyze a dataset
help                                                   # Show available commands
/bye                                                   # Exit the program
```

### Natural Language Queries
- "Build a forecasting model for next 10 timesteps"
- "Show me the trend analysis"
- "Generate visualizations for each series"
- "Export forecasts to CSV"
- "Compare performance between series"
- "Show statistical summary"

### Fix Command
When code execution produces empty or incorrect results:
```bash
fix <instructions>
```
Examples:
- `fix write results to csv`
- `fix add error handling`
- `fix handle missing values`
- `fix add data validation`

### Interactive Commands During Code Execution
When code is generated, you can:
- Type `yes` to execute the code
- Type `no` to skip execution
- Type `quit` to cancel the operation

### Data Analysis Commands
```bash
# Load and analyze a dataset
analyze /path/to/data.csv target_column series_id_column

# Example
analyze sales_data.csv sales_amount store_id
```

### Forecasting Commands
Natural language examples:
- "Forecast next 10 timesteps for all series"
- "Generate hourly forecast for next week"
- "Predict monthly values with confidence intervals"
- "Create forecast with seasonal decomposition"

### Visualization Commands
Natural language examples:
- "Plot time series for each store"
- "Show trend comparison between series"
- "Generate forecast plots with confidence intervals"
- "Create seasonal decomposition plots"

### Export Commands
Natural language examples:
- "Export forecasts to CSV"
- "Save analysis results to file"
- "Export visualizations as PNG"
- "Generate PDF report"

## Architecture

### Core Components

1. **ForecastingAgent** (`src/agent.py`)
   - Main agent orchestrating all components
   - Handles user interactions
   - Manages conversation context
   - Coordinates code generation and execution

2. **MemoryManager** (`src/memory_manager.py`)
   - Manages conversation history
   - Stores analysis results
   - Maintains execution context
   - Persists sessions to disk

3. **ToolsManager** (`src/tools_manager.py`)
   - Handles code execution
   - Manages dependencies
   - Provides analysis tools
   - Formats output

4. **OllamaManager** (`src/ollama_manager.py`)
   - Manages LLM connection
   - Handles model selection
   - Provides retry logic

### Data Flow

1. User Input → Agent
2. Agent → LLM (for query understanding)
3. LLM → Code Generation
4. Code → Tools Manager
5. Results → Memory Manager
6. Formatted Output → User

## Implementation Details

### Code Generation

The system uses LLM to generate Python code based on user queries:
- Automatic import handling
- Dynamic dependency installation
- Error recovery and fixes
- Context-aware code generation

### Forecasting Capabilities

Supports multiple forecasting approaches:
- Prophet
- ARIMA
- Exponential Smoothing
- Custom models

### Error Handling

- Automatic dependency detection
- Package installation prompts
- Execution error recovery
- Code fix suggestions

### Memory Management

- Conversation history tracking
- Context preservation
- Session persistence
- Analysis caching

## Advanced Usage

### Custom Tools

You can extend the ToolsManager with custom tools:

```python
class ToolsManager:
    def add_custom_tool(self, name, func):
        setattr(self, name, func)
```

### Session Management

Sessions are automatically saved:

1. Session Files Location:
```
sessions/
  ├── YYYYMMDD_HHMMSS/
  │   └── memory.json
```

2. Session Data Contains:
- Conversation history
- Analysis results
- Generated code
- Execution results

### Environment Configuration

```bash
# Ollama Configuration
export OLLAMA_HOST=localhost  # Default
export OLLAMA_PORT=11434     # Default

# Memory Management
export FC_AGENT_MEMORY_SIZE=1000  # Number of interactions to keep
export FC_AGENT_SESSION_DIR=/path/to/sessions
```

## Best Practices

1. Data Preparation
- Ensure consistent date format
- Handle missing values before analysis
- Check for duplicate timestamps
- Verify data types

2. Forecasting
- Start with simple models
- Validate forecasts with test data
- Consider seasonality
- Check for outliers

3. Code Generation
- Review generated code before execution
- Use fix command for refinements
- Save successful code for reuse
- Add error handling

4. Memory Management
- Regular session saves
- Clear old sessions
- Export important results
- Document modifications

## Troubleshooting

1. Code Execution Issues
- Check package installations
- Verify data format
- Review error messages
- Use fix command with specific instructions

2. Memory Issues
- Clear python cache
- Restart agent
- Export results frequently
- Use smaller data chunks

3. LLM Connection
- Check Ollama service
- Verify model availability
- Check network connection
- Review API responses

## Publishing to PyPI

1. Build the package:
```bash
# Install build tools
pip install --upgrade build twine

# Build package
python -m build
```

2. Test on TestPyPI first:
```bash
# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*

# Test installation
pip install --index-url https://test.pypi.org/simple/ fc-agent
```

3. Publish to PyPI:
```bash
# Upload to PyPI
python -m twine upload dist/*
```

4. Installation after publishing:
```bash
pip install fc-agent
```

5. Version Updates:
- Update version in `setup.py` and `pyproject.toml`
- Create new build
- Upload to PyPI

# Install production dependencies
pip install -r requirements.txt

# Install with dev tools
pip install -e ".[dev]"

# Build the package
python -m build