import json
import os
from datetime import datetime

class MemoryManager:
    """
    Manages conversation history, execution context, and session persistence.
    
    Handles:
    - Short-term and long-term memory storage
    - Conversation history tracking
    - Session management and persistence
    - Context retrieval for LLM
    
    Attributes:
        session_id (str): Unique identifier for current session
        short_term_memory (list): Recent interactions and context
        long_term_memory (list): Persistent storage of important information
        conversation_history (list): Complete interaction history
        last_execution (dict): Most recent execution details
    """

    def __init__(self):
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.short_term_memory = []
        self.long_term_memory = []
        self.conversation_history = []
        self.last_execution = None
        
    def store_dataset_info(self, info):
        """
        Store information about the loaded dataset.
        
        Args:
            info (dict): Dataset metadata including shape, columns, etc.
        """
        self.short_term_memory.append({
            'type': 'dataset_info',
            'content': info,
            'timestamp': datetime.now().isoformat()
        })
        
    def store_analysis(self, analysis):
        """
        Store analysis results in memory.
        
        Args:
            analysis (dict): Analysis results and statistics
        """
        self.short_term_memory.append({
            'type': 'analysis',
            'content': analysis,
            'timestamp': datetime.now().isoformat()
        })
        
    def save_to_disk(self):
        memory_dir = os.path.join('sessions', self.session_id)
        os.makedirs(memory_dir, exist_ok=True)
        
        with open(os.path.join(memory_dir, 'memory.json'), 'w') as f:
            json.dump({
                'short_term': self.short_term_memory,
                'long_term': self.long_term_memory
            }, f)
    
    def store_interaction(self, query, response, code=None, error=None, fixes=None, result=None):
        """
        Store a complete interaction including query, response, and execution details.
        
        Args:
            query (str): User's original query
            response (str): System's response
            code (str, optional): Generated/executed code
            error (str, optional): Error message if any
            fixes (list, optional): Applied fixes
            result (str, optional): Execution results
        """
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'response': response,
            'code': code,
            'error': error,
            'fixes': fixes,
            'result': result
        }
        self.conversation_history.append(interaction)
        self.last_execution = interaction
        self.short_term_memory.append({
            'type': 'interaction',
            'content': interaction,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_relevant_context(self):
        """
        Retrieve relevant context for current interaction.
        
        Returns:
            dict: Context including recent analysis and conversation history
        """
        context = {}
        
        # Get most recent analysis
        for item in reversed(self.short_term_memory):
            if item['type'] == 'analysis':
                context['last_analysis'] = item['content']
                break
        
        # Get conversation history
        context['conversation_history'] = self.conversation_history
        
        return context
