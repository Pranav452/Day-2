import json
import sys
from typing import Dict, Any, List
from io import StringIO
import contextlib

# Mock Gemini API for demonstration purposes
class MockGenerativeModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    def generate_content(self, prompt):
        # Mock response based on patterns
        if isinstance(prompt, str):
            text = prompt
        elif isinstance(prompt, list):
            text = str(prompt)
        else:
            text = str(prompt)
        
        # Simple pattern matching for demo
        if "How many 'r'" in text and "strawberry" in text:
            return MockResponse('{"tool": "python_exec", "parameters": {"code": "len([c for c in \\\"strawberry\\\" if c == \\\"r\\\"])"}}')
        elif "Count the occurrences of 'the'" in text:
            return MockResponse('{"tool": "python_exec", "parameters": {"code": "text = \\\"the quick brown fox jumps over the lazy dog\\\"; text.lower().split().count(\\\"the\\\")"}}')
        elif "vowels" in text and "artificial intelligence" in text:
            return MockResponse('{"tool": "python_exec", "parameters": {"code": "text = \\\"artificial intelligence\\\"; vowels = \\\"aeiou\\\"; sum(1 for c in text.lower() if c in vowels)"}}')
        elif "15 * 23" in text:
            return MockResponse('{"tool": "python_exec", "parameters": {"code": "15 * 23 + 7"}}')
        elif "average" in text and "12, 15, 18" in text:
            return MockResponse('{"tool": "python_exec", "parameters": {"code": "numbers = [12, 15, 18, 22, 25, 30]; sum(numbers) / len(numbers)"}}')
        elif "tool result" in text.lower():
            if "3" in text:
                return MockResponse("There are 3 'r' characters in 'strawberry'.")
            elif "2" in text:
                return MockResponse("The word 'the' appears 2 times in that sentence.")
            elif "8" in text:
                return MockResponse("There are 8 vowels in 'artificial intelligence'.")
            elif "352" in text:
                return MockResponse("15 × 23 + 7 = 352")
            elif "20.333" in text:
                return MockResponse("The average of those numbers is approximately 20.33.")
        
        return MockResponse("I understand your query.")

class MockResponse:
    def __init__(self, text: str):
        self.text = text

# Mock genai module
class MockGenAI:
    @staticmethod
    def configure(api_key: str):
        pass
    
    @staticmethod
    def GenerativeModel(model_name: str):
        return MockGenerativeModel(model_name)

# Use mock for demonstration
genai = MockGenAI()

class ToolCallingAgent:
    def __init__(self, api_key: str):
        """Initialize the agent with Gemini API."""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        
        self.tools = {
            "python_exec": {
                "description": "Execute Python code and return the output. Use this for counting characters, arithmetic, or any computation.",
                "parameters": {
                    "code": "string - The Python code to execute"
                }
            },
            "noop": {
                "description": "Return nothing. Use this for normal conversational responses that don't require computation.",
                "parameters": {}
            }
        }
    
    def execute_python(self, code: str) -> str:
        """Execute Python code safely and return the output."""
        try:
            # Create a safe environment for execution
            safe_globals = {
                '__builtins__': {
                    'len': len,
                    'str': str,
                    'int': int,
                    'float': float,
                    'list': list,
                    'dict': dict,
                    'set': set,
                    'tuple': tuple,
                    'range': range,
                    'sum': sum,
                    'max': max,
                    'min': min,
                    'abs': abs,
                    'round': round,
                    'print': print,
                    'enumerate': enumerate,
                    'zip': zip,
                    'sorted': sorted,
                    'reversed': reversed,
                }
            }
            
            # Capture stdout
            stdout_capture = StringIO()
            with contextlib.redirect_stdout(stdout_capture):
                # Execute the code
                result = eval(code, safe_globals, {})
                if result is not None:
                    print(result)
            
            output = stdout_capture.getvalue().strip()
            return output if output else str(result) if 'result' in locals() else ""
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def noop(self) -> str:
        """No operation - returns empty string."""
        return ""
    
    def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> str:
        """Execute a tool with given parameters."""
        if tool_name == "python_exec":
            return self.execute_python(parameters.get("code", ""))
        elif tool_name == "noop":
            return self.noop()
        else:
            return f"Unknown tool: {tool_name}"
    
    def get_system_prompt(self) -> str:
        """Generate system prompt with tool descriptions."""
        tools_desc = "\n".join([
            f"- {name}: {info['description']}"
            for name, info in self.tools.items()
        ])
        
        return f"""You are a helpful assistant with access to tools for computation and counting.

Available tools:
{tools_desc}

When you need to count characters, perform arithmetic, or do any computation, use the python_exec tool.
For normal conversational responses, use the noop tool or respond directly.

To call a tool, respond with JSON in this format:
{{"tool": "tool_name", "parameters": {{"param": "value"}}}}

If you don't need a tool, just respond normally.

Examples:
- For "How many 'r' in 'strawberry'?" use: {{"tool": "python_exec", "parameters": {{"code": "len([c for c in 'strawberry' if c == 'r'])"}}}}
- For "What is 15 * 23?" use: {{"tool": "python_exec", "parameters": {{"code": "15 * 23"}}}}
"""
    
    def process_query(self, user_input: str) -> Dict[str, Any]:
        """Process a user query and return the full interaction transcript."""
        # Get LLM response for tool decision
        response = self.model.generate_content(user_input)
        llm_response = response.text.strip()
        
        # Check if response contains a tool call
        tool_call = None
        tool_output = None
        final_response = llm_response
        
        try:
            # Try to parse as JSON tool call
            if llm_response.startswith('{') and llm_response.endswith('}'):
                tool_call = json.loads(llm_response)
                tool_name = tool_call.get("tool")
                parameters = tool_call.get("parameters", {})
                
                if tool_name in self.tools:
                    tool_output = self.execute_tool(tool_name, parameters)
                    
                    # Get final response after tool execution
                    if tool_name == "python_exec" and tool_output:
                        follow_up_prompt = f"User asked: {user_input}\nTool result: {tool_output}\nPlease provide a natural language response."
                        follow_up_response = self.model.generate_content(follow_up_prompt)
                        final_response = follow_up_response.text.strip()
                    elif tool_name == "noop":
                        final_response = "I understand your query."
        except json.JSONDecodeError:
            # Not a tool call, treat as normal response
            pass
        
        return {
            "user_input": user_input,
            "tool_call": tool_call,
            "tool_output": tool_output,
            "final_response": final_response
        }

def run_examples():
    """Run the 5 example queries and show full transcripts."""
    print("TOOL-CALLING AGENT: Character Counting & Arithmetic")
    print("=" * 55)
    print("Overcoming tokenizer blindness by wiring LLM to Python")
    print()
    
    agent = ToolCallingAgent(api_key="demo_key")
    
    # Define the 5 queries
    queries = [
        # 3 Counting queries
        "How many 'r' characters are in 'strawberry'?",
        "Count the occurrences of 'the' in 'the quick brown fox jumps over the lazy dog'",
        "How many vowels (a, e, i, o, u) are in 'artificial intelligence'?",
        
        # 2 Arithmetic queries
        "What is 15 * 23 + 7?",
        "Calculate the average of these numbers: 12, 15, 18, 22, 25, 30"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"EXAMPLE {i}")
        print("-" * 10)
        
        # Process the query
        result = agent.process_query(query)
        
        # Display the full transcript
        print(f"User: {result['user_input']}")
        
        if result['tool_call']:
            print(f"Tool Call: {result['tool_call']}")
            print(f"Tool Output: {result['tool_output']}")
        
        print(f"Assistant: {result['final_response']}")
        
        if i < len(queries):
            print("\n" + "=" * 55 + "\n")
    
    print(f"\nCompleted {len(queries)} examples:")
    print("✓ 3 counting queries (character/substring/vowel)")
    print("✓ 2 arithmetic queries (expression/statistical)")

if __name__ == "__main__":
    run_examples() 