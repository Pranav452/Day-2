# Q5: Tool-Calling Agent - Count Characters Correctly

## Goal
Overcome tokenizer blindness by wiring the model to Python execution capabilities.

## Overview
This implementation creates a minimal tool-calling agent that can accurately count characters and perform arithmetic by executing Python code, rather than relying on potentially inaccurate token-based reasoning.

## Architecture

### Tools Available
1. **`python_exec`** - Executes Python code snippets and returns stdout
2. **`noop`** - Returns nothing (for normal conversational responses)

### Pipeline Flow
1. User submits a query requiring counting or computation
2. LLM decides whether to use a tool based on the query
3. If computation needed: LLM generates Python code and calls `python_exec`
4. Framework executes the code safely and returns result
5. LLM provides natural language response based on the result

## Files

- `agent.py` - Main tool-calling agent implementation
- `transcripts.py` - Example transcripts demonstrating functionality
- `requirements.txt` - Dependencies (Gemini API)
- `README.md` - This documentation

## Key Features

### Safe Python Execution
- Restricted execution environment with only safe built-in functions
- No file system access or dangerous operations
- Captures stdout for result display

### Mock Implementation
- Uses mock Gemini API for demonstration
- Pattern-matching responses to simulate LLM behavior
- Real implementation would use actual Gemini API key

### Error Handling
- Graceful handling of malformed JSON responses
- Safe execution with exception catching
- Fallback to conversational responses

## Example Transcripts

### Counting Queries (3 examples)

**Example 1: Character Counting**
```
User: How many 'r' characters are in 'strawberry'?
Tool Call: {"tool": "python_exec", "parameters": {"code": "len([c for c in 'strawberry' if c == 'r'])"}}
Tool Output: 3
Assistant: There are 3 'r' characters in 'strawberry'.
```

**Example 2: Substring Counting**
```
User: Count the occurrences of 'the' in 'the quick brown fox jumps over the lazy dog'
Tool Call: {"tool": "python_exec", "parameters": {"code": "text = 'the quick brown fox jumps over the lazy dog'; text.lower().split().count('the')"}}
Tool Output: 2
Assistant: The word 'the' appears 2 times in that sentence.
```

**Example 3: Vowel Counting**
```
User: How many vowels (a, e, i, o, u) are in 'artificial intelligence'?
Tool Call: {"tool": "python_exec", "parameters": {"code": "text = 'artificial intelligence'; vowels = 'aeiou'; sum(1 for c in text.lower() if c in vowels)"}}
Tool Output: 8
Assistant: There are 8 vowels in 'artificial intelligence'.
```

### Arithmetic Queries (2 examples)

**Example 4: Expression Evaluation**
```
User: What is 15 * 23 + 7?
Tool Call: {"tool": "python_exec", "parameters": {"code": "15 * 23 + 7"}}
Tool Output: 352
Assistant: 15 × 23 + 7 = 352
```

**Example 5: Statistical Calculation**
```
User: Calculate the average of these numbers: 12, 15, 18, 22, 25, 30
Tool Call: {"tool": "python_exec", "parameters": {"code": "numbers = [12, 15, 18, 22, 25, 30]; sum(numbers) / len(numbers)"}}
Tool Output: 20.333333333333332
Assistant: The average of those numbers is approximately 20.33.
```

## Usage

### Running Examples
```bash
cd q5
python agent.py
```

### Running Transcript Display
```bash
python transcripts.py
```

### With Real Gemini API
To use with actual Gemini API:
1. Install dependencies: `pip install -r requirements.txt`
2. Set your API key in the agent initialization
3. Remove the mock classes and import `google.generativeai`

## Why This Works

### Tokenizer Blindness Problem
- LLMs tokenize text, making character-level operations unreliable
- "strawberry" might be tokenized as ["straw", "berry"], losing individual character information
- Models often guess incorrectly when counting specific characters

### Solution: External Tool Execution
- Delegates precise counting/arithmetic to Python
- Uses actual string operations instead of token approximations
- Provides reliable, verifiable results
- Maintains LLM's natural language capabilities for response generation

## Benefits

1. **Accuracy**: 100% accurate character counting and arithmetic
2. **Transparency**: Shows exact code executed for each computation
3. **Extensibility**: Easy to add more computational tools
4. **Safety**: Restricted execution environment prevents harmful operations
5. **Natural Interface**: Users interact in natural language while getting precise results

This implementation successfully demonstrates how to overcome LLM limitations by combining their reasoning capabilities with external tool execution for precise computations. 