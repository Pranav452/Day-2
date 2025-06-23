"""
Example Transcripts: Tool-Calling Agent for Character Counting and Arithmetic
============================================================================

This file contains 5 example transcripts showing how the agent handles:
- 3 counting queries (character/substring counting)
- 2 arithmetic queries (calculations and derived quantities)

Each transcript shows the full interaction flow:
1. User input
2. Tool call (if any)
3. Tool output
4. Final LLM response
"""

def display_transcripts():
    """Display all example transcripts."""
    
    transcripts = [
        {
            "title": "Example 1: Character Counting",
            "user_input": "How many 'r' characters are in 'strawberry'?",
            "tool_call": {
                "tool": "python_exec",
                "parameters": {"code": "len([c for c in 'strawberry' if c == 'r'])"}
            },
            "tool_output": "3",
            "final_response": "There are 3 'r' characters in 'strawberry'."
        },
        {
            "title": "Example 2: Substring Counting",
            "user_input": "Count the occurrences of 'the' in 'the quick brown fox jumps over the lazy dog'",
            "tool_call": {
                "tool": "python_exec", 
                "parameters": {"code": "text = 'the quick brown fox jumps over the lazy dog'; text.lower().split().count('the')"}
            },
            "tool_output": "2",
            "final_response": "The word 'the' appears 2 times in that sentence."
        },
        {
            "title": "Example 3: Vowel Counting",
            "user_input": "How many vowels (a, e, i, o, u) are in 'artificial intelligence'?",
            "tool_call": {
                "tool": "python_exec",
                "parameters": {"code": "text = 'artificial intelligence'; vowels = 'aeiou'; sum(1 for c in text.lower() if c in vowels)"}
            },
            "tool_output": "8", 
            "final_response": "There are 8 vowels in 'artificial intelligence'."
        },
        {
            "title": "Example 4: Arithmetic Expression",
            "user_input": "What is 15 * 23 + 7?",
            "tool_call": {
                "tool": "python_exec",
                "parameters": {"code": "15 * 23 + 7"}
            },
            "tool_output": "352",
            "final_response": "15 × 23 + 7 = 352"
        },
        {
            "title": "Example 5: Statistical Calculation",
            "user_input": "Calculate the average of these numbers: 12, 15, 18, 22, 25, 30",
            "tool_call": {
                "tool": "python_exec",
                "parameters": {"code": "numbers = [12, 15, 18, 22, 25, 30]; sum(numbers) / len(numbers)"}
            },
            "tool_output": "20.333333333333332",
            "final_response": "The average of those numbers is approximately 20.33."
        }
    ]
    
    print("TOOL-CALLING AGENT TRANSCRIPTS")
    print("=" * 50)
    
    for i, transcript in enumerate(transcripts, 1):
        print(f"\n{transcript['title']}")
        print("-" * len(transcript['title']))
        
        print(f"\n🔵 User: {transcript['user_input']}")
        
        if transcript.get('tool_call'):
            print(f"\n🔧 Tool Call:")
            print(f"   Tool: {transcript['tool_call']['tool']}")
            if transcript['tool_call']['parameters']:
                print(f"   Code: {transcript['tool_call']['parameters']['code']}")
            
            print(f"\n📊 Tool Output: {transcript['tool_output']}")
        
        print(f"\n🤖 Assistant: {transcript['final_response']}")
        
        if i < len(transcripts):
            print("\n" + "="*50)
    
    return transcripts

if __name__ == "__main__":
    transcripts = display_transcripts()
    print(f"\n\nSummary: Generated {len(transcripts)} transcripts")
    print("- 3 counting queries (character/substring/vowel counting)")
    print("- 2 arithmetic queries (expression evaluation and statistical calculation)") 