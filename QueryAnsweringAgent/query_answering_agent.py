import sys
from typing import TypedDict, List, Dict, Optional
from openai import AzureOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
import os
import re
import subprocess
import json
load_dotenv('environment_variables')


# keys
MODEL_4o = 'gpt-4o-mini'
OPENAI_API_VERSION_4o = '2024-08-01-preview'
AZURE_OPENAI_API_KEY = os.getenv('CLASS_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT_4o = os.getenv('SUBSCRIPTION_OPENAI_ENDPOINT_4o')

client = AzureOpenAI(
    api_key= AZURE_OPENAI_API_KEY, 
    api_version= OPENAI_API_VERSION_4o, 
    azure_endpoint = AZURE_OPENAI_ENDPOINT_4o
    )

MAX_ATTEMPTS = 2


class MyState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        query_name: The name of the query.
        query: The content of the query file.
        data_files: A list of dictionaries, where each dictionary contains
                    'name' (file name) and 'description' (file description).
        query_program: The Python code generated to answer the query.
        query_answer: The result of executing the query program (JSON string).
        program_errors: Any errors encountered during program execution.
        reflection: The LLM's reflection on errors.
        prompt_history: A list of prompts used to generate/regenerate code.
        attempts: The number of attempts made to generate a correct program.
    """
    query_name: str
    query: str
    data_files: List[Dict[str, str]]
    query_program: str
    query_answer: str
    program_errors: List[str]
    reflection: str
    prompt_history: List[str]
    attempts: int
    chk_for_errors_map: str


def clean_code_block(response_content: str) -> str:
    # Remove triple backticks and optional 'python' specifier
    if response_content.startswith("```"):
        lines = response_content.split("\n")
        if lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines)
    return response_content


# Placeholder for LLM interaction - replace with your actual LLM client
def get_llm_response(prompt: str) -> str:
    """
    Generate a response from LLM based on a prompt.
    """

    response = client.chat.completions.create(
        model=MODEL_4o,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=750
    )
    
    # Extract and return the response content
    res = clean_code_block(response.choices[0].message.content)
    # print(f"get_llm_response res:\n{res}")
    return res


def GetQueryDetails(state: MyState) -> MyState:
    """
    This tool reads the input files (query_input.txt and <name_of_query>.txt)
    and initializes the state of the graph.
    """
    print("** Entering GetQueryDetails Tool **")
    state["query_program"] = ""
    state["query_answer"] = ""
    state["program_errors"] = []
    state["reflection"] = ""
    state["prompt_history"] = []
    state["attempts"] = 0
    state["chk_for_errors_map"] = ""
    data_files = []

    try:
        with open("query_input.txt", "r") as f:
            content = f.read()

        # Extract query_name
        query_name_match = re.search(r"query_name:\s*(.*?)\s*(?=data_file:|\Z)", content, re.DOTALL)  # re.search(r"query_name:([^\n]+)", content)
        print(f"query_name_match:\t {query_name_match}")
        if query_name_match:
            query_name = query_name_match.group(1).strip()
            query_file_path = f"{query_name}.txt" # We still need this to open the query file
        else:
            raise ValueError("query_name not found in query_input.txt")

        # Read the query file content
        try:
            with open(query_file_path, "r") as qf:
                query_content = qf.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Query file '{query_file_path}' not found.")
        print(f"query_content:\t {query_content}")

        # Extract data files and descriptions
        data_file_pattern = re.compile(r"data_file:\s*([^\n]+)\s*description:(.*?)(?=data_file:|\Z)", re.DOTALL)  # re.compile(r"data_file:([^\n]+)\s*description:([^\n]+)", re.DOTALL)
        for match in data_file_pattern.finditer(content):
            file_name = match.group(1).strip()
            description = match.group(2).strip()
            data_files.append({"name": file_name, "description": description})

        if not data_files:
            raise ValueError("No data_file and description found in query_input.txt")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        state["program_errors"].append(f"Error: {e}")
        return state
    except Exception as e:
        print(f"Error parsing input files: {e}")
        state["program_errors"].append(f"Error parsing input files: {e}")
        return state

    # Initialize other state variables
    state["query_name"] = query_name
    state["query"] = query_content # Storing the content directly
    state["data_files"] = data_files
    return state


def GenQueryProgram(state: MyState) -> MyState:
    """
    This tool prompts the LLM to generate the query program PQ.
    """
    print("** Entering GenQueryProgram Tool **")

    query = state["query"]
    data_files = state["data_files"]

    # Construct the data file descriptions for the prompt
    data_file_descriptions = ""
    for df in data_files:
        data_file_descriptions += (
            f"- File: `{df['name']}`\n"
            f"  Description: {df['description']}\n"
            f"  Note: The first line of this CSV file contains the column names.\n"
        )

    # Define the prompt for the LLM
    prompt = f"""
You are an expert Python programmer tasked with writing a program to answer a specific query using provided CSV data files.

Here is the query you need to answer:
---
{query}
---

Here are the data files available, with their names and descriptions:
---
{data_file_descriptions}
---

Your program must perform the following:
1.  Read and process the data from the specified CSV files.
2.  Implement the logic required to answer the query.
3.  The final output of your program must be a **JSON string** printed to standard output.
    -   The JSON string should represent the solution to the query, as specified in the query itself.
    -   **Important:** If no result is found that matches the query criteria, your program should print an **empty JSON object: {{}}**.
4.  Do not include any other print statements or extraneous output. Only the final JSON result should be printed.
5.  Use standard Python libraries only. Do not use libraries that require additional installation (e.g., pandas, numpy). Focus on `csv` module for CSV parsing if needed.

Your response should be only the Python code. Do not include any explanations, comments, or surrounding text.
```python
# Your code here
"""
    llm_program = get_llm_response(prompt)
    state["query_program"] = llm_program
    state["prompt_history"].append(prompt)

    return state


def ExecuteProgram(state: MyState) -> MyState:
    """
    This tool executes the program PQ and captures any runtime errors.
    It will also check and make sure that valid, non-empty JSON was generated.
    """
    print("** Entering ExecuteProgram Tool **")

    query_name = state["query_name"]
    query_program_code = state["query_program"]
    program_file_name = f"{query_name}.py"

    # 1. Write the program to a .py file
    try:
        with open(program_file_name, "w") as f:
            f.write(query_program_code)
    except IOError as e:
        error_message = f"Error writing query program to file '{program_file_name}': {e}"
        print(error_message)
        state["program_errors"].append(error_message)
        return state

    # 2. Execute the program and capture output and errors
    try:
        # Use subprocess.run to execute the Python script
        # capture_output=True captures stdout and stderr
        # text=True decodes stdout and stderr as text
        # timeout can be added to prevent infinite loops (e.g., timeout=60)
        process = subprocess.run(
            ["python", program_file_name],
            capture_output=True,
            text=True,
            check=False,  # Don't raise CalledProcessError for non-zero exit codes
            timeout=120   # Add a timeout to prevent endless execution
        )

        stdout = process.stdout.strip()
        stderr = process.stderr.strip()

        if process.returncode != 0:
            # Program exited with an error
            error_message = f"Program execution failed with exit code {process.returncode}.\n"
            if stderr:
                error_message += f"Stderr:\n{stderr}\n"
            if stdout: # Sometimes errors are printed to stdout before crashing
                error_message += f"Stdout before crash:\n{stdout}\n"
            print(error_message)
            state["program_errors"].append(error_message)
            return state

        if stderr:
            # Program executed but printed something to stderr (warnings, non-fatal errors)
            # We treat any stderr output as an error for this assignment's strictness
            error_message = f"Program produced stderr output:\n{stderr}\n"
            if stdout:
                error_message += f"Stdout (potential partial output):\n{stdout}\n"
            print(error_message)
            state["program_errors"].append(error_message)
            return state


        # 3. Check for valid, non-empty JSON output
        if not stdout:
            error_message = "Program executed successfully but produced no output."
            print(error_message)
            state["program_errors"].append(error_message)
            return state

        try:
            json_output = json.loads(stdout)
            if not json_output: # Check for empty JSON object {}
                error_message = "Program produced empty JSON output: {}."
                print(error_message)
                state["program_errors"].append(error_message)
                return state
            
            # If valid and non-empty, store the JSON output
            state["query_answer"] = stdout
            
        except json.JSONDecodeError as e:
            error_message = f"Program output is not valid JSON. Error: {e}\nOutput received:\n{stdout}"
            print(error_message)
            state["program_errors"].append(error_message)
            return state

    except subprocess.TimeoutExpired:
        process.kill() # Terminate the process if it timed out
        error_message = f"Program execution timed out after {process.timeout} seconds."
        print(error_message)
        state["program_errors"].append(error_message)
        return state
    except Exception as e:
        error_message = f"An unexpected error occurred during program execution: {e}"
        print(error_message)
        state["program_errors"].append(error_message)
        return state

    # If everything is successful, program_errors remains empty, and query_answer is populated
    return state


def Chk4rErr(state: MyState) -> MyState:
    """
    This tool checks if there were errors in the execution or in generating valid non-empty JSON.
    If yes, it will direct the program to the "ReflectOnErr" agent to execute next.
    Otherwise, it will direct the program to END or to a process that finalizes the output.
    """
    print("** Entering Chk4rErr Tool **")
    if state["attempts"] >= MAX_ATTEMPTS:
        state["chk_for_errors_map"] = "max_attempts_reached"
    elif len(state["program_errors"]) > state["attempts"]:
        print("Errors found in program execution. Directing to ReflectOnErr.")
        state["attempts"] = state["attempts"] + 1
        state["chk_for_errors_map"] = "reflect_on_error"
    else:
        print("No errors found. Program executed successfully with valid JSON output. Finalizing.")
        state["chk_for_errors_map"] = "finalize"

    return state


def ReflectOnErr(state: MyState) -> MyState:
    """
    This tool executes when the PQ had errors or did not output the appropriate JSON.
    In this case, it has the LLM reflect on the problem and generate a reflection summarizing the problem.
    This summary will be used in the tool "ReGenQueryPgm" to fix PQ.
    """
    print("** Entering ReflectOnErr Tool **")
    current_program = state["query_program"]
    program_errors = state["program_errors"][-1]
    prev_prompt = state["prompt_history"][-1]  # if state["prompt_history"] else ""
    # Formulate the prompt for the LLM to reflect on the error
    added_text = f"""
---
You previously tried to solve this query.
Here is the Python program that was generated:
---
```python
{current_program}
Here are the error messages and output from the program's execution:
{program_errors}
Your task is to reflect on the errors and determine what caused them. Provide a concise summary of the problem(s) and what needs to be changed in the code to fix them.
Focus on identifying the root cause of the error (e.g., incorrect file parsing, wrong column names, logical flaw, incorrect JSON formatting).
This reflection will be used to guide the regeneration of the program.

Provide only the reflection text, without any conversational filler or code blocks.
"""
    prompt = prev_prompt + added_text
    # Get the reflection from the LLM
    reflection_text = get_llm_response(prompt)

    # Update the state with the generated reflection
    state["reflection"] = reflection_text
    # state["prompt_history"].append(prompt)

    return state


def ReGenQueryPgm(state: MyState) -> MyState:
    """
    This tool uses previous prompts and the reflection (computed by agent ReflectOnErr)
    to prompt the LLM to fix the bug and regenerate the query program.
    """
    print("** Entering ReGenQueryPgm Tool **")
    current_program = state["query_program"]
    program_errors = state["program_errors"][-1]
    reflection = state["reflection"]
    prev_prompt = state["prompt_history"][-1]
    added_text = f"""
---
You previously tried to solve this query.
Here is the Python program that was generated:
---
```python
{current_program}
Here are the error messages and output from the program's execution:
{program_errors}
Here is a reflection on the errors and suggested fixes:
{reflection}

Based on the errors and the reflection, regenerate the complete and corrected Python program.
Your response should be only the Python code. Do not include any explanations, comments, or surrounding text.

```python

# Your corrected code here
"""
    prompt = prev_prompt + added_text
    corrected_program = get_llm_response(prompt)
    state["query_program"] = corrected_program
    # Append the new prompt to history.
    state["prompt_history"].append(prompt) 

    return state


def FinalizeOutput(state: MyState) -> MyState:
    """
    This tool finalizes the output by writing results to files and printing the answer.
    """
    print("** Entering FinalizeOutput Tool **")
    query_name = state["query_name"]
    query_program = state["query_program"]
    query_answer = state["query_answer"]
    program_errors = state["program_errors"] if len(state["program_errors"]) > 0 else [""]
    reflection = state["reflection"]

    # Write <query_name>.py (last generated program)
    try:
        with open(f"{query_name}.py", "w") as f:
            f.write(query_program)
    except IOError as e:
        print(f"Error writing {query_name}.py: {e}")

    # Write <query_name>_answer.txt (only if successful and answer exists)
    # The assignment states it will not exist if PQ cannot generate an answer.
    if query_answer and len(program_errors) <= MAX_ATTEMPTS: # Only write if successful and there's an answer
        try:
            with open(f"{query_name}_answer.txt", "w") as f:
                f.write(query_answer)
            print(f"Successfully generated program that computed solution. Solution in {query_name}_answer.txt")
        except IOError as e:
            print(f"Error writing {query_name}_answer.txt: {e}")
    else:
        # Ensure answer file is removed if it previously existed from a partial run or failed run
        if os.path.exists(f"{query_name}_answer.txt"):
            os.remove(f"{query_name}_answer.txt")


    # Write <query_name>_errors.txt
    try:
        with open(f"{query_name}_errors.txt", "w") as f:
            f.write(program_errors[-1]) # Will be empty if successful on first try
    except IOError as e:
        print(f"Error writing {query_name}_errors.txt: {e}")

    # Write <query_name>_reflect.txt
    try:
        with open(f"{query_name}_reflect.txt", "w") as f:
            f.write(reflection) # Will be empty if successful on first try
    except IOError as e:
        print(f"Error writing {query_name}_reflect.txt: {e}")

    # Print the query result to stdout
    print(f"Answer is {query_answer}")
    print("\nGraph execution complete.")

    return state


def main():    
    # Build the graph
    workflow = StateGraph(MyState)
    workflow.add_node("get_query_details", GetQueryDetails)
    workflow.add_node("gen_query_program", GenQueryProgram)
    workflow.add_node("execute_program", ExecuteProgram)
    workflow.add_node("chk_for_errors", Chk4rErr)
    workflow.add_node("reflect_on_error", ReflectOnErr)
    workflow.add_node("regen_query_program", ReGenQueryPgm)
    workflow.add_node("finalize_output", FinalizeOutput)

    # add edges
    workflow.add_edge(START, "get_query_details")
    workflow.add_edge("get_query_details", "gen_query_program")
    workflow.add_edge("gen_query_program", "execute_program")
    workflow.add_edge("execute_program", "chk_for_errors")
    workflow.add_conditional_edges(
        "chk_for_errors",
        lambda state: state["chk_for_errors_map"],
        {
            "reflect_on_error": "reflect_on_error",
            "finalize": "finalize_output",
            "max_attempts_reached": "finalize_output"
        }
    )
    workflow.add_edge("reflect_on_error", "regen_query_program")
    workflow.add_edge("regen_query_program", "execute_program")
    # Final successful path
    workflow.add_edge("finalize_output", END)

    # entry
    workflow.set_entry_point("get_query_details")

    runnable = workflow.compile()
    graphState = MyState()
    runnable.invoke(graphState)


if __name__ == "__main__":
    main()
