## LLM-Powered Dynamic Query Answering Agent
This project implements a multi-node LangGraph agent designed to dynamically answer natural language queries based on provided CSV data files. 
The agent leverages a Large Language Model (LLM) to generate, execute, debug, and refine Python programs that process data and produce JSON results.

## Project Overview
The core idea is to create an intelligent system that can understand a user's query about specific data, write the necessary code to extract the answer, execute that code, and then self-correct if errors occur. 
This iterative refinement process allows the agent to be robust to initial LLM hallucinations or logical errors in the generated code.

## Features
- Natural Language Query Processing: Accepts user queries in plain English.
- Dynamic Code Generation: Utilizes an LLM to generate Python code based on the query and data file descriptions.
- Data File Handling: Reads and processes multiple CSV data files, inferring their structure from provided descriptions.
- Automated Execution & Error Detection: Executes the generated Python program and captures runtime errors, invalid output, or empty results.
- Iterative Debugging & Reflection: If errors occur, the LLM reflects on the problem, generates a summary, and uses this reflection to regenerate a corrected program.
- Limited Retries: The agent attempts to fix the program up to two additional times after the initial failure.
- Structured Output: Ensures the final answer is a valid, non-empty JSON structure.
- Comprehensive Output Files: Generates .py (program), _answer.txt (result), _errors.txt (last error), and _reflect.txt (last reflection) files for each query.

## How it Works (LangGraph Workflow)
The agent operates as a state machine using LangGraph, with the following key nodes (tools):
1. GetQueryDetails: Reads query_input.txt to extract the query name, the query itself, and details about the data files (name, description). Initializes the graph's state.
2. GenQueryProgram: Prompts the LLM (using the query and data descriptions) to generate the initial Python program (PQ) to answer the query.
3. ExecuteProgram: Writes PQ to a .py file, executes it, captures its standard output and error streams. It also validates if the output is valid, non-empty JSON.
4. Chk4Err: Checks the program_errors in the state.
- If errors exist and retry attempts are within limits (max 2 retries), it directs to ReflectOnErr.
- If errors exist and max retry attempts are exceeded, it terminates the graph.
- If no errors, it directs to FinalizeOutput.
5. ReflectOnErr: Prompts the LLM to analyze the program_errors and the query_program, generating a textual reflection summarizing the root cause of the problem.
6. ReGenQueryPgm: Uses the original prompt, the problematic query_program, the program_errors, and the reflection to prompt the LLM to generate a corrected version of the program. This leads back to ExecuteProgram for another attempt.
7. FinalizeOutput: Upon successful program execution, this tool writes the final program, answer, errors (empty if successful), and reflection (empty if successful) to their respective files and prints the final JSON answer to the console.
