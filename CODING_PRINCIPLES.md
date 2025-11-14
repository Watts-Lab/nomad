# Coding Principles

1. **Avoid bloating the code at all costs.** This means no unnecessary variables, no defensive coding that nobody asked for, no checking of data types (unless we are purposefully debugging a problem). This bloats the code and is NOT equal to good coding.
2. **No blind coding or guessing function arguments with if else.** For instance "if 'building_id' in df" because you don't know the right column name. No. Go and check the codebase for how the objects are defined. Same with other libraries, go online and check the correct function signature.
3. **Absolutely no ad-hoc hot patches just to pass tests**, consider that the test might be outdated, or that there might be a deeper rework needed which DEFINITELY would need user input.
4. **No type hints in function signatures.** Instead, the parameter types are in the docstring.
5. **All imports go at the beginning of the file, never inside functions.**
6. **Double for loops are to be avoided at all costs.** As well as dictionaries when a pandas series or dataframe can do the job. We need to rely on the highly optimized and vectorized functions provided by numpy, pandas, and geopandas.
7. **Do not edit the code before sharing a plan with the user and explaining WHY it adheres to the coding principles.**

## Testing and Development Environment

### Virtual Environment
- This project uses a Python virtual environment located at `.venv/`
- Activate with: `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Unix)

### Running Tests
- Tests are located in `nomad/tests/`
- Run tests with pytest: `pytest nomad/tests/test_<module>.py -v`
- Run specific test: `pytest nomad/tests/test_<module>.py::test_function_name -v`
- The venv must be active or use direct path to pytest in venv

### Terminal/PowerShell Syntax
- This projects uses the default powershell the system provides via the terminal command tool. At least in this machine, it seems to be PSVersion 7.5.4 on a Windows 10 OS
- Example: `cd "C:\path\to\project"; pytest nomad\tests\test_file.py -v`
- Always quote paths with spaces
