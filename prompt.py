summary_prompt='''
## Data related code ##
{code}

As an experienced data scientist, please summarize the main aspects of the code provided. Your summary should include:

Key Steps: Describe the primary steps and processes carried out in the code.
Key Findings: Highlight the main results or outcomes derived from the code.
Your summary should be formal and clear.
'''.strip()

give_docstring_prompt='''
## ML model code ##
{code}
You are an experienced ML engineer. Analyze the provided code to identify the machine learning models used. Create a new Python file that retrieves the docstring of the ML model class. 
Return python code with NO other texts

Answer in the following format: ```python your_python_code ```
'''.strip()


list_hps_prompt='''
## Data preprocessing report ## 
{report}

## ML model docstring ##
{docstring}

## ML model ##
{model_code}

You are an experienced machine learning engineer. 
Using the data preprocessing report and the ML model's docstring, your task is to:
Identify ALL key hyperparameters of the ML model that should be optimized using Bayesian Optimization (BO).
Answer in the following format:
```json
{{
  "key_hyperparameters_l": list of key hyperparameters
}}
```
Think step by step and give your answer. 
'''.strip()


search_space_prompt='''
## Data preprocessing report ## 
{report}

## ML model docstring ##
{docstring}

## Key hyper-params list ##
{hps_l}

## Previous suggestions and performance ##
{history}

You are an experienced machine learning engineer. Your task is to provide search spaces for each hyperparameter in the {algo_name} algorithm. You already have history suggestions and the corresponding regret values, with the parameter bounds organized in descending order based on their regret value, where lower values indicate better performance. Analyze each hyperparameter to determine reasonable search spaces that optimize model performance, ensuring these search spaces are grounded in empirical evidence or established best practices. Your insights will be crucial for refining and optimizing the tuning process for {algo_name} models.
Here are some previously suggested ranges and their performance:
Using the data preprocessing report and the ML model's docstring, you are supposed to:
For each hyper-param in key hyper-param list, decide the type and define the search space.
For "int", "num", "pow" type hyper-params, you should suggest the lower_bound and upper_bound of search_space,
for "cat" type, you should suggest a list of categorical values, for bool type, the search_space is an empty list.
Please provide a new set of recommended lower and upper bounds for each hyperparameter, ensuring that these ranges are different from any previously suggested ranges in history. Additionally, ensure that the regret value associated with these new ranges is lower than any previously mentioned values. Do not write code.
Your output must follow this json format:
Answer in the following format:
```json
{{
  "suggestions":
    [
      {{
        "hp_name": name of the hyperparameter,
        "hp_type": choose from ["int", "num" (float), "bool", "pow" (varies in log space), "cat" (categorical value)],
        "search_space": [lower_bound, upper_bound] if hp_type is in ["int","num","pow"], else [categorical_list]
      }},
      // Add more hyperparameters as needed
    ]
}}
```
'''.strip()

meta_prompt = """
As an ML engineer, your task is to provide recommended lower and upper bounds for each hyperparameter in the {algo_name} algorithm. You already have reference data on some ranges and the corresponding regret value for these bounds, with the parameter bounds organized in descending order based on their regret value, where lower values indicate better performance. Analyze each hyperparameter to determine reasonable ranges that optimize model performance, ensuring these bounds are grounded in empirical evidence or established best practices. Your insights will be crucial for refining and optimizing the tuning process for {algo_name} models.
Here are some previously suggested ranges and their performance:
  """.strip()