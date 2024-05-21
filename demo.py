from tqdm import tqdm
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO
from hpobench.benchmarks.ml.xgboost_benchmark_old import XGBoostBenchmark as Benchmark
import time
from hpobench.util.openml_data_manager import get_openmlcc18_taskids
import asyncio
import prompt_utils
from dotenv import load_dotenv, find_dotenv
import openai
import numpy as np
import os
# os.environ['HTTP_PROXY'] = '127.0.0.1:7890'
# os.environ['HTTPS_PROXY'] = '127.0.0.1:7890'
import datetime
import functools
import json
import os
import re
import sys
import math
from banks import Prompt
from loguru import logger
from pathlib import Path
import json_repair
ROOT_PATH = str(Path(__file__).parent.resolve())
print('ROOT_PATH: ', ROOT_PATH)
sys.path.insert(0, ROOT_PATH)


_ = load_dotenv(find_dotenv())
_OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
_OPTIMIZER = "Meta-Llama-3-8B-Instruct"
_OPTIMIZER = "gpt-3.5-turbo"
# _OPTIMIZER='gpt-4o'
# ============== set optimization experiment configurations ================
num_points = 50  # number of points in linear regression
max_num_steps = 3  # the number of optimization steps
num_reps = 2  # the number of repeated runs
max_num_pairs = 15  # the maximum number of input-output pairs in meta-prompt
num_generated_points_in_each_step = 8
# num_output_decimals = 0

# fidelity=

# hebo_config=

# ================ load LLM settings ===================
optimizer_llm_name = _OPTIMIZER
# assert optimizer_llm_name in {
#     "gpt-3.5-turbo",
#     "gpt-4o",
# }

if optimizer_llm_name in {"gpt-3.5-turbo", "gpt-4o"}:
    # assert openai_api_key, "The OpenAI API key must be provided."
    openai.api_key = _OPENAI_API_KEY

# =================== create the result directory ==========================
datetime_str = (
    str(datetime.datetime.now().replace(microsecond=0))
    .replace(" ", "-")
    .replace(":", "-")
)
save_folder = os.path.join(
    ROOT_PATH,
    "outputs",
    "optimization-results",
    f"llm_hebo-o-{optimizer_llm_name}-{datetime_str}/",
)
os.makedirs(save_folder)
logger.add(save_folder+"log.log",
           format="{time} {level} {message}", level="DEBUG")
print(f"result directory:\n{save_folder}")


# ====================== optimizer model configs ============================
optimizer_gpt_max_decode_steps = 1024
optimizer_gpt_temperature = 1.0
optimizer_huggingface_max_decode_steps = 1024
optimizer_huggingface_temperature = 1.0
optimizer_gpt_max_decode_steps = 1024
optimizer_gpt_temperature = 1.0
optimizer_llm_dict = dict()
optimizer_llm_dict["max_decode_steps"] = optimizer_gpt_max_decode_steps
optimizer_llm_dict["temperature"] = optimizer_gpt_temperature
optimizer_llm_dict["batch_size"] = 1
call_optimizer_server_func = functools.partial(
    prompt_utils.call_openai_server_func,
    model=optimizer_llm_name,
    max_decode_steps=optimizer_gpt_max_decode_steps,
    temperature=optimizer_gpt_temperature,
)
# call_optimizer_server_func = functools.partial(
#     prompt_utils.call_huggingface_func,
#     model='/home/venido/LMs/LLMs/models/'+optimizer_llm_name,
#     max_decode_steps=optimizer_huggingface_max_decode_steps,
#     temperature=optimizer_huggingface_temperature,
# )
# ====================== try calling the servers ============================
print("\n======== testing the optimizer server ===========")
optimizer_test_output = asyncio.run(call_optimizer_server_func(
    "Does the sun rise from the north? Just answer yes or no.",
    temperature=1.0
))
print(f"optimizer test output: {optimizer_test_output}")
print("Finished testing the optimizer server.")
print("\n=================================================")
# ====================== benchmark settings ============================


task_ids = get_openmlcc18_taskids()
task_no, task_id = 0, task_ids[0]
other_info = {}

# ================= generate the starting points =====================
# init range

for task_no, task_id in enumerate(task_ids[:1]):
    print(
        f'#################### TASK {task_no + 1} of {len(task_ids)}: Task-Id: {task_id} ################### #')
    benchmark = Benchmark(task_id=task_id)
    if benchmark:
        start = time.time()
        cs = benchmark.get_configuration_space()
        results = []
        default_bounds = []
        print("Hyperparameter default bounds:")
        for hyperparameter in list(cs.values()):
            name = hyperparameter.name
            lower = hyperparameter.lower
            upper = hyperparameter.upper
            log = hyperparameter.log
            check_int = True if "check_int" in dir(hyperparameter) else False
            if log:
                lower = (math.log(lower, 2))
                upper = (math.log(upper, 2))
            default_bounds.append((lower, upper))
            other_info[name] = [log, check_int]
            print(
                f"{name}: Lower = {lower}, Upper = {upper}, Check_int = {check_int}, Log_sample = {log}")
# add categorical values
# ====================== utility functions ============================


def evaluate_loss(benchmark, space_list, fidelity):
    """
    Evaluate the loss for a given benchmark and design space.

    Args:
        benchmark: The benchmark to evaluate.
        space_list: The design space to evaluate.
        fidelity: The fidelity parameters for the evaluation.

    Returns:
        A dictionary containing the best configuration, fidelity, test loss, valid loss, and train loss.
    """
    def preprocess_func(df):
        return_dict = {}
        for key in df.columns:
            value = df[key].iloc[0]
            if isinstance(df[key].iloc[0], np.int64):
                value = int(df[key].iloc[0])
            elif isinstance(df[key].iloc[0], np.int64):
                value = float(df[key].iloc[0])
            return_dict[key.split('_log')[0]
                        ] = 2.0**(value) if '_log' in key else value
        return return_dict

    def objective(df):
        config = preprocess_func(df)
        result_dict = benchmark.objective_function(config, fidelity=fidelity)
        return result_dict
    logger.info(space_list)
    sp = DesignSpace().parse(space_list)
    opt = HEBO(sp, model_name='gp', rand_sample=4)

    for _ in tqdm(range(8)):
        # print('#####',opt.y)
        try:
            rec = opt.suggest(n_suggestions=1)
            result_dict = objective(rec)
            valid_loss = result_dict['function_value']
            train_loss = result_dict['info']['train_loss']
            y = np.array([-valid_loss], dtype=np.float64)
            opt.observe(rec, y)
        except Exception as e:
            logger.debug(e)
            continue
        # logger.info('After %d iterations, best obj is %.2f' % (i, -opt.best_y))
    try:
        best_config = preprocess_func(opt.best_x)
        result_dict_test = benchmark.objective_function_test(best_config)
        test_loss = result_dict_test['function_value']
        logger.info(f"Best params: {best_config}, test loss: {test_loss}")
        return {
            'configuration': best_config,
            'fidelity': fidelity,
            'test_loss': -np.round(test_loss, 3),
            'valid_loss': valid_loss,
            'train_loss': train_loss
        }
    except Exception as e:
        logger.debug(e)
        return {}


def gen_meta_prompt(
    bound_info,
    test_loss,
    old_value_pairs_set,
    max_num_pairs=100,
):
    """
    Generate a meta-prompt for optimization.

    Args:
        bound_info: Dictionary containing the bounds information.
        test_loss: The test loss value.
        old_value_pairs_set: Set of old (bounds,test_loss) pairs.
        max_num_pairs: Maximum number of exemplars in the meta-prompt.

    Returns:
        The generated meta-prompt as a string.
    """
    bound_names = bound_info.keys()
    info = tuple((('bound_range', v['bound_range']),
                  ('is_log_sample', v['is_log_sample']),
                  ('is_int', v['is_int']))
                 for v in bound_info.values())
    old_value_pairs_set.add((info, test_loss))

    old_value_pairs = list(old_value_pairs_set)
    old_value_pairs = sorted(old_value_pairs, key=lambda x: -x[1])[
        -max_num_pairs:
    ]

    # print(old_value_pairs)
    old_value_pairs_substr = ""
    for i, pair in enumerate(old_value_pairs):
        old_value_pairs_substr += f"\nSuggestion {i}: "
        # print('pair: ',pair)
        infos, test_loss = pair
        for name, info in zip(bound_names, infos):
            old_value_pairs_substr += f"{name} : "
            old_value_pairs_substr += '( '+', '.join(
                [f'{key}: {value}' for key, value in info])+'), '
        old_value_pairs_substr += f' test loss: {test_loss}'

    meta_prompt = """
  As an ML engineer, your task is to provide recommended lower and upper bounds for each hyperparameter in the {algo.name} algorithm. You already have reference data on some ranges and the corresponding test loss for these bounds, with the parameter bounds organized in descending order based on their test loss, where lower values indicate better performance. Analyze each hyperparameter to determine reasonable ranges that optimize model performance, ensuring these bounds are grounded in empirical evidence or established best practices. Your insights will be crucial for refining and optimizing the tuning process for {algo.name} models.
Here are some previously suggested ranges and their performance:
  """.strip()
    meta_prompt += "\n\n"
    meta_prompt += old_value_pairs_substr.strip()
    meta_prompt += "\n\n"
    meta_prompt += """
  Please provide a new set of recommended lower and upper bounds for each hyperparameter, ensuring that these ranges are different from any previously suggested ranges. Additionally, ensure that the test loss value associated with these new ranges is lower than any previously mentioned values. Do not write code. 
Your output must follow this json format:
  """.strip()
    prompt_template = '''
{
  {% for hyper_param in hyper_params_l %}
  "{{ hyper_param }}": {
      "lower_bound": "your lower_bound here",
      "upper_bound": "your upper_bound here"
  },

  {% endfor %}
}
  '''
    p = Prompt(prompt_template)
    meta_prompt += p.text({"hyper_params_l": list(bound_info.keys())})
    meta_prompt += '''
where lower_bound and upper_bound are all numerical values. 

Answer:
```json
  '''
    return meta_prompt


def extract_string(input_string):
    """
    Extract the string from the input_string.

    Args:
        input_string: The input string.

    Returns:
        The extracted string.
    """
    raw_result = input_string.split('```')[0]
    return raw_result


def parse_output(extracted_output):
    """
    Parse the json from extracted string output.

    Args:
        extracted_output: The extracted string output.

    Returns:
        Parsed output as a list of bounds.
    """
    if not extracted_output:
        return
    bounds = []
    try:
        bounds_dict = eval(extracted_output)
    except:
        good_json_string = json_repair.repair_json(
            extracted_output, skip_json_loads=True)
        bounds_dict = json.loads(good_json_string)
    for param_name, b_range in bounds_dict.items():
        if 'lower_bound' in b_range and 'upper_bound' in b_range:
            lower_bound = eval(b_range['lower_bound']) if isinstance(
                b_range['lower_bound'], str) else b_range['lower_bound']
            upper_bound = eval(b_range['upper_bound']) if isinstance(
                b_range['upper_bound'], str) else b_range['upper_bound']
            bounds.append((lower_bound, upper_bound))
        else:
            bounds = []
            break
    return bounds


def process_output(bounds, other_info):
    """
    Process the output bounds.

    Args:
        bounds: List of bounds.
        other_info: Dictionary containing additional information for each hyperparameter.

    Returns:
        Tuple containing bound information and space list.
    """
    space_list = []
    bound_info = {}
    for hyper_param_info, bound in zip(other_info.items(), bounds):
        param_name, param_info = hyper_param_info
        if param_info[0]:  # is_log_sample
            space_dict = {
                'name': param_name+'_log',
                'type': 'int',
                'lb': bound[0],
                'ub': bound[1],
            }
            bound_info[param_name] = {
                'bound_range': (bound[0], bound[1]), "is_log_sample": param_info[0], "is_int": param_info[1]
            }
        else:
            if param_info[1]:  # is_int
                space_dict = {
                    'name': param_name,
                    'type': 'int',
                    'lb': int(bound[0]),
                    'ub': int(bound[1])}
                bound_info[param_name] = {
                    'bound_range': (int(bound[0]), int(bound[1])), "is_log_sample": param_info[0], "is_int": param_info[1]
                }

            else:
                space_dict = {
                    'name': param_name,
                    'type': 'num',
                    'lb': bound[0],
                    'ub': bound[1]}
                bound_info[param_name] = {
                    'bound_range': (bound[0], bound[1]), "is_log_sample": param_info[0], "is_int": param_info[1]
                }

        space_list.append(space_dict)
    return bound_info, space_list


configs_dict = dict()
results_dict = dict()
num_convergence_steps = []

for i_rep in range(num_reps):
    found_optimal = False
    print(f"\nRep {i_rep}:")
    # ================= generate the starting points =====================
    # init range
    init_bounds = default_bounds
    init_fidelity = {'n_estimators': 8, 'dataset_fraction': 0.4}

    # ====================== run optimization ============================
    configs_dict_single_rep = {
        "optimizer_llm_configs": optimizer_llm_dict,
        "init_bounds": init_bounds,
        "max_num_steps": max_num_steps,
        "max_num_pairs": max_num_pairs,
        "num_generated_points_in_each_step": num_generated_points_in_each_step,
    }
    configs_dict[i_rep] = configs_dict_single_rep
    configs_json_path = os.path.join(save_folder, "configs.json")
    print(f"saving configs to\n{configs_json_path}")
    with open(configs_json_path, "w") as f:
        json.dump(configs_dict, f, indent=4)

    old_value_pairs_set = set()
    # format: [([(lower_bound,upper_bound,is_log),...],valid_loss,i_step)]
    old_value_pairs_with_i_step = []
    meta_prompts_dict = dict()  # format: {i_step: meta_prompt}
    raw_outputs_dict = dict()  # format: {i_step: raw_outputs}
    init_space_list = []
    bound_info, init_space_list = process_output(init_bounds, other_info)
    init_test_loss = evaluate_loss(
        benchmark, init_space_list, init_fidelity)['test_loss']
    bound_range = tuple([v['bound_range']for v in bound_info.values()])
    old_value_pairs_with_i_step.append((bound_range, init_test_loss, -1))

    print("\n================ run optimization ==============")

    results_json_path = os.path.join(save_folder, "results.json")
    print(f"saving results to\n{results_json_path}")
    test_loss = init_test_loss
    for i_step in range(max_num_steps):
        print(f"\nStep {i_step}:")
        meta_prompt = gen_meta_prompt(
            bound_info,
            test_loss,
            old_value_pairs_set,
            max_num_pairs=max_num_pairs,
        )
    # print(meta_prompt)
        if not i_step % 5:
            print("\n=================================================")
            # print(f"meta_prompt:\n{meta_prompt}")
        meta_prompts_dict[i_step] = meta_prompt
# print(meta_prompt)

        # generate a maximum of the given number of points in each step
        remaining_num_points_to_generate = num_generated_points_in_each_step
        raw_outputs = []
        while remaining_num_points_to_generate > 0:
            raw_outputs += asyncio.run(call_optimizer_server_func(meta_prompt))
            remaining_num_points_to_generate -= optimizer_llm_dict["batch_size"]
        raw_outputs = raw_outputs[:num_generated_points_in_each_step]
        raw_outputs_dict[i_step] = raw_outputs
        parsed_outputs = []
        for string in raw_outputs:
            try:
                parsed_output = parse_output(
                    extract_string(string)
                )
                if parsed_output:
                    parsed_outputs.append(parsed_output)
            except Exception as e:
                print(e, string)
        parsed_outputs = [tuple(item) for item in parsed_outputs]
        print(f"proposed points: {parsed_outputs}")

        single_step_values = []
        for parsed_bounds in parsed_outputs:
            bound_info, space_list = process_output(parsed_bounds, other_info)
            bound_range = tuple([v['bound_range']for v in bound_info.values()])
            loss = evaluate_loss(benchmark, space_list, init_fidelity)
            if 'test_loss' in loss:
                test_loss = loss['test_loss']
                single_step_values.append(test_loss)
                bound_names = bound_info.keys()
                info = tuple((('bound_range', v['bound_range']),
                             ('is_log_sample', v['is_log_sample']),
                              ('is_int', v['is_int']))
                             for v in bound_info.values())
                old_value_pairs_set.add((info, test_loss))
                old_value_pairs_with_i_step.append(
                    (bound_range, test_loss, i_step))
        logger.info(f"single_step_values: {single_step_values}")

        # ====================== save results ============================
        results_dict_single_rep = {
            "meta_prompts": meta_prompts_dict,
            "raw_outputs": raw_outputs_dict,
            "old_value_pairs_with_i_step": old_value_pairs_with_i_step,
        }
        results_dict[i_rep] = results_dict_single_rep
        with open(results_json_path, "w") as f:
            json.dump(results_dict, f, indent=4)
