import matplotlib.pyplot as plt
import warnings
import torch
import json_repair
from pathlib import Path
from loguru import logger
from banks import Prompt
import math
import sys
import re
import json
import functools
import datetime
from tqdm import tqdm
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO
from hebo.optimizers.bo import BO
import ConfigSpace
from random_search import RandomSearch
from hpobench.benchmarks.ml.xgboost_benchmark_old import XGBoostBenchmark
import time
from hpobench.util.openml_data_manager import get_openmlcc18_taskids
import asyncio
import prompt_utils
from dotenv import load_dotenv, find_dotenv
import openai
import numpy as np
import torch
import pandas as pd
import random

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)  # If using GPU
random.seed(42)
import os

os.environ['HTTP_PROXY'] = '127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = '127.0.0.1:7890'
torch.set_default_tensor_type(torch.DoubleTensor)
warnings.filterwarnings("ignore")
ROOT_PATH = str(Path(__file__).parent.resolve())
print('ROOT_PATH: ', ROOT_PATH)
sys.path.insert(0, ROOT_PATH)


_ = load_dotenv(find_dotenv())
_OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
_OPTIMIZER = "Meta-Llama-3-8B-Instruct"
_OPTIMIZER = "gpt-3.5-turbo"
# _OPTIMIZER='gpt-4o'
# ============== set optimization experiment configurations ================
max_num_steps = 8  # the number of optimization steps
num_reps = 1  # the number of repeated runs
max_num_pairs = 15  # the maximum number of input-output pairs in meta-prompt
num_generated_points_in_each_step = 1
num_output_decimals = 3


# ================ load LLM settings ===================
optimizer_llm_name = _OPTIMIZER

if optimizer_llm_name in {"gpt-3.5-turbo", "gpt-4o"}:
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
optimizer_llm_dict["batch_size"] = num_generated_points_in_each_step


# ====================== benchmark settings ============================

task_ids = get_openmlcc18_taskids()
task_no, task_id = 0, task_ids[0]
other_info = {}

# ================= generate the starting points =====================
# init range

for task_no, task_id in enumerate(task_ids[:1]):
    print(
        f'#################### TASK {task_no + 1} of {len(task_ids)}: Task-Id: {task_id} ####################')
    benchmark = XGBoostBenchmark(task_id=task_id,rng=10)
    if benchmark:
        start = time.time()
        cs = benchmark.get_configuration_space(seed=10)
        results = []
        default_bounds = []
        algo_name = type(benchmark).__name__
        logger.info(f"Algo: {algo_name}")
        logger.info("Hyperparameter default bounds: \n")
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
default_fidelity = {'n_estimators': 8, 'dataset_fraction': 0.4}


# ====================== utility functions ============================


def evaluate_loss(benchmark, space_list, fidelity, result_l=[], hebo_config={},past_X=None,past_y=None,bound_range=None):
    """
    Evaluate the loss for a given benchmark and design space.

    Args:
        benchmark: The benchmark to evaluate.
        space_list: The design space to evaluate.
        fidelity: The fidelity parameters for the evaluation.

    Returns:
        A dictionary containing the best configuration, fidelity, test loss, valid loss, and train loss.
    """
    raw_result_l=result_l[:]
    def preprocess_func(df):
        return_dict = {}
        for key in df.columns:
            value = np.round(df[key].iloc[0], 5)
            if isinstance(df[key].iloc[0], np.int64):
                value = int(df[key].iloc[0])
            elif isinstance(df[key].iloc[0], np.int64):
                value = float(df[key].iloc[0])
            return_dict[key.split('_log')[0]
                        ] = 2.0**(value) if '_log' in key else value
        return return_dict

    def objective(df):
        config = preprocess_func(df)
        result_dict = benchmark.objective_function(configuration=config, fidelity=fidelity,rng=10)
        return result_dict
    # Function to check if a value is within the legal range
    def in_legal_range(value, legal_range):
        return legal_range[0] <= value <= legal_range[1]
    sp = DesignSpace().parse(space_list)

    # random search

    opt = eval(hebo_config['optimizer'])(
        sp, model_name='gp', rand_sample=hebo_config['rand_sample'])
    # opt=BO(sp,model_name='gp', rand_sample=hebo_config['rand_sample'])
    # opt = HEBO(sp, model_name='gp', rand_sample=hebo_config['rand_sample'])
    if 'scramble_seed' in hebo_config:opt.scramble_seed=hebo_config['scramble_seed']
    if past_X is not None:
        legal_ranges = bound_range
        # Get the indices of rows where all values are within their legal ranges
        legal_row=past_X[past_X.apply(lambda row: all(in_legal_range(row[col], legal_ranges[i]) for i,col in enumerate(past_X.columns)), axis=1)]
        legal_row_indices = legal_row.index
        # logger.info('legal_row: ',legal_row)
        logger.info(f'legal_row_indices: {legal_row_indices.tolist()}')
        if legal_row_indices.tolist():
            opt.X=legal_row
            opt.y=past_y[legal_row_indices]
        else:
            opt.X=past_X
            opt.y=past_y
        # logger.info(f'X: {opt.X}, y: {opt.y}')
    logger.info(f"{hebo_config['optimizer']} searching... \n")
    for i in range(hebo_config['hebo_iteration']):
        # print('#####',opt.y)
        try:
            rec = opt.suggest(n_suggestions=hebo_config['n_suggestions'],fix_input=None)
            result_dict = objective(rec)
            valid_loss = result_dict['function_value']
            train_loss = result_dict['info']['train_loss']
            y = np.array([valid_loss], dtype=np.float64).reshape(-1, 1)
            opt.observe(rec, y)
            best_config = preprocess_func(opt.best_x)
            result_dict_test = benchmark.objective_function_test(configuration=best_config,rng=10)
            test_loss = result_dict_test['function_value']
            result_l.append(test_loss)
        except Exception as e:
            logger.debug(e)
            result_l=raw_result_l[:]
            test_loss=1.0
            break

    best_config = preprocess_func(opt.best_x)
    result_dict_test = benchmark.objective_function_test(configuration=best_config,rng=10)

    return {
        'configuration': best_config,
        'fidelity': fidelity,
        'test_loss': np.round(test_loss, 5),
        # 'valid_loss': valid_loss,
        # 'train_loss': train_loss,
        'best_params': best_config,
        'hebo_seq': result_l,
        'past_X':opt.X,
        'past_y':opt.y

    }

def gen_meta_prompt(
    algo_name,
    bound_info,
    best_params,
    regret_value,
    old_value_pairs_set,
    max_num_pairs=100,
):
    """
    Generate a meta-prompt for optimization.

    Args:
        bound_info: Dictionary containing the bounds information.
        regret_value: The regret value.
        old_value_pairs_set: Set of old (bounds,regret_value) pairs.
        max_num_pairs: Maximum number of exemplars in the meta-prompt.

    Returns:
        The generated meta-prompt as a string.
    """
    bound_names = bound_info.keys()
    logger.info(f'old_value_pairs_set len: {len(old_value_pairs_set)}')

    old_value_pairs = list(old_value_pairs_set)
    old_value_pairs = sorted(old_value_pairs, key=lambda x: -x[-1])[
        -max_num_pairs:
    ]

    # print(old_value_pairs)
    old_value_pairs_substr = ""
    for i, pair in enumerate(old_value_pairs):
        old_value_pairs_substr += f"\nSuggestion search space {i}: "
        # print('pair: ',pair)
        infos, best_params, regret_value = pair
        for name, info in zip(bound_names, infos):
            old_value_pairs_substr += f"{name} : "
            old_value_pairs_substr += '( '+', '.join(
                [f'{key}: {value}' for key, value in info])+'), '
        old_value_pairs_substr += f' best_params in this search space: {(best_params)},'
        old_value_pairs_substr += f' regret_value: {regret_value}'

    meta_prompt = """
  As an ML engineer, your task is to provide recommended lower and upper bounds for each hyperparameter in the {algo_name} algorithm. You already have reference data on some ranges and the corresponding regret value for these bounds, with the parameter bounds organized in descending order based on their regret value, where lower values indicate better performance. Analyze each hyperparameter to determine reasonable ranges that optimize model performance, ensuring these bounds are grounded in empirical evidence or established best practices. Your insights will be crucial for refining and optimizing the tuning process for {algo_name} models.
Here are some previously suggested ranges and their performance:
  """.strip()
    meta_prompt = meta_prompt.format(algo_name=algo_name)
    meta_prompt += "\n\n"
    meta_prompt += old_value_pairs_substr.strip()
    meta_prompt += "\n\n"
    meta_prompt += """
  Please provide a new set of recommended lower and upper bounds for each hyperparameter, ensuring that these ranges are different from any previously suggested ranges. Additionally, ensure that the regret value associated with these new ranges is lower than any previously mentioned values. Do not write code.
Your output must follow this json format:
  """.strip()
    prompt_template = '''
{
  {% for i in range(hyper_params_l | length) %}
  "{{ hyper_params_l[i] }}": {
      "lower_bound": "your lower_bound here, lower_bound must be set between {{ default_bounds[i] }}",
      "upper_bound": "your upper_bound here, upper_bound must be set between (lower_bound, {{default_bounds[i][1]}})"
  }{% if not loop.last %},{% endif %}

  {% endfor %}
}
  '''
    p = Prompt(prompt_template)
    meta_prompt += p.text({"hyper_params_l": list(bound_info.keys()),
                          "default_bounds": default_bounds})
    meta_prompt += '''
where lower_bound and upper_bound are all numerical values.

Answer:
```json
  '''
    return meta_prompt, old_value_pairs_set


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


results_plot_path = os.path.join(save_folder, "result.png")


def plot_regret(seqs, labels, ideal_point, plot_path=results_plot_path):
    plt.figure(figsize=(8, 6))
    for seq, label in zip(seqs, labels):
        if seq.shape[1] ==1: 
            plt.semilogy(seq - ideal_point, 'x-', label=label)
        else: 
            mean_regret = np.mean(seq - ideal_point, axis=0)
            std_regret = np.std(seq - ideal_point, axis=0)
            plt.semilogy(mean_regret, 'x-', label=label,)  # Mean regret line
            plt.fill_between(range(len(mean_regret)), 
                                mean_regret - std_regret, 
                                mean_regret + std_regret, 
                                alpha=0.3, )  # Filled area with slightly lower transparency
    plt.xlabel('Evaluation')
    plt.ylabel('Regret')
    plt.legend()
    plt.savefig(plot_path)


async def run_tasks(hebo_config={}, call_optimizer_server_func=None):
    # ====================== try calling the servers ============================
    print("\n======== testing the optimizer server ===========")
    optimizer_test_output = await call_optimizer_server_func(
        "Does the sun rise from the north? Just answer yes or no.",
        temperature=1.0
    )
    print(f"optimizer test output: {optimizer_test_output}")
    print("Finished testing the optimizer server.")
    print("\n=================================================")
    configs_dict = dict()
    results_dict = dict()
    old_value_pairs_set = set()

    for i_rep in range(num_reps):
        # found_optimal = False
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
        init_loss = evaluate_loss(
            benchmark, init_space_list, init_fidelity, [], hebo_config=hebo_config)
        init_test_loss = init_loss['test_loss']
        init_regret_value = np.round(init_test_loss, num_output_decimals)
        init_best_params = init_loss['best_params']
        result_l = init_loss['hebo_seq']
        past_X=init_loss['past_X']
        past_y=init_loss['past_y']
        init_best_params = {k: np.round(v, 4)
                            for k, v in init_best_params.items()}
        bound_range = tuple([v['bound_range']for v in bound_info.values()])
        old_value_pairs_with_i_step.append(
            (bound_range, init_regret_value, -1))

        print("\n================ run optimization ==============")
        # result_l = []


        results_json_path = os.path.join(save_folder, "results.json")
        print(f"saving results to\n{results_json_path}")
        test_loss = init_test_loss
        regret_value = init_regret_value
        best_params = init_best_params
        info = tuple((('bound_range', v['bound_range']),
                        ('is_log_sample', v['is_log_sample']),
                        ('is_int', v['is_int'])
                      )
                     for v in bound_info.values())
        old_value_pairs_set.add(
            (info, str(init_best_params), init_regret_value))

        # print(f"\nStep {i_step}:")
        i_step=0
        with tqdm(total=max_num_steps) as pbar:
            while i_step < max_num_steps:
                # print(f"\nStep {i_step}:")
                meta_prompt, old_value_pairs_set = gen_meta_prompt(
                    algo_name,
                    bound_info,
                    best_params,
                    regret_value,
                    old_value_pairs_set,
                    max_num_pairs=max_num_pairs,
                )

                meta_prompts_dict[i_step] = meta_prompt

                raw_outputs = []
                raw_outputs += await call_optimizer_server_func(
                    [meta_prompt]*num_generated_points_in_each_step)
                raw_outputs = raw_outputs[:num_generated_points_in_each_step]
                raw_outputs_dict[i_step] = raw_outputs
                parsed_outputs = []

                # logger.info(f'Raw output length: {len(raw_outputs)}')
                for string in raw_outputs:
                    try:
                        parsed_output = parse_output(
                            extract_string(string)
                        )
                        if parsed_output:
                            parsed_outputs.append(parsed_output)
                    except Exception as e:
                        logger.debug(e, string)
                parsed_outputs = [tuple(item) for item in parsed_outputs]

                single_step_values = []
                bound_info, space_list = process_output(
                    parsed_outputs[0], other_info)
                bound_range = tuple([v['bound_range']
                                    for v in bound_info.values()])
                logger.info(bound_range)
                loss = evaluate_loss(
                    benchmark, space_list, init_fidelity, result_l=result_l, hebo_config=hebo_config,past_X=past_X,past_y=past_y,bound_range=bound_range)
                if 'test_loss' in loss and loss['test_loss']<1.0:
                    test_loss = loss['test_loss']
                    regret_value = np.round(test_loss, num_output_decimals)
                    best_params = loss['best_params']
                    result_l = loss['hebo_seq']
                    past_X=loss['past_X']
                    past_y=loss['past_y']
                    best_params = {k: np.round(v, 4)
                                    for k, v in best_params.items()}
                    single_step_values.append(regret_value)
                    info = tuple((('bound_range', v['bound_range']),
                                     ('is_log_sample', v['is_log_sample']),
                                      ('is_int', v['is_int']),
                                    )
                                    for v in bound_info.values())
                    old_value_pairs_set.add(
                        (info, (str(best_params)), regret_value))
                    old_value_pairs_with_i_step.append(
                        (bound_range, regret_value, i_step))
                    i_step+=1
                    pbar.update(1)
                else: 
                    result_l=result_l[:(i_step+1)*hebo_config['hebo_iteration']]
                    continue

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
    return result_l
def main(hebo_config, call_optimizer_server_func):
    return asyncio.run(run_tasks(hebo_config, call_optimizer_server_func))

init_space_list = []
bound_info, init_space_list = process_output(default_bounds, other_info)
call_optimizer_server_func = functools.partial(
    prompt_utils.call_openai_server_func,
    model='gpt-3.5-turbo',
    max_decode_steps=optimizer_gpt_max_decode_steps,
    temperature=optimizer_gpt_temperature,
)


hebo_config = {
    'optimizer': 'HEBO',
    "rand_sample": 4,
    "hebo_iteration": 36,
    "n_suggestions": 1,
    'scramble_seed': 42
}

conv_llm35_bo_seq_l=[]
# run LLM+HEBO with 6 repeated runs
for _ in range(6):
    llm35_bo_seq = np.array(main(hebo_config,call_optimizer_server_func)).reshape(-1, 1)
    conv_llm35_bo_seq=np.minimum.accumulate(llm35_bo_seq)
    conv_llm35_bo_seq_l.append(conv_llm35_bo_seq)

conv_bo_seq_l=[]
conv_hebo_seq_l=[]
conv_rds_seq_l=[]

# run BO with 6 random seeds

for _ in range(6):
    bo_result = evaluate_loss(
        benchmark, init_space_list, default_fidelity, [],
        hebo_config={
            "optimizer": "BO",
            "rand_sample": hebo_config['rand_sample'],
            "hebo_iteration": (max_num_steps+1)*hebo_config['hebo_iteration'],
            "n_suggestions": 1,

        })
    bo_seq = np.array(bo_result['hebo_seq']).reshape(-1, 1)
    conv_bo_seq = np.minimum.accumulate(bo_seq)
    conv_bo_seq_l.append(conv_bo_seq)

# run HEBO with 6 random seeds

for i in range(6):
    hebo_result = evaluate_loss(
    benchmark, init_space_list, default_fidelity, [],
    hebo_config={
        "optimizer": "HEBO",
        "rand_sample": hebo_config['rand_sample'],
        "hebo_iteration": (max_num_steps+1)*hebo_config['hebo_iteration'],
        "n_suggestions": 1,
        'scramble_seed':i+42


    })
    hebo_seq = np.array(hebo_result['hebo_seq']).reshape(-1, 1)
    conv_hebo_seq = np.minimum.accumulate(hebo_seq)
    conv_hebo_seq_l.append(conv_hebo_seq)

# run RandomSearch with 6 random seeds

for i in range(6):
    rds_result= evaluate_loss(
    benchmark, init_space_list, default_fidelity, [],
    hebo_config={
        "optimizer": "RandomSearch",
        "rand_sample": hebo_config['rand_sample'],
        "hebo_iteration": (max_num_steps+1)*hebo_config['hebo_iteration'],
        "n_suggestions": 1,
        'scramble_seed':i+42


    })
    rds_seq=np.array(rds_result['hebo_seq']).reshape(-1, 1)
    conv_rds_seq=np.minimum.accumulate(rds_seq)
    conv_rds_seq_l.append(conv_rds_seq)



plot_regret([
    np.array(conv_rds_seq_l).reshape(len(conv_rds_seq_l),-1),
    np.array(conv_bo_seq_l).reshape(len(conv_bo_seq_l),-1),
    np.array(conv_hebo_seq_l).reshape(len(conv_hebo_seq_l),-1),
    np.array(conv_llm35_bo_seq_l).reshape(len(conv_llm35_bo_seq_l),-1),
    ],
    [
    f'Random-{(max_num_steps+1)*hebo_config["hebo_iteration"]}iters',
    f'BO-{(max_num_steps+1)*hebo_config["hebo_iteration"]}iters',
    f'HEBO-{(max_num_steps+1)*hebo_config["hebo_iteration"]}iters',
    f'gpt3.5-HEBO-{max_num_steps+1}steps',
    #  f'gpt4o-HEBO-{num_reps}reps-{max_num_steps}steps-{num_generated_points_in_each_step}pts'
     ], 0.0)
