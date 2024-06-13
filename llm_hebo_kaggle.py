import matplotlib.pyplot as plt
import warnings
import torch
import contextlib
import ast
from pathlib import Path
ROOT_PATH= Path(__file__).parent.resolve()

from loguru import logger
from banks import Prompt
import math
import sys
import re
import json
import functools
from tqdm import tqdm
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO
from hebo.optimizers.bo import BO
from random_search import RandomSearch
import datetime
from prompt import (summary_prompt,list_hps_prompt,search_space_prompt,
                    give_docstring_prompt,meta_prompt)
import prompt_utils
import numpy as np
import torch
import pandas as pd
import random
import os
import subprocess
import tiktoken
RANDOM_SEED=42


np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)  # If using GPU
random.seed(RANDOM_SEED)

project_path=ROOT_PATH / 'kaggle'/ 'titanic'
target_column='Survived'
script_path=project_path / 'script.py'
train_file=project_path / 'train.csv'
test_file=project_path / 'test.csv'

# ============== set optimization experiment configurations ================
max_num_steps = 4  # the number of optimization steps
num_reps = 1  # the number of repeated runs
max_num_pairs = 15  # the maximum number of input-output pairs in meta-prompt
num_generated_points_in_each_step = 1
num_output_decimals = 5


# ================ load LLM settings ===================
optimizer_gpt_max_decode_steps = 4096
optimizer_gpt_temperature = 0.1
optimizer_gpt_topp=1.0
optimizer_gpt_model='gpt-3.5-turbo'
_OPTIMIZER=optimizer_gpt_model
optimizer_llm_name = _OPTIMIZER

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


from sklearn.ensemble import RandomForestClassifier
ml_model=RandomForestClassifier
algo_name=ml_model.__name__
logger.info(f'{algo_name=}')

optimizer_llm_dict = dict()
optimizer_llm_dict["max_decode_steps"] = optimizer_gpt_max_decode_steps
optimizer_llm_dict["temperature"] = optimizer_gpt_temperature
optimizer_llm_dict["top_p"] = optimizer_gpt_topp

def parse_json(rsp):
    rsp=rsp.replace('null','None')
    if isinstance(rsp,str):
        json_blocks = (
            re.findall(r"```json(.*?)```", rsp, re.DOTALL) if "```json" in rsp else [rsp]
        )

        json_blocks=[v.strip() for v in json_blocks]
        # logger.info(f'{json_blocks=}')
        try: 
            result=[eval(ele) for ele in json_blocks]
        except Exception as e:
            result=rsp
            logger.debug(e)
    else: result=rsp
    return result


def parse_code(rsp):
    for pattern in (r"(.*?```python.*?\s+)?(?P<code>.*)(```.*?)", r"(.*?```python.*?\s+)?(?P<code>.*)"):
        match = re.search(pattern, rsp, re.DOTALL)
        if not match:
            continue
        code = match.group("code")
        if not code:
            continue
        with contextlib.suppress(Exception):
            ast.parse(code)
            return code
    raise ValueError("Invalid python code")

def run_code(code):
    run_result=''
    try:
        result = subprocess.run(["python3", "-c", code],check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        run_result = result.stdout
    except subprocess.CalledProcessError as e:
        run_result = e.stderr
    return run_result


def num_tokens_from_string(string: str, encoding_name: str=optimizer_gpt_model) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


# gpt-4o 128,000
# gpt-3.5-turbo 16,385
call_optimizer_server_func = functools.partial(
    prompt_utils.call_openai_server_func,
    model=optimizer_gpt_model,
    max_decode_steps=optimizer_gpt_max_decode_steps,
    temperature=optimizer_gpt_temperature,
    top_p=optimizer_gpt_topp
)
# call_optimizer_server_func = functools.partial(
#     prompt_utils.call_ollama_func,
#     model='llama3-gradient:8b-instruct-1048k-fp16',
#     max_decode_steps=optimizer_gpt_max_decode_steps,
#     temperature=optimizer_gpt_temperature,
# )

# ====================== try calling the servers ============================
print("\n======== testing the optimizer server ===========")
optimizer_test_output = call_optimizer_server_func(
    "Does the sun rise from the north? Just answer yes or no.",
    temperature=1.0
)
print(f"optimizer test output: {optimizer_test_output}")
print("Finished testing the optimizer server.")
print("\n=================================================")


# ====================== read and process data ============================
if not os.path.exists(project_path / 'origin_test.csv'):
    print('saving')
    origin_train_df=pd.read_csv(train_file)
    origin_test_df=pd.read_csv(test_file)
    origin_test_df.to_csv(project_path / 'origin_test.csv')
    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(origin_train_df, test_size=0.2, stratify=origin_train_df[target_column], random_state=RANDOM_SEED)
    train_df.to_csv(project_path / 'train.csv')
    test_df.to_csv(project_path / 'test.csv')

with open(script_path,'r') as f:
    script=f.read()

def divide_script(script):
    sections = {'import':'', 'data': '', 'model': '', 'evaluation': ''}
    
    data_start = script.find('### Data ###')
    model_start = script.find('### Model ###')
    evaluation_start = script.find('### Evaluation ###')
    
    if data_start != -1:
        sections['import'] = script[:data_start].strip()
        if model_start != -1:
            sections['data'] += '\n' + script[data_start:model_start].strip()
            if evaluation_start != -1:
                sections['model'] = script[model_start:evaluation_start].strip()
                sections['evaluation'] = script[evaluation_start:].strip()
            else:
                sections['model'] = script[model_start:].strip()
        else:
            sections['data'] += '\n' + script[data_start:].strip()
    else:
        sections['data'] = script.strip()
    
    return sections


code_scetions=divide_script(script)


def initilize(code_scetions):
    # ====================== initialize search space ============================
    #1. summarize the data preprocessing code
    logger.info(f"{num_tokens_from_string(summary_prompt.format(code=code_scetions['import']+code_scetions['data']))} tks")
    rsp=call_optimizer_server_func(summary_prompt.format(code=code_scetions['import']+code_scetions['data']))
    data_report=rsp[0]

    #2. find the ml model and its docstring:
    logger.info(f"{num_tokens_from_string(give_docstring_prompt.format(code=code_scetions['import']+code_scetions['model']))} tks")
    rsp=call_optimizer_server_func(give_docstring_prompt.format(code=code_scetions['import']+code_scetions['model']))
    get_docstring_code=rsp[0]
    logger.info(f'{get_docstring_code=}')
    docstring=run_code(parse_code(get_docstring_code))
    logger.info(f'{docstring=}')

    #3. give hps list
    logger.info(f"{num_tokens_from_string(list_hps_prompt.format(report=data_report, docstring=docstring, model_code=code_scetions['import']+code_scetions['model']))} tks")
    rsp=call_optimizer_server_func(list_hps_prompt.format(report=data_report, docstring=docstring, model_code=code_scetions['import']+code_scetions['model']))
    hps_l=str(parse_json(rsp[0]))
    logger.info(f'{hps_l=}')

    return data_report,docstring,hps_l




data_code=code_scetions['import']+code_scetions['data']
data_code+="\ndf_train,df_test=divide_df(df_all)\ndf_train.to_pickle('./df_train.pkl')\ndf_test.to_pickle('./df_test.pkl')"
os.chdir(project_path)
exec_result=run_code(data_code)


model_code=code_scetions['import']
model_code+="""\ndf_train=pd.read_pickle('df_train.pkl')
df_test=pd.read_pickle('df_test.pkl')

X_train = StandardScaler().fit_transform(df_train.drop(columns=['Survived']))
y_train = df_train['Survived'].values
X_test = StandardScaler().fit_transform(df_test.drop(columns=['Survived']))
y_test=df_test['Survived'].values
"""
model_code+=code_scetions['model']
exec_result=run_code(model_code)
# print(exec_result)
# logger.info(f'{exec_result=}')

match = re.search(r'Average Score: (\d+\.\d+)', exec_result)
average_score = float(match.group(1)) if match else None
origin_script_regret=1-average_score


df_train=pd.read_pickle(project_path / 'df_train.pkl')
df_test=pd.read_pickle(project_path / 'df_test.pkl')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

#

X_train = StandardScaler().fit_transform(df_train.drop(columns=['Survived']))
y_train = df_train['Survived'].values
X_test = StandardScaler().fit_transform(df_test.drop(columns=['Survived']))
y_test=df_test['Survived'].values
# logger.info(f'{X_train.shape=}')
# logger.info(f'{X_test.shape=}')
ml_model_params={
    "criterion":'gini',
    "oob_score":True,
    "random_state":RANDOM_SEED,
    "n_jobs":-1,
    "verbose":0
}

logger.info(f'{ml_model_params=}')



def gen_history_prompt(
    bound_info,
    old_value_pairs_set,
    max_num_pairs=100,
):

    bound_names = bound_info.keys()
    logger.info(f'{old_value_pairs_set=}')

    old_value_pairs = list(old_value_pairs_set)
    old_value_pairs = sorted(old_value_pairs, key=lambda x: -x[-1])[
        -max_num_pairs:
    ]

    # print(old_value_pairs)
    old_value_pairs_substr = ""
    for i, pair in enumerate(old_value_pairs):
        old_value_pairs_substr += f"\nSuggestion_{i}: search_spaces: "
        # print('pair: ',pair)
        infos, best_params, regret_value = pair
        for name, info in zip(bound_names, infos):
            old_value_pairs_substr += f"{name} : "
            old_value_pairs_substr += '( '+', '.join(
                [f'{key}: {value}' for key, value in info])+'), '
        old_value_pairs_substr += f' best_params: {(best_params)},'
        old_value_pairs_substr += f' regret_value: {regret_value}'

    history=''
    history += old_value_pairs_substr.strip()

    return history



def process_output(search_space):
    """
    Process the output bounds.

    Args:
        search_space: suggestions from LLM

    Returns:
        Tuple containing bound information and space list.
    """
    space_list = []
    bound_info={}
    for hp_d in search_space[:]:
        if hp_d['hp_name'] in ml_model_params:
            search_space.remove(hp_d)
        elif hp_d['hp_name'] in ['n_estimators', 'max_depth',  'max_features',]:
            if hp_d['hp_type'] =='int':
                space_list.append(
                    {'name' : hp_d['hp_name'], 'type' : hp_d['hp_type'], 'lb' : -10, 'ub' : 10},
                )
            elif hp_d['hp_type'] =='num':
                space_list.append(
                    {'name' : hp_d['hp_name'], 'type' : hp_d['hp_type'], 'lb' : -1.0, 'ub' : 1.0},
                )
            elif hp_d['hp_type'] in ['cat']:
                space_list.append(
                    {'name' : hp_d['hp_name'], 'type' : hp_d['hp_type'], 'categories' : [str(ele) for ele in hp_d['search_space']]},
                )
            bound_info[hp_d['hp_name']] = {
                'search_space': hp_d['search_space'], "hp_type": hp_d['hp_type']
            }
    return bound_info, space_list


def evaluate_loss(ml_model,ml_model_params,space_list, result_l=[], hebo_config={},past_X=None,past_y=None,bound_range=None):
    #5. search and evaluation 
    raw_result_l=result_l[:]

    def inverse_scale_from_range(scaled_value, min_value, max_value, new_min, new_max):
        return (scaled_value - new_min) * (max_value - min_value) / (new_max - new_min) + min_value


    def preprocess_func(df):
        return_dict = {}
        for key,info,range in zip(df.columns,space_list,bound_range):
            value=df[key].iloc[0]
            hp_type=info['type']
            if hp_type == 'int':
                value=inverse_scale_from_range(value,range[0],range[1],-10,10)
                value=int(value)
            elif hp_type == 'num':
                value=inverse_scale_from_range(value,range[0],range[1],-1.0,1.0)
                value = np.round(value, 5)
            elif hp_type == 'cat' : 
                try: 
                    value= eval(value)
                except: value=value
            else: value= value

            return_dict[key]=value
        return return_dict

    def objective(df):
        return_dict=preprocess_func(df)
        # logger.info(f'{return_dict=}')
        single_best_model = ml_model(**{**ml_model_params, **return_dict})
        # `StratifiedKFold` is used for stratifying the target variable. The folds are made by preserving the percentage of samples for each class in target variable (`Survived`).

        N = 5
        oob = 0
        probs = pd.DataFrame(np.zeros((len(X_test), N * 2)), columns=[
                            'Fold_{}_Prob_{}'.format(i, j) for i in range(1, N + 1) for j in range(2)])

        skf = StratifiedKFold(n_splits=N, random_state=N, shuffle=True)
        for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
            # print('Fold {}\n'.format(fold))

            # Fitting the model
            single_best_model.fit(X_train[trn_idx], y_train[trn_idx])

            # X_test probabilities
            probs.loc[:, 'Fold_{}_Prob_0'.format(
                fold)] = single_best_model.predict_proba(X_test)[:, 0]
            probs.loc[:, 'Fold_{}_Prob_1'.format(
                fold)] = single_best_model.predict_proba(X_test)[:, 1]

            oob += single_best_model.oob_score_ / N
            # print('Fold {} OOB Score: {}\n'.format(fold, ml_model.oob_score_))
        return 1-np.round(oob,5)


    # Function to check if a value is within the legal range
    def is_in_legal_range(value, legal_range):
        if isinstance(value,int) or isinstance(value,float): 
            return legal_range[0] <= value <= legal_range[1]
        else: return value in legal_range
    sp = DesignSpace().parse(space_list)
    logger.info(f'{sp=}')
    opt = eval(hebo_config['optimizer'])(
        sp, model_name='gp', rand_sample=hebo_config['rand_sample'])
    # opt=BO(sp,model_name='gp', rand_sample=hebo_config['rand_sample'])
    # opt = HEBO(sp, model_name='gp', rand_sample=hebo_config['rand_sample'])
    if 'scramble_seed' in hebo_config:opt.scramble_seed=hebo_config['scramble_seed']
    if past_X is not None:
        legal_ranges = bound_range
        # Get the indices of rows where all values are within their legal ranges
        legal_row=past_X[past_X.apply(lambda row: all(is_in_legal_range(row[col], legal_ranges[i]) for i,col in enumerate(past_X.columns)), axis=1)]
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
            logger.info(f'{rec=}')
            result = objective(rec)
            y = np.array([result], dtype=np.float64).reshape(-1, 1)
            opt.observe(rec, y)
            result_l.append(result)
        except Exception as e:
            logger.debug(e)
            result_l=raw_result_l[:]
            result=1.0
            break
    best_config = preprocess_func(opt.best_x)

    return {
        'result': result,
        'best_params': best_config,
        'result_seq': result_l,
        'past_X':opt.X,
        'past_y':opt.y

    }


def run_tasks(hebo_config={}, call_optimizer_server_func=None):

    configs_dict = dict()
    results_dict = dict()
    old_value_pairs_set = set()
    data_report,docstring,hps_l=initilize(code_scetions)

    for i_rep in range(num_reps):

        # ====================== run optimization ============================
        configs_dict_single_rep = {
            "optimizer_llm_configs": optimizer_llm_dict,
            "max_num_steps": max_num_steps,
            "max_num_pairs": max_num_pairs,
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
        print("\n================ run optimization ==============")
        # result_l = []


        results_json_path = os.path.join(save_folder, "results.json")
        print(f"saving results to\n{results_json_path}")

        # print(f"\nStep {i_step}:")
        i_step=0
        past_X,past_y=None,None
        result_l=[]
        with tqdm(total=max_num_steps) as pbar:
            while i_step < max_num_steps:
                # print(f"\nStep {i_step}:")
                single_step_values=[]
                history = gen_history_prompt(
                    bound_info,
                    old_value_pairs_set,
                    max_num_pairs=max_num_pairs,
                ) if old_value_pairs_set else ''
                prompt=search_space_prompt.format(report=data_report, docstring=docstring, hps_l=hps_l,history=history,algo_name=algo_name)
                meta_prompts_dict[i_step] = prompt
                #4. givesuggestions

                rsp=call_optimizer_server_func(prompt)
                raw_outputs_dict[i_step] = rsp
                tmp=parse_json(rsp[0])
                search_space=tmp[0]['suggestions'] if "suggestions" in tmp[0] else tmp[0]
                logger.info(f'{search_space=}')
                bound_info, space_list = process_output(
                    search_space)
                bound_range = tuple([v['search_space']
                                    for v in bound_info.values()])
                logger.info(bound_range)
                loss = evaluate_loss(
                    ml_model, ml_model_params,space_list, result_l=result_l, hebo_config=hebo_config,past_X=past_X,past_y=past_y,bound_range=bound_range)
                if 'result' in loss and loss['result']<1.0:
                    result = loss['result']
                    regret_value = np.round(result, num_output_decimals)
                    best_params = loss['best_params']
                    result_l = loss['result_seq']
                    past_X=loss['past_X']
                    past_y=loss['past_y']
                    single_step_values.append(regret_value)
                    info = tuple((('search_space', tuple(v['search_space'])),
                                        ('hp_type', v['hp_type']),
                                    )
                                    for v in bound_info.values())
                    old_value_pairs_set.add(
                        (info, (str(best_params)), regret_value))
                    old_value_pairs_with_i_step.append(
                        (bound_range, regret_value, i_step))
                    i_step+=1
                    pbar.update(1)
                else: 
                    result_l=result_l[:(i_step)*hebo_config['hebo_iteration']]
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

hebo_config = {
    'optimizer': 'HEBO',
    "rand_sample": 4,
    "hebo_iteration": 24,
    "n_suggestions": 1,
    'scramble_seed': RANDOM_SEED
}

conv_llm35_bo_seq_l=[]
for _ in range(4):
    # run LLM+HEBO with 6 repeated runs
    res_l=run_tasks(hebo_config,call_optimizer_server_func)
    # for _ in range(6):
    llm35_bo_seq = np.array(res_l).reshape(-1, 1)
    conv_llm35_bo_seq=np.minimum.accumulate(llm35_bo_seq)
    conv_llm35_bo_seq_l.append(conv_llm35_bo_seq)
# logger.info(res_l)
    
# Most important features for RF algo: 
# n_estimators,max_features, max_depth

baseline_space_list=[
    {'name' : 'n_estimators', 'type' : 'int', 'lb' : 100, 'ub' : 2000},
    {'name' : 'max_depth', 'type' : 'int', 'lb' : 3, 'ub' : 20},
    # {'name' : 'min_samples_split', 'type' : 'int', 'lb' : 2, 'ub' : 12},
    # {'name' : 'min_samples_leaf', 'type' : 'num', 'lb' : 1.0, 'ub' : 7.0},
    {'name' : 'max_features', 'type' : 'cat', 'categories' :  ["sqrt","log2","None","1","2","3","4","5"]},
]
vanilla_seq_l=[]
for i in range(4):
    res_l=evaluate_loss(ml_model,ml_model_params,space_list=baseline_space_list,result_l=[],
                           hebo_config={
                                'optimizer': hebo_config['optimizer'],
                                "rand_sample": hebo_config['rand_sample'],
                                "hebo_iteration": max_num_steps*hebo_config['hebo_iteration'],
                                "n_suggestions": 1,
                                'scramble_seed': RANDOM_SEED+i
                           }
                           )['result_seq']
    vanilla_seq = np.array(res_l).reshape(-1, 1)
    vanilla_seq=np.minimum.accumulate(vanilla_seq)
    vanilla_seq_l.append(vanilla_seq)

results_plot_path = os.path.join(save_folder, "result.png")


def plot_regret(seqs, labels, ideal_point, ori_script_regret, plot_path=results_plot_path):
    plt.figure(figsize=(8, 6))

    for seq, label in zip(seqs, labels):
        logger.info(f'{seq.shape=}')
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
    
    plt.axhline(y=ori_script_regret, color='r', linestyle='--', label=f'Original script: {ori_script_regret}')

    plt.xlabel('Evaluation')
    plt.ylabel('Regret')
    plt.legend()
    plt.savefig(plot_path)

logger.info(f'{np.array(vanilla_seq_l).reshape(len(vanilla_seq_l),-1)=}')

plot_regret([
    np.array(vanilla_seq_l).reshape(len(vanilla_seq_l),-1),
    np.array(conv_llm35_bo_seq_l).reshape(len(conv_llm35_bo_seq_l),-1),
    # conv_llm35_bo_seq
    ],
    [
    # f'Random-{(max_num_steps+1)*hebo_config["hebo_iteration"]}iters',
    # f'BO-{(max_num_steps+1)*hebo_config["hebo_iteration"]}iters',
    f'HEBO-{(max_num_steps)*hebo_config["hebo_iteration"]}iters',
    f'gpt3.5-HEBO-{max_num_steps}steps',
    #  f'gpt4o-HEBO-{num_reps}reps-{max_num_steps}steps-{num_generated_points_in_each_step}pts'
     ], 0.0,ori_script_regret=origin_script_regret)
