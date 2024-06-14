
task_context = {
    'model': 'RandomForest', 
    'task': 'classification', 
    'tot_feats': 30, 
    'cat_feats': 0, 
    'num_feats': 30, 
    'n_classes': 2, 
    'metric': 'accuracy', 
    'lower_is_better': False, 
    'num_samples': 455, 
    'hyperparameter_constraints': {
        'max_depth': ['int', 'linear', [1, 15]],        # [type, transform, [min_value, max_value]]
        'max_features': ['float', 'logit', [0.01, 0.99]], 
        'min_impurity_decrease': ['float', 'linear', [0.0, 0.5]], 
        'min_samples_leaf': ['float', 'logit', [0.01, 0.49]], 
        'min_samples_split': ['float', 'logit', [0.01, 0.99]], 
        'min_weight_fraction_leaf': ['float', 'logit', [0.01, 0.49]]
    }
}

import pandas as pd
import numpy as np
from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_val_score
import warnings
import random
from loguru import logger
import json
import pickle

from hebo.optimizers.util import parse_space_from_bayesmark

BAYESMARK_TASK_MAP = {
    'breast': ['classification', 'accuracy'],
    'digits': ['classification', 'accuracy'],
    'wine': ['classification', 'accuracy'],
    'iris': ['classification', 'accuracy'],
    'diabetes': ['regression', 'neg_mean_squared_error'],
}

PRIVATE_TASK_MAP = {
    'cutract': ['classification', 'accuracy'],
    'maggic': ['classification', 'accuracy'],
    'seer': ['classification', 'accuracy'],
    'griewank': ['regression', 'neg_mean_squared_error'],
    'ktablet': ['regression', 'neg_mean_squared_error'],
    'rosenbrock': ['regression', 'neg_mean_squared_error'],
}




    # define resu

class BayesmarkExpRunner:
    def __init__(self, task_context, dataset, seed):
        self.seed = seed
        self.model = task_context['model']
        self.task = task_context['task']
        self.metric = task_context['metric']
        self.dataset = dataset
        self.hyperparameter_constraints = task_context['hyperparameter_constraints']
        self.bbox_func = get_bayesmark_func(self.model, self.task, dataset['test_y'])
    
    def generate_initialization(self, n_samples):
        '''
        Generate initialization points for BO search
        Args: n_samples (int)
        Returns: init_configs (list of dictionaries, each dictionary is a point to be evaluated)
        '''

        # Read from fixed initialization points (all baselines see same init points)
        init_configs = pd.read_json(f'bayesmark/configs/{self.model}/{self.seed}.json').head(n_samples)
        init_configs = init_configs.to_dict(orient='records')

        assert len(init_configs) == n_samples

        return init_configs
        
    def evaluate_point(self, candidate_config):
        '''
        Evaluate a single point on bbox
        Args: candidate_config (dict), dictionary containing point to be evaluated
        Returns: (dict, dict), first dictionary is candidate_config (the evaluated point), second dictionary is fvals (the evaluation results)
        '''
        np.random.seed(self.seed)
        random.seed(self.seed)

        X_train, X_test, y_train, y_test = self.dataset['train_x'], self.dataset['test_x'], self.dataset['train_y'], self.dataset['test_y']

        for hyperparam, value in candidate_config.items():
            if self.hyperparameter_constraints[hyperparam][0] == 'int':
                candidate_config[hyperparam] = int(value)

        if self.task == 'regression':
            mean_ = np.mean(y_train)
            std_ = np.std(y_train)
            y_train = (y_train - mean_) / std_
            y_test = (y_test - mean_) / std_

        model = self.bbox_func(**candidate_config)
        scorer = get_scorer(self.metric)

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            S = cross_val_score(model, X_train, y_train, scoring=scorer, cv=5)
        cv_score = np.mean(S)
        
        model = self.bbox_func(**candidate_config)  
        model.fit(X_train, y_train)
        generalization_score = scorer(model, X_test, y_test)

        if self.metric == 'neg_mean_squared_error':
            cv_score = -cv_score
            generalization_score = -generalization_score

        return candidate_config, {'score': cv_score, 'generalization_score': generalization_score}

import pickle
import os
import pandas as pd

chat_engine = 'gpt-3.5-turbo' # LLM Chat Engine, currently our code only supports OpenAI LLM API
os.environ["OPENAI_API_TYPE"]="open_ai"
os.environ["OPENAI_API_VERSION"]=""
os.environ["OPENAI_API_KEY"]='sk-'
os.environ["OPENAI_API_BASE"]='https://api.openai.com/v1'
os.environ['HTTP_PROXY'] = '127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = '127.0.0.1:7890'

from llambo.llambo import LLAMBO
from bayesmark.bbox_utils import get_bayesmark_func



from hebo.optimizers.hebo import HEBO
from hebo.design_space.design_space import DesignSpace



import matplotlib.pyplot as plt

def plot_regret(seqs, labels, ideal_point, plot_path,task_context):
    plt.figure(figsize=(8, 6))
    for seq, label in zip(seqs, labels):
        if seq.shape[1] ==1: 
            plt.plot(seq-ideal_point, 'x-', label=label)
        else: 
            mean_regret = np.mean(seq-ideal_point, axis=0)
            std_regret = np.std(seq-ideal_point, axis=0)
            plt.plot(mean_regret, 'x-', label=label,)  # Mean regret line
            plt.fill_between(range(len(mean_regret)), 
                                mean_regret - std_regret, 
                                mean_regret + std_regret, 
                                alpha=0.3, )  # Filled area with slightly lower transparency
    plt.xlabel(f"Evaluation on {task_context['task']} using {task_context['model']}")
    plt.ylabel('Regret')
    plt.legend()
    plt.savefig(plot_path)



def evaluate_loss(benchmark, space_list, result_l=[], hebo_config={},past_X=None,past_y=None,bound_range=None):
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
        return_dict = {key:np.round(df[key].iloc[0], 5)  for key in df.columns}
        return return_dict

    def objective(df):
        # logger.info(f'{df=}')
        config = preprocess_func(df)
        # logger.info(f'{cfg=}')
        result_dict = benchmark.evaluate_point(config)
        return result_dict

    sp = space_list

    # random search

    opt = eval(hebo_config['optimizer'])(
        sp, model_name='gp', rand_sample=hebo_config['rand_sample'])
    if 'scramble_seed' in hebo_config:opt.scramble_seed=hebo_config['scramble_seed']
    if past_X is not None:
        opt.X=past_X
        opt.y=past_y
    # logger.info(f'X: {opt.X}, y: {opt.y}')
    logger.info(f"{hebo_config['optimizer']} searching... \n")
    for i in range(hebo_config['hebo_iteration']):
        rec = opt.suggest(n_suggestions=hebo_config['n_suggestions'])
        # logger.info(f'{rec=}')
        result_dict = objective(rec)
        valid_loss = result_dict[1]['score']
        # logger.info(f'{valid_loss=}')

        y = np.array([valid_loss], dtype=np.float64).reshape(-1, 1)
        # print(rec.iloc[0])
        # print(y)
        opt.observe(rec, y)
        best_config = preprocess_func(opt.best_x)
        result_dict_test = benchmark.evaluate_point(best_config)
        test_loss = result_dict_test[1]['generalization_score']
        # logger.info(f'{test_loss=}')

        result_l.append(test_loss)

    best_config = preprocess_func(opt.best_x)
    return {
        'configuration': best_config,
        # 'test_loss': np.round(test_loss, 5),
        'best_params': best_config,
        'hebo_seq': result_l,
        'past_X':opt.X,
        'past_y':opt.y

    }

for dataset in ["breast","digits"]:
    for model in ["RandomForest","SVM"]:

        # dataset = 'digits'
        # model = "SVM"
        num_seeds =0
        chat_engine = 'gpt-3.5-turbo'
        sm_mode = "discriminative"

        assert sm_mode in ['discriminative', 'generative']
        if sm_mode == 'generative':
            top_pct = 0.25
        else:
            top_pct = None

        # Load training and testing data
        if dataset in BAYESMARK_TASK_MAP:
            TASK_MAP = BAYESMARK_TASK_MAP
            pickle_fpath = f'bayesmark/data/{dataset}.pickle'
            with open(pickle_fpath, 'rb') as f:
                data = pickle.load(f)
            X_train = data['train_x']
            y_train = data['train_y']
            X_test = data['test_x']
            y_test = data['test_y']
        elif dataset in PRIVATE_TASK_MAP:
            TASK_MAP = PRIVATE_TASK_MAP
            pickle_fpath = f'private_data/{dataset}.pickle'
            with open(pickle_fpath, 'rb') as f:
                data = pickle.load(f)
            X_train = data['train_x']
            y_train = data['train_y']
            X_test = data['test_x']
            y_test = data['test_y']
        else:
            raise ValueError(f'Invalid dataset: {dataset}')


        # Describe task context
        task_context = {}
        task_context['model'] = model
        task_context['task'] = TASK_MAP[dataset][0]
        task_context['tot_feats'] = X_train.shape[1]
        task_context['cat_feats'] = 0       # bayesmark datasets only have numerical features
        task_context['num_feats'] = X_train.shape[1]
        task_context['n_classes'] = len(np.unique(y_train))
        task_context['metric'] = TASK_MAP[dataset][1]
        task_context['lower_is_better'] = True if 'neg' in task_context['metric'] else False
        task_context['num_samples'] = X_train.shape[0]

        with open('hp_configurations/bayesmark.json', 'r') as f:
            task_context['hyperparameter_constraints'] = json.load(f)[model]

        print(task_context['hyperparameter_constraints'])



        # load data
        pickle_fpath = f'bayesmark/data/{dataset}.pickle'
        with open(pickle_fpath, 'rb') as f:
            data = pickle.load(f)

        # instantiate BayesmarkExpRunner
        benchmark = BayesmarkExpRunner(task_context, data, num_seeds)
        n_steps=100
        n_rep=4
        # instantiate LLAMBO
        llambo = LLAMBO(task_context, sm_mode='discriminative', n_candidates=10, n_templates=2, n_gens=5, 
                        alpha=0.1, n_initial_samples=5, n_trials=n_steps, 
                        init_f=benchmark.generate_initialization,
                        bbox_eval_f=benchmark.evaluate_point, 
                        chat_engine=chat_engine)

        hyperparams={}
        for k,v in task_context['hyperparameter_constraints'].items():
            # print(v)
            type,space,_range = v
            hyperparams[k]={
                'type':"real" if type =='float' else type,
                'space': space ,
                "range": _range,

            }

        logger.info(hyperparams)
        bound_range = [v['range'] for k, v in hyperparams.items()]
        sp=parse_space_from_bayesmark(hyperparams)
        past_X=benchmark.generate_initialization(5)
        past_y=[]
        for cfg in past_X:
            past_y.append(benchmark.evaluate_point(cfg)[1]['score'])

        conv_llambo_seq_l,conv_hebo_seq_l=[],[]
        # run optimization
        for i in range(n_rep):

            llambo.seed=num_seeds+i
            configs, fvals = llambo.optimize()

            llambo_result = fvals['generalization_score'].tail(n_steps).tolist()

            llambo_seq=1.0-np.array(llambo_result).reshape(-1, 1)
            conv_llambo_seq=np.minimum.accumulate(llambo_seq)
            conv_llambo_seq_l.append(conv_llambo_seq)

            hebo_config = {
            "optimizer":"HEBO",
            "rand_sample": 4,
            "hebo_iteration": n_steps,
            "n_suggestions": 1,
            'scramble_seed': 42+i
            }
            loss=evaluate_loss(benchmark=benchmark,
                    space_list=sp,
                    result_l=[],
                    hebo_config=hebo_config,
                    # past_X=pd.DataFrame(past_X),
                    # past_y=np.array(past_y).reshape(-1,1),
                    bound_range=bound_range)


            hebo_result=loss['hebo_seq']
            

            hebo_seq=1.0-np.array(hebo_result).reshape(-1, 1)
            conv_hebo_seq=np.minimum.accumulate(hebo_seq)
            conv_hebo_seq_l.append(conv_hebo_seq)




        plot_regret([
            np.array(conv_hebo_seq_l).reshape(len(conv_hebo_seq_l),-1),
            # conv_llambo_seq,

            ],
            [

            f'HEBO-{n_steps}iters',
            # f'LLAMBO-{n_steps}steps',
            ], 0.0 ,plot_path=f'./llambo-hebo-{dataset}-{model}.png',task_context=task_context)
