import numpy as np
import pandas as pd
from torch.quasirandom import SobolEngine
from hebo.design_space.design_space import DesignSpace
from hebo.acquisitions.acq import MACE
from hebo.acq_optimizers.evolution_optimizer import EvolutionOpt
from hebo.optimizers.abstract_optimizer import AbstractOptimizer
from typing import Optional

class RandomSearch(AbstractOptimizer):
    support_parallel_opt = True
    support_combinatorial = True
    support_contextual = True

    def __init__(self, space, model_name='',rand_sample: Optional[int] = None, scramble_seed: Optional[int] = None):
        super().__init__(space)
        self.space = space
        self.X = pd.DataFrame(columns=self.space.para_names)
        self.y = np.zeros((0, 1))
        self.rand_sample = 1 + self.space.num_paras if rand_sample is None else max(2, rand_sample)
        self.scramble_seed = scramble_seed
        self.sobol = SobolEngine(self.space.num_paras, scramble=True, seed=scramble_seed)

    def quasi_sample(self, n, fix_input=None):
        samp = self.sobol.draw(n)
        samp = samp * (self.space.opt_ub - self.space.opt_lb) + self.space.opt_lb
        x = samp[:, :self.space.num_numeric]
        xe = samp[:, self.space.num_numeric:]
        for i, n in enumerate(self.space.numeric_names):
            if self.space.paras[n].is_discrete_after_transform:
                x[:, i] = x[:, i].round()
        df_samp = self.space.inverse_transform(x, xe)
        if fix_input is not None:
            for k, v in fix_input.items():
                df_samp[k] = v
        return df_samp

    def suggest(self, n_suggestions=1, fix_input=None):
        sample = self.quasi_sample(n_suggestions, fix_input)
        return sample

    def observe(self, X, y):
        
        valid_id = np.where(np.isfinite(y.reshape(-1)))[0].tolist()
        XX = X.iloc[valid_id]
        yy = y[valid_id].reshape(-1, 1)
        self.X = pd.concat([self.X, XX], axis=0, ignore_index=True)
        self.y = np.vstack([self.y, yy])

    @property
    def best_x(self) -> pd.DataFrame:
        if self.X.shape[0] == 0:
            raise RuntimeError('No data has been observed!')
        else:
            return self.X.iloc[[self.y.argmin()]]

    @property
    def best_y(self) -> float:
        if self.X.shape[0] == 0:
            raise RuntimeError('No data has been observed!')
        else:
            return self.y.min()

