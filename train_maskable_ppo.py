import os
from typing import Optional, Tuple, Union
from datetime import datetime
import fire
import torch

import numpy as np
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

from alphagen.rl.env.wrapper import AlphaEnv
from alphagen.rl.policy import LSTMSharedNet
from alphagen.utils.random import reseed_everything
from alphagen.rl.env.core import AlphaEnvCore
from backtester.StrategySimulator import StrategySimulator


class CustomCallback(BaseCallback):
    def __init__(self,
                 save_freq: int,
                 show_freq: int,
                 save_path: str,
                 strategy_simulator: StrategySimulator,
                 name_prefix: str = 'rl_model',
                 timestamp: Optional[str] = None,
                 verbose: bool = False):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.show_freq = show_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.verbose = verbose

        self.strategy_simulator = strategy_simulator

        if timestamp is None:
            self.timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        else:
            self.timestamp = timestamp

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        assert self.logger is not None
        # self.logger.record('pool/size', self.pool.size)
        # self.logger.record('pool/significant', (np.abs(self.pool.weights[:self.pool.size]) > 1e-4).sum())
        # self.logger.record('pool/best_ic_ret', self.pool.best_ic_ret)
        # self.logger.record('pool/eval_cnt', self.pool.eval_cnt)
        # ic_test, rank_ic_test = self.pool.test_ensemble(self.test_calculator)
        # self.logger.record('test/ic', ic_test)
        # self.logger.record('test/rank_ic', rank_ic_test)
        self.save_checkpoint()

    def save_checkpoint(self):
        path = os.path.join(self.save_path, f'{self.name_prefix}_{self.timestamp}', f'{self.num_timesteps}_steps')
        self.model.save(path)   # type: ignore
        if self.verbose:
            print(f'Saving model checkpoint to {path}')
        # with open(f'{path}_pool.json', 'w') as f:
        #     json.dump(self.pool.to_dict(), f)

    # def show_pool_state(self):
    #     state = self.pool.state
    #     n = len(state['exprs'])
    #     print('---------------------------------------------')
    #     for i in range(n):
    #         weight = state['weights'][i]
    #         expr_str = str(state['exprs'][i])
    #         ic_ret = state['ics_ret'][i]
    #         print(f'> Alpha #{i}: {weight}, {expr_str}, {ic_ret}')
    #     print(f'>> Ensemble ic_ret: {state["best_ic_ret"]}')
    #     print('---------------------------------------------')

    # @property
    # def pool(self) -> AlphaPoolBase:
    #     return self.env_core.pool

    @property
    def env_core(self) -> AlphaEnvCore:
        return self.training_env.envs[0].unwrapped  # type: ignore


def main(
    seed: int = 0,
    instruments: str = "nasdaq",
    steps: int = 200_000
):
    reseed_everything(seed)

    # Set simulation start and end dates
    sim_start = '2011-01-01'
    sim_end = '2017-12-07'
    # Set in-sample and out-of-sample start dates
    is_start = '2013-04-10'
    os_start = '2017-01-03'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Initialize simulator
    strategy_simulator = StrategySimulator(sim_start, sim_end, is_start, os_start)

    env = AlphaEnv(device=device, print_expr=True)

    name_prefix = f"new_{instruments}_{seed}"
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    checkpoint_callback = CustomCallback(
        save_freq=10000,
        show_freq=10000,
        save_path='./checkpoints',
        strategy_simulator=strategy_simulator,
        name_prefix=name_prefix,
        timestamp=timestamp,
        verbose=1,
    )

    model = MaskablePPO(
        'MlpPolicy',
        env,
        policy_kwargs=dict(
            features_extractor_class=LSTMSharedNet,
            features_extractor_kwargs=dict(
                n_layers=2,
                d_model=128,
                dropout=0.1,
                device=device,
            ),
        ),
        gamma=1.,
        ent_coef=0.01,
        batch_size=128,
        tensorboard_log='./tensorboard',
        device=device,
        verbose=1,
    )
    model.learn(
        total_timesteps=steps,
        callback=checkpoint_callback,
        tb_log_name=f'{name_prefix}_{timestamp}',
    )


def fire_helper(
    seed: Union[int, Tuple[int]],
    instruments: str,
    steps: int = None
):
    if isinstance(seed, int):
        seed = (seed, )
    default_steps = {
        10: 250_000,
        20: 300_000,
        50: 350_000,
        100: 400_000
    }
    for _seed in seed:
        main(_seed,
             instruments,
             default_steps[10] if steps is None else int(steps)
             )


if __name__ == '__main__':
    fire.Fire(fire_helper)
