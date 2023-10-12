import os
from typing import Optional, Tuple, Union
from datetime import datetime
import fire
import torch

import numpy as np
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

from alphagen.rl.env.wrapper import AlphaEnv
from alphagen.rl.policy import LSTMSharedNet, TransformerSharedNet
from alphagen.utils.random import reseed_everything
from alphagen.rl.env.core import AlphaEnvCore


class CustomCallback(BaseCallback):
    def __init__(self,
                 save_freq: int,
                 show_freq: int,
                 save_path: str,
                 name_prefix: str = 'rl_model',
                 timestamp: Optional[str] = None,
                 verbose: bool = False):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.show_freq = show_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.verbose = verbose

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
        self.save_checkpoint()

    def save_checkpoint(self):
        path = os.path.join(self.save_path, f'{self.name_prefix}_{self.timestamp}', f'{self.num_timesteps}_steps')
        self.model.save(path)   # type: ignore
        if self.verbose:
            print(f'Saving model checkpoint to {path}')

    @property
    def env_core(self) -> AlphaEnvCore:
        return self.training_env.envs[0].unwrapped  # type: ignore


def main(
    seed: int = 0,
    instruments: str = "nasdaq",
    steps: int = 200_000
):
    reseed_everything(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    env = AlphaEnv(device=device, print_expr=True)

    name_prefix = f"new_{instruments}_{seed}"
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    checkpoint_callback = CustomCallback(
        save_freq=10000,
        show_freq=10000,
        save_path='./checkpoints',
        name_prefix=name_prefix,
        timestamp=timestamp,
        verbose=1,
    )

    model = MaskablePPO(
        'MlpPolicy',
        env,
        # UNCOMMENT FOR LSTM POLICY
        # policy_kwargs=dict(
        #     features_extractor_class=LSTMSharedNet,
        #     features_extractor_kwargs=dict(
        #         n_layers=2,
        #         d_model=128,
        #         dropout=0.1,
        #         device=device,
        #     ),
        # ),
        policy_kwargs=dict(
            features_extractor_class=TransformerSharedNet,
            features_extractor_kwargs=dict(
                n_encoder_layers=2,
                d_model=128,
                n_head=4,
                d_ffn=256,
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
        1: 100_000,
        10: 250_000,
        20: 300_000,
        50: 350_000,
        100: 400_000
    }
    for _seed in seed:
        main(_seed,
             instruments,
             default_steps[1] if steps is None else int(steps)
             )


if __name__ == '__main__':
    fire.Fire(fire_helper)
