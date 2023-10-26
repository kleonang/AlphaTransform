import fire
import torch
import warnings
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from alphagen.rl.env.wrapper import AlphaEnvWrapper, AlphaEnv
from alphagen.representation.tree import ExpressionBuilder

# Use the loaded model for inference or evaluation
def generate_alpha(model: MaskablePPO, env: AlphaEnvWrapper) -> 'tuple[ExpressionBuilder, float]':
    obs = env.reset()[0]
    done = False
    while not done:
        action, _ = model.predict(obs, action_masks=get_action_masks(env), deterministic=False)
        obs, reward, done, _, _ = env.step(action)
    tree = env.unwrapped._builder
    return tree, reward

def generate_n_distinct_alphas(model: MaskablePPO, env: AlphaEnvWrapper, n: int) -> 'list[tuple[ExpressionBuilder, float]]':
    precision = 1e-6 # Round reward to 6 decimal places
    alphas = []
    alpha_rewards = set()
    while len(alphas) < n:
        alpha, reward = generate_alpha(model, env)
        reward = round(reward / precision) * precision
        if reward not in alpha_rewards:
            alphas.append((alpha, reward))
            alpha_rewards.add(reward)
    return alphas

def calculate_mean_reward(alphas: 'list[tuple[ExpressionBuilder, float]]') -> float:
    return sum(reward for _, reward in alphas) / len(alphas)

def main(checkpoint_path: str, num_distinct_alphas_to_generate: int = 100):
    warnings.filterwarnings('ignore')
    # Load model from checkpoint and test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = AlphaEnv(device=device, print_expr=True)

    # checkpoint_path = "./checkpoints/new_sharpe_42_20231023125445/100352_steps.zip" # COMMENT THIS OUT
    model = MaskablePPO.load(checkpoint_path, env=env, verbose=1)
    alphas = generate_n_distinct_alphas(model, env, n=num_distinct_alphas_to_generate)
    if num_distinct_alphas_to_generate > 5:
        top_k = [i for i in range(5, num_distinct_alphas_to_generate + 1, 5)]
    else:
        top_k = [i for i in range(1, num_distinct_alphas_to_generate + 1)]
    for k in top_k:
        print(f"Top {k} mean reward: {calculate_mean_reward(alphas[:k])}")

if __name__ == '__main__':
    fire.Fire(main)
