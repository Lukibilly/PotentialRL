import torch as tr
import torch.nn as nn
import numpy as np

def NNSequential(dimensions, activation, output_activation=nn.Identity):
    layers = []
    for i in range(len(dimensions)-1):
        act = activation if i < len(dimensions)-1 else output_activation
        layers += [nn.Linear(dimensions[i], dimensions[i+1]), act()]
    return nn.Sequential(*layers)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def update_target_agent(agent, target_agent):
    target_agent.actor.load_state_dict(agent.actor.state_dict()) #NOT NEEDED
    target_agent.critic1.load_state_dict(agent.critic1.state_dict())
    target_agent.critic2.load_state_dict(agent.critic2.state_dict())

class ReplayBuffer:
    def __init__(self, state_dim, act_dim, size, agent_batch_size):
        self.state_buf = np.zeros((size, agent_batch_size, state_dim), dtype=np.float32)
        self.state2_buf = np.zeros((size, agent_batch_size, state_dim), dtype=np.float32)
        self.action_buf = np.zeros((size, agent_batch_size, act_dim), dtype=np.float32)
        self.reward_buf = np.zeros((size, agent_batch_size), dtype=np.float32)
        self.done_buf = np.zeros((size, agent_batch_size), dtype=np.float32)
        self.pointer, self.size, self.max_size = 0, 0, size

    def store(self, state, act, rew, next_obs, done):
        self.state_buf[self.pointer] = state
        self.state2_buf[self.pointer] = next_obs
        self.action_buf[self.pointer] = act.detach().cpu().numpy()
        self.reward_buf[self.pointer] = rew
        self.done_buf[self.pointer] = done
        self.pointer = (self.pointer+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, memory_batch_size=32, device='cuda'):
        idxs = np.random.randint(0, self.size, size=memory_batch_size)
        batch = dict(observation=self.state_buf[idxs],
                     observation2=self.state2_buf[idxs],
                     action=self.action_buf[idxs],
                     reward=self.reward_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: tr.as_tensor(v, dtype=tr.float32, device=device) for k,v in batch.items()}
