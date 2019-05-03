import multiprocessing
from utils.replay_memory import Memory, MultiAgentMemory
from utils.remote_vector_env import dict_to_array
from utils.torch import *
import numpy as np
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'mask', 'next_state',
                                       'reward'))

class MultiAgent:

    def __init__(self, env, policy_net):
        self.env = env
        self.policy_net = policy_net
        self.memory = MultiAgentMemory()
        
    def collect_samples(self, num_steps):
        """
        collect samples and store it to memory
        reset at first
        if __all__ is True, automatically reset
        """
        self.memory = MultiAgentMemory()
        state = self.env.reset()
        for i in range(num_steps):
            with torch.no_grad():
                action = self.policy_net.select_action(state)
            next_state, rew, done, info = self.env.step(action)
            # if __all__ True, reset
            need_reset = list(map(lambda value:1 if value["__all__"] else 0,\
                                          list(done.values())))
            action, state, listed_next_state, rew, done, info, id_list = \
                    dict_to_array(action, state, next_state, rew, done, info)
            mask = list(map(lambda d: 0 if d else 1, done))    
            self.memory.push(state, action, mask, listed_next_state, rew)
            state = next_state
            # reset if need_reset is 1
            reseted_state = self.env.reset(need_reset)
            for key, value in reseted_state.items():
                state[key] = value
                
    def batch_generator(self, min_batch_size):
        """
        sample the memory by min_batch_size
        output Transition tuple "batch"
        """
        self.memory.convert_ndarray()
        batch_size = len(self.memory)
        for _ in range(batch_size // min_batch_size):
            rand_ids = np.random.randint(0, batch_size, min_batch_size)
            batch = Transition(self.memory.state[rand_ids], self.memory.action[rand_ids], \
                self.memory.mask[rand_ids], self.memory.next_state[rand_ids], \
                self.memory.reward[rand_ids])
            
            yield batch