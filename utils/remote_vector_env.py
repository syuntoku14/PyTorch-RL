from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import time
import ray
from collections import OrderedDict
import numpy as np

logger = logging.getLogger(__name__)
FREE_DELAY_S = 10.0
MAX_FREE_QUEUE_SIZE = 100
ASYNC_RESET_RETURN = "async_reset_return"
_DUMMY_AGENT_ID = "agent0"
_last_free_time = 0.0
_to_free = []


def dict_to_array(*args):
    """
    convert obs, rew, action, etc dictionaries to lists for pytorch model 
    id_list ensures the corresponding pair of [env_id, agent]
    input: {1: {"agent1": obs, "agent2": obs, ...}, 2: {"agent1": obs}}
    output: obss: [obs, obs, obs...]
            rews: [rew, rew, ...]
            id_list: [[1, "agent1"], [1, "agent2"], [2, ...]]
    """
    def make_id_list(*args):
        obs = args[0]
        id_list = []
        for env_id, value in obs.items():
            for agent_id in value:
                id_list.append([env_id, agent_id])
        return id_list

    result = []
    id_list = make_id_list(*args)
    
    for arg in args: # obs, rew, ...
        # init the result with np.nan * shape
        try: # if arg is np.array
            shape = list(list(arg.values())[0].values())[0].shape
            listed = [np.full(shape, np.nan)] * len(id_list)
        except: # if arg is int, bool, etc
            listed = [np.nan] * len(id_list)
        for env_id, value in arg.items():
            for agent_id, v in value.items():
                if agent_id == '__all__':
                    continue
                id_ = [env_id, agent_id]
                if id_ in id_list:
                    index = id_list.index(id_)
                    listed[index] = v
        result.append(listed)
    return (*result), id_list


def ray_get_and_free(object_ids):
    """Call ray.get and then queue the object ids for deletion.
    This function should be used whenever possible in RLlib, to optimize
    memory usage. The only exception is when an object_id is shared among
    multiple readers.
    Args:
        object_ids (ObjectID|List[ObjectID]): Object ids to fetch and free.
    Returns:
        The result of ray.get(object_ids).
    """

    global _last_free_time
    global _to_free

    result = ray.get(object_ids)
    if type(object_ids) is not list:
        object_ids = [object_ids]
    _to_free.extend(object_ids)

    # batch calls to free to reduce overheads
    now = time.time()
    if (len(_to_free) > MAX_FREE_QUEUE_SIZE
            or now - _last_free_time > FREE_DELAY_S):
        ray.internal.free(_to_free)
        _to_free = []
        _last_free_time = now

    return result

class RemoteVectorEnv:
    """Vector env that executes envs in remote workers.
    This provides dynamic batching of inference as observations are returned
    from the remote simulator actors. Both single and multi-agent child envs
    are supported, and envs can be stepped synchronously or async.
    """

    def __init__(self, make_env, num_envs, multiagent,
                 remote_env_batch_wait_ms):
        self.make_local_env = make_env
        self.num_envs = num_envs
        self.multiagent = multiagent
        self.poll_timeout = int(remote_env_batch_wait_ms / 1000)

        self.actors = []
        self.pending = OrderedDict()  
        
    def poll(self):
        # when there is no pending, reset
        if len(self.pending) == 0:
            self.pending = {a.reset.remote(): a for a in self.actors}

        # each keyed by env_id in [0, num_remote_envs)
        obs, rewards, dones, infos = OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict()
        ready = []

        if len(self.pending) == 0:
            return None
        
        # Wait for at least 1 env to be ready here
        while not ready:
            ready, _ = ray.wait(
                list(self.pending),
                num_returns=len(self.pending),
                timeout=self.poll_timeout)

        # Get and return observations for each of the ready envs
        env_ids = set()
        for obj_id in ready:
            actor = self.pending.pop(obj_id)
            env_id = self.actors.index(actor)
            env_ids.add(env_id)
            ob, rew, done, info = ray_get_and_free(obj_id)
            obs[env_id] = ob
            rewards[env_id] = rew
            dones[env_id] = done
            infos[env_id] = info

        logger.debug("Got obs batch for actors {}".format(env_ids))
        return obs, rewards, dones, infos

    def send_actions(self, action_dict):
        for env_id, actions in action_dict.items():
            actor = self.actors[env_id]
            obj_id = actor.step.remote(actions)
            self.pending[obj_id] = actor

    def try_reset(self, env_id):
        actor = self.actors[env_id]
        obj_id = actor.reset.remote()
        self.pending[obj_id] = actor
        return ASYNC_RESET_RETURN

    def stop(self):
        if len(self.actors) is not 0:
            for actor in self.actors:
                actor.__ray_terminate__.remote()


@ray.remote(num_cpus=0)
class _RemoteMultiAgentEnv(object):
    """Wrapper class for making a multi-agent env a remote actor."""

    def __init__(self, make_env, i):
        self.env = make_env(i)

    def reset(self):
        obs = self.env.reset()
        # each keyed by agent_id in the env
        rew = {agent_id: 0 for agent_id in obs.keys()}
        info = {agent_id: {} for agent_id in obs.keys()}
        done = {"__all__": False}
        return obs, rew, done, info

    def step(self, action_dict):
        return self.env.step(action_dict)


@ray.remote(num_cpus=0)
class _RemoteSingleAgentEnv(object):
    """Wrapper class for making a gym env a remote actor."""

    def __init__(self, make_env, i):
        self.env = make_env(i)

    def reset(self):
        obs = {_DUMMY_AGENT_ID: self.env.reset()}
        rew = {agent_id: 0 for agent_id in obs.keys()}
        info = {agent_id: OrderedDict() for agent_id in obs.keys()}
        done = {"__all__": False}
        return obs, rew, done, info

    def step(self, action):
        obs, rew, done, info = self.env.step(action[_DUMMY_AGENT_ID])
        obs, rew, done, info = [{
            _DUMMY_AGENT_ID: x
        } for x in [obs, rew, done, info]]
        done["__all__"] = done[_DUMMY_AGENT_ID]
        return obs, rew, done, info
    
class MultiAgentVecEnv(RemoteVectorEnv):
    """
    Vectorised multiagent env
    reset: reset with mask
    step: step all the environment
    """
    def __init__(self, make_env, num_envs, remote_env_batch_wait_ms):
        super().__init__(make_env, num_envs, True, remote_env_batch_wait_ms)
        
    def step(self, action_dicts):
        self.send_actions(action_dicts)
        return self._poll_and_return()
    
    def reset(self, mask=None):
        # create actors when there is no actors
        if len(self.actors) == 0:
            def make_remote_env(i):
                logger.info("Launching env {} in remote actor".format(i))
                if self.multiagent:
                    return _RemoteMultiAgentEnv.remote(self.make_local_env, i)
                else:
                    return _RemoteSingleAgentEnv.remote(self.make_local_env, i)

            self.actors = [make_remote_env(i) for i in range(self.num_envs)]

        for id_ in range(len(self.actors)):
            if mask is None:
                self.try_reset(id_)
            elif mask[id_]:
                self.try_reset(id_)
                
        obs, rew, done, info = self._poll_and_return()
        return obs
    
    def _poll_and_return(self):
        obs, rewards, dones, info = OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict()
        while len(self.pending) != 0:
            ob_, rew_, done_, info_ = self.poll()
            obs.update(ob_)
            rewards.update(rew_)
            dones.update(done_)
            info.update(info_)
        return obs, rewards, dones, info
    