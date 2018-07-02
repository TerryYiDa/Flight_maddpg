import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import animation
# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, angle_callback=None, done_callback=None):

        self.world = world
        self.agents = self.world.policy_agents
        # set required vectorized gym env property
        self.n = len(world.policy_agents)
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.angle_callback = angle_callback
        self.done_callback = done_callback
        # environment parameters
        self.discrete_action_space = True
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False
        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = world.discrete_action if hasattr(world, 'discrete_action') else False
        # if true, every agent has the same reward
        self.shared_reward = False
        self.time = 0

        # configure spaces
        self.action_space = []
        self.observation_space = []
        for agent in self.agents:
            total_action_space = []
            # physical action space
            if self.discrete_action_space:
                u_action_space = spaces.Discrete(world.dim_p )
            else:
                u_action_space = spaces.Box(low=-agent.u_range, high=+agent.u_range, shape=(world.dim_p,))
            if agent.movable:
                total_action_space.append(u_action_space)
            # communication action space
            if self.discrete_action_space:
                c_action_space = spaces.Discrete(world.dim_c)
            else:
                c_action_space = spaces.Box(low=0.0, high=1.0, shape=(world.dim_c,))
            if not agent.silent:
                total_action_space.append(c_action_space)
            # total action space
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = spaces.MultiDiscrete([[0,act_space.n-1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])
            # observation space
            obs_dim = len(observation_callback(agent, self.world))
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,),))
            agent.action.c = np.zeros(self.world.dim_c)

    def step(self, action_n):
        obs_n = []
        reward_n = []
        # done_n = []
        # set action for each agent
        for i, agent in enumerate(self.agents):
            # self.set_action(action_n[i], agent, self.action_space[i])
            ag = action_n[i] * np.pi /9
            agent.angle += ag
            angle = agent.angle[0] /np.pi *180
            if angle < 0:
                angle += 360
            if angle >= 360:
                angle -= 360
            agent.state.p_pos += [agent.state.p_vel[0] * math.cos(angle/180 * math.pi),agent.state.p_vel[0] * math.sin(angle/180 * math.pi )]
        # advance world state
        # self.world.step()
        adv = [agent for agent in self.agents if agent.adversary]
        # record observation for each agent
        for agent in self.agents:
            if agent.adversary:
                agent.game_time+=1
            obs_n.append(self.get_obs(agent))
            reward_n.append(self.get_reward(agent))
        # for agent in adv:
        done_n = self.get_done(adv)
        # if False in done_n:
        #     info = False
        # else:info = True
        # all agents get total reward in cooperative case
        reward = np.sum(reward_n)
        if self.shared_reward:
            reward_n = [reward] * self.n
        return obs_n, reward_n, done_n

    def reset(self):
        # reset world
        self.reset_callback(self.world)
        # record observations for each agent
        obs_n = []
        self.agents = self.world.policy_agents
        for agent in self.agents:
            obs_n.append(self.get_obs(agent))
        return obs_n

    # get observation for a particular agent
    def get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    # get dones for a particular agent
    def get_done(self, agent):
        if self.done_callback is None:
            return False
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)

    # set env action for a particular agent
    def set_action(self, action, agent, action_space):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)
        # process action
        if isinstance(action_space, spaces.MultiDiscrete):
            act = []
            size = action_space.high - action_space.low + 1
            index = 0
            for s in size:
                act.append(action[index:(index + s)])
                index += s
            action = act
        else:
            action = [action]

        if agent.movable:
            # physical action
            if self.discrete_action_input:
                agent.action.u = np.zeros(self.world.dim_p)
                # process discrete action
                if action[0] == 1: agent.action.u[0] = -1.0
                if action[0] == 2: agent.action.u[0] = +1.0
                if action[0] == 3: agent.action.u[1] = -1.0
                if action[0] == 4: agent.action.u[1] = +1.0
            else:
                if self.force_discrete_action:
                    d = np.argmax(action[0])
                    action[0][:] = 0.0
                    action[0][d] = 1.0
                if self.discrete_action_space:
                    # agent.action.u[0] += action[0][1] - action[0][2]
                    agent.action.u[0] += action[0][0]
                else:
                    agent.action.u = action[0]
            sensitivity = 1.25
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity
            # a = np.clip(np.random.normal(action[0], 2), *(-1, 1))  # add randomness to action selection for exploration
            action_s = np.clip(agent.action.u,*(-1,1))
            agent.action.u = action_s * np.pi / 9
            action = action[1:]
        if not agent.silent:
            # communication action
            if self.discrete_action_input:
                agent.action.c = np.zeros(self.world.dim_c)
                agent.action.c[action[0]] = 1.0
            else:
                agent.action.c = action[0]
            action = action[1:]
        # make sure we used all elements of action
        assert len(action) == 0, 'action len is not 2'

    def render(self, state):
        plt.plot(state[0][0], state[0][1], c='r', marker='o')
        plt.plot(state[1][0], state[1][1], c='r', marker='o')
        plt.plot(state[2][0], state[2][1], c='b', marker='o')
        plt.xlim(-2,4)
        plt.ylim(-5,5)
        # plt.show()
        plt.pause(0.5)
#



