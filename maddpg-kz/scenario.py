# -*- coding: utf-8 -*-
import numpy as np
from core import World, Agent
import math
class Scenario(object):
    def __init__(self):
        # self.game_time = 0
        self.one_battle_time = 300
        self.danger_time = 3
        # self.lock_time1 = 0
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_good_agents = 1
        num_adversaries = 2
        num_agents = num_adversaries + num_good_agents
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.survived = True
            # agent.game_time = 0
            agent.leader = True if i == 0 else False
            # agent.silent = True if i > 0 else False
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            agent.accel = 1 if agent.adversary else 0.5
            agent.max_speed = 4 if agent.adversary else 2.5
        self.reset_world(world)
        return world


    def reset_world(self, world):
        # set random initial states
        for i, agent in enumerate(world.agents):
            if i ==0:
                agent.state.p_pos = np.array([0, 1],dtype=np.float32)
                agent.danger_distance = 0.9
                agent.detect_distance = 0.45
                agent.angle = np.array([0], dtype=np.float32)
                agent.game_time = 0
                agent.lock_time =0
                agent.detect_angle =np.array(100/180 *np.pi,dtype=np.float32)
                agent.state.p_vel = np.array([0.1])
                agent.state.c = np.zeros(world.dim_c)
            elif i ==1:
                agent.state.p_pos = np.array([0, -1],dtype=np.float32)
                agent.danger_distance = 0.9
                agent.detect_distance = 0.45
                agent.angle = np.array([0],dtype=np.float32)
                agent.lock_time = 0
                agent.game_time = 0
                agent.detect_angle = np.array(100/180 * np.pi,dtype=np.float32)
                agent.state.p_vel = np.array([0.1])
                agent.state.c = np.zeros(world.dim_c)
            else:
                agent.state.p_pos = np.array([2.5, 1],dtype=np.float32)
                agent.danger_distance = 0.9
                agent.detect_distance = 2
                agent.angle = np.array([np.pi],dtype=np.float32)
                agent.detect_angle = np.array(np.pi/6,dtype= np.float32)
                agent.state.p_vel = np.array([0.12])
                agent.state.c = np.zeros(world.dim_c)
                agent.game_time =0
    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def has_winner(self,agent, world):
        state_info = [ False for i in range(3)]

        for ga in self.good_agents(world):
            for j,dd in enumerate(agent):
                # print('飞机的战斗时间是',dd.game_time)
                # 我方飞机是否取胜
                if dd.game_time >= self.one_battle_time :
                    print('超时！超时！超时！')
                    dd.game_time = 0
                    state_info[j] = True
                    state_info[2] = True
                    return state_info
                distance = np.sqrt(np.sum(np.square(ga.state.p_pos - dd.state.p_pos)))
                # is_survive = dd.survived
                if distance <= ga.danger_distance:
                    state_info[j] = True
                    print('我方飞机胜利')
                    return state_info

                # 我方飞机是否被锁定
                is_lock = self.detect_is_lock(dd, ga)
                if distance <= ga.detect_distance and is_lock:
                    dd.lock_time += 1
                    # print("我方飞机 {} 被敌机锁定，且距离为：{}".format(distance))
                    if dd.lock_time >= self.danger_time:
                        dd.survived = False
                        print("我方飞机被敌机锁定超过3s,失败！")
                        state_info[j] = True
                else:
                    dd.lock_time = 0
                    state_info[j] = False
        # print(state_info)
        return state_info


    def get_direct_angle(self, agent1,agent2):
        x = agent1.state.p_pos[0] - agent2.state.p_pos[0]
        y = agent1.state.p_pos[1] - agent2.state.p_pos[1]
        temp = math.atan2(y, x) / math.pi * 180
        angle = (temp + 360) % 360
        q = agent2.angle/np.pi * 180
        angle = angle - q
        if angle > 180:
            angle = angle - 360
        elif angle < -180:
            angle = angle + 360
        return angle
    def detect_is_lock(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        distance = np.sqrt(np.sum(np.square(delta_pos)))*100
        if distance > 200:
            return False
        else:
            if abs(self.get_direct_angle(agent2, agent1)) <= 60:
                if 80 <= abs(self.get_direct_angle(agent1, agent2)) < 85:
                    if distance < 120:
                        return True
                    else:
                        return False
                elif 85 <= abs(self.get_direct_angle(agent1, agent2)) < 95:
                    if distance< 100:
                        return True
                    else:
                        return False
                else:
                    return True
            else:
                return False
    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        #boundary_reward = -10 if self.outside_boundary(agent) else 0
        main_reward = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        return main_reward


    def agent_reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        rew = 0
        adversaries = self.adversaries(world)
        for adv in adversaries:

            # rew -= 0.01 * np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
            distance = np.clip(np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos))), -1, 1)

            reward = 1 - distance ** 0.4
            rew += reward
        for a in adversaries:
            if self.detect_is_lock(a, agent):
                rew += 3
        def bound(x):
            if x < 3:
                return 0
            if x < 3.5:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)

        return rew

    def adversary_reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        rew = 0
        shape = True
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        if shape:
            # print(agents[0].state.p_pos)
            for adv in adversaries:
                # rew -= 0.01 * min([np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos))) for a in agents])
                distance = np.clip(np.sqrt(np.sum(np.square(agents[0].state.p_pos - adv.state.p_pos))),-1,1)

                reward = 1 - distance**0.4
                rew += reward
        for ag in agents:
            for adv in adversaries:
                if self.detect_is_lock(adv, ag):
                    rew -= 5

        def bound(x):
            if x < -2:
                return 5
            if x < 3.5 :
                return 0
            return 8
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        agent_pos = []
        other_vel = []
        comm1 = []
        for other_agent in world.agents:  # world.entities:
            if other_agent == agent:continue
            comm1.append(other_agent.state.c)
            agent_pos.append(other_agent.state.p_pos - agent.state.p_pos)
            if not other_agent.adversary:
                other_vel.append(other_agent.state.p_vel)
        # comm2 = [world.agents[0].state.c]
        bool_lock = []
        for other_agent in world.agents:
            if other_agent.adversary and not agent.adversary:
                if self.get_direct_angle(other_agent,agent):
                    bool_lock.append([-1])
                else:
                    bool_lock.append([1])

        return np.concatenate([agent.state.p_pos] + [agent.state.p_vel] + agent_pos+ [agent.angle]  + other_vel + bool_lock)