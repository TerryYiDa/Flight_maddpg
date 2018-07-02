from environment import MultiAgentEnv
from scenario import Scenario

def make_env():

    w = Scenario()
    world = w.make_world()
    env = MultiAgentEnv(world, w.reset_world, w.reward, w.observation, w.get_direct_angle, w.has_winner)
    return env
