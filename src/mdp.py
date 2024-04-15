"""
The abstract class MDP defines abstract methods, with their implementations located in gridworld.py. 
This class serves as a template for modeling various Markov Decision Processes in a grid-based environment.
"""
import random
from abc import ABC, abstractmethod


def pretty(d, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            pretty(value, indent + 1)
        else:
            print('\t' * (indent + 1) + str(value))


class MDP(ABC):
    @abstractmethod
    def get_states(self):
        """ Return all states of this MDP """
        pass

    @abstractmethod
    def get_actions(self, state):
        """ Return all actions with non-zero probability from this state """
        pass

    @abstractmethod
    def get_transitions(self, state, action):
        """ 
        Return all non-zero probability transitions for this action from this state,
        as a list of (state, probability) pairs
        """
        pass

    @abstractmethod
    def get_reward(self, state, action, next_state):
        """ Return the reward for transitioning from state to nextState via action """
        pass

    @abstractmethod
    def is_terminal(self, state):
        """ Return true if and only if state is a terminal state of this MDP """
        pass

    @abstractmethod
    def get_discount_factor(self):
        """ Return the discount factor for this MDP """
        pass

    @abstractmethod
    def get_initial_state(self):
        """ Return the initial state of this MDP """
        pass

    @abstractmethod
    def get_goal_states(self):
        """ Return all goal states of this MDP """
        pass

    def execute(self, state, action):
        """
        Return a new state and a reward for executing action in state, based on the underlying probability.
        This can be used for model-free learning methods, but requires a model to operate.
        Override for simulation-based learning
        """
        rand = random.random()
        cumulative_probability = 0.0
        for (new_state, probability) in self.get_transitions(state, action):
            if cumulative_probability <= rand <= probability + cumulative_probability:
                return (new_state, self.get_reward(state, action, new_state))
            cumulative_probability += probability
            if cumulative_probability >= 1.0:
                raise ValueError(
                        "Cumulative probability >= 1.0 for action "
                        + str(action)
                        + " from "
                        + str(state)
                )

        raise ValueError(
                "No outcome state in simulation for action"
                + str(action)
                + " from "
                + str(state)
        )

    def execute_policy(self, policy, episodes=100):
        """ Execute a policy on this mdp for a number of episodes """
        for _ in range(episodes):
            state = self.get_initial_state()
            while not self.is_terminal(state):
                action = policy.select_action(state)
                (next_state, reward) = self.execute(state, action)
                state = next_state
