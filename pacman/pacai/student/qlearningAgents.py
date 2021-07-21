from pacai.agents.learning.reinforcement import ReinforcementAgent
from pacai.util import reflection
from pacai.util import counter
import random
from pacai.util import probability

class QLearningAgent(ReinforcementAgent):
    """
    A Q-Learning agent.

    Some functions that may be useful:

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getAlpha`:
    Get the learning rate.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getDiscountRate`:
    Get the discount rate.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getEpsilon`:
    Get the exploration probability.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getLegalActions`:
    Get the legal actions for a reinforcement agent.

    `pacai.util.probability.flipCoin`:
    Flip a coin (get a binary value) with some probability.

    `random.choice`:
    Pick randomly from a list.

    Additional methods to implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Compute the action to take in the current state.
    With probability `pacai.agents.learning.reinforcement.ReinforcementAgent.getEpsilon`,
    we should take a random action and take the best policy action otherwise.
    Note that if there are no legal actions, which is the case at the terminal state,
    you should choose None as the action.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.update`:
    The parent class calls this to observe a state transition and reward.
    You should do your Q-Value update here.
    Note that you should never call this function, it will be called on your behalf.

    DESCRIPTION: Initialized q-values to 0 then updated based on the q-value update alg
    getQValue returns qvalue from values counter
    getValue returns highest val of all actions possible from s
    getPolicy returns best action from s, breaks ties using random.choice
    getAction uses coin flip on epsilon, if true pick random, false pick best val
    update updates qVal based on alg
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

        # You can initialize Q-values here.
        self.values = counter.Counter()
        self.ReinforcementAgent = ReinforcementAgent

    def getQValue(self, state, action):
        """
        Get the Q-Value for a `pacai.core.gamestate.AbstractGameState`
        and `pacai.core.directions.Directions`.
        Should return 0.0 if the (state, action) pair has never been seen.
        """
        return self.values[state, action]

    def getValue(self, state):
        """
        Return the value of the best action in a state.
        I.E., the value of the action that solves: `max_action Q(state, action)`.
        Where the max is over legal actions.
        Note that if there are no legal actions, which is the case at the terminal state,
        you should return a value of 0.0.

        This method pairs with `QLearningAgent.getPolicy`,
        which returns the actual best action.
        Whereas this method returns the value of the best action.
        """
        actions = self.ReinforcementAgent.getLegalActions(self, state)
        values = []
        for action in actions:
            values.append(self.getQValue(state, action))
        return max(values, default=0.0)

    def getPolicy(self, state):
        """
        Return the best action in a state.
        I.E., the action that solves: `max_action Q(state, action)`.
        Where the max is over legal actions.
        Note that if there are no legal actions, which is the case at the terminal state,
        you should return a value of None.

        This method pairs with `QLearningAgent.getValue`,
        which returns the value of the best action.
        Whereas this method returns the best action itself.
        """
        actions = self.ReinforcementAgent.getLegalActions(self, state)
        bestAction = None
        bestValue = float('-inf')
        for action in actions:
            value = self.getQValue(state, action)
            if value > bestValue:
                bestValue = value
                bestAction = action
            elif value == bestValue:
                bestAction = random.choice([bestAction, action])
        return bestAction

    def getAction(self, state):
        coin = probability.flipCoin(self.ReinforcementAgent.getEpsilon(self))
        actions = self.ReinforcementAgent.getLegalActions(self, state)
        if coin:
            return random.choice(actions)
        else:
            return self.getPolicy(state)

    def update(self, state, action, nextState, reward):
        alpha = self.ReinforcementAgent.getAlpha(self)
        discount = self.ReinforcementAgent.getDiscountRate(self)
        oldQValue = self.values[state, action]
        self.values[state, action] = oldQValue + (alpha * (reward
            + (discount * self.getValue(nextState)) - oldQValue))

class PacmanQAgent(QLearningAgent):
    """
    Exactly the same as `QLearningAgent`, but with different default parameters.
    """

    def __init__(self, index, epsilon = 0.05, gamma = 0.8, alpha = 0.2, numTraining = 0, **kwargs):
        kwargs['epsilon'] = epsilon
        kwargs['gamma'] = gamma
        kwargs['alpha'] = alpha
        kwargs['numTraining'] = numTraining

        super().__init__(index, **kwargs)

    def getAction(self, state):
        """
        Simply calls the super getAction method and then informs the parent of an action for Pacman.
        Do not change or remove this method.
        """

        action = super().getAction(state)
        self.doAction(state, action)

        return action

class ApproximateQAgent(PacmanQAgent):
    """
    An approximate Q-learning agent.

    You should only have to overwrite `QLearningAgent.getQValue`
    and `pacai.agents.learning.reinforcement.ReinforcementAgent.update`.
    All other `QLearningAgent` functions should work as is.

    Additional methods to implement:

    `QLearningAgent.getQValue`:
    Should return `Q(state, action) = w * featureVector`,
    where `*` is the dotProduct operator.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.update`:
    Should update your weights based on transition.

    DESCRIPTION: weights mapping features to weights,
    getQValue takes features -> multiplies each val by resp. weight sums and rets
    update takes weights of features -> updates each weight based on alg
    """

    def __init__(self, index,
            extractor = 'pacai.core.featureExtractors.IdentityExtractor', **kwargs):
        super().__init__(index, **kwargs)
        self.featExtractor = reflection.qualifiedImport(extractor)

        # You might want to initialize weights here.
        self.weights = counter.Counter()

    def getQValue(self, state, action):
        features = self.featExtractor.getFeatures(self.featExtractor, state, action)
        qValue = 0.0
        for key, value in features.items():
            weight = self.weights[key]
            qValue += weight * value
        return qValue

    def update(self, state, action, nextState, reward):
        features = self.featExtractor.getFeatures(self.featExtractor, state, action)
        alpha = self.ReinforcementAgent.getAlpha(self)
        discount = self.ReinforcementAgent.getDiscountRate(self)
        correct = (reward + (discount * self.getValue(nextState))) - self.getQValue(state, action)
        for key, value in features.items():
            weight = self.weights[key]
            self.weights[key] = weight + (alpha * correct * value)

    def final(self, state):
        """
        Called at the end of each game.
        """

        # Call the super-class final method.
        super().final(state)

        # Did we finish training?
        if self.episodesSoFar == self.numTraining:
            # You might want to print your weights here for debugging.
            # *** Your Code Here ***
            for key, value in self.weights.items():
                print(str(key) + ', ' + str(value))
