from pacai.agents.learning.value import ValueEstimationAgent
from pacai.util import counter

class ValueIterationAgent(ValueEstimationAgent):
    """
    A value iteration agent.

    Make sure to read `pacai.agents.learning` before working on this class.

    A `ValueIterationAgent` takes a `pacai.core.mdp.MarkovDecisionProcess` on initialization,
    and runs value iteration for a given number of iterations using the supplied discount factor.

    Some useful mdp methods you will use:
    `pacai.core.mdp.MarkovDecisionProcess.getStates`,
    `pacai.core.mdp.MarkovDecisionProcess.getPossibleActions`,
    `pacai.core.mdp.MarkovDecisionProcess.getTransitionStatesAndProbs`,
    `pacai.core.mdp.MarkovDecisionProcess.getReward`.

    Additional methods to implement:

    `pacai.agents.learning.value.ValueEstimationAgent.getQValue`:
    The q-value of the state action pair (after the indicated number of value iteration passes).
    Note that value iteration does not necessarily create this quantity,
    and you may have to derive it on the fly.

    `pacai.agents.learning.value.ValueEstimationAgent.getPolicy`:
    The policy is the best action in the given state
    according to the values computed by value iteration.
    You may break ties any way you see fit.
    Note that if there are no legal actions, which is the case at the terminal state,
    you should return None.
    """

    def __init__(self, index, mdp, discountRate = 0.9, iters = 100, **kwargs):
        super().__init__(index, **kwargs)

        self.mdp = mdp
        self.discountRate = discountRate
        self.iters = iters
        self.values = counter.Counter()  # A Counter is a dict with default 0

        # Compute the values here.
        states = self.mdp.getStates()

        # repeat iter # of times
        for i in range(self.iters):
            # copy vi, loop through every state, update vi's
            valuesCopy = self.values.copy()
            for state in states:
                actions = self.mdp.getPossibleActions(state)
                # find best action from state
                newValues = []
                for action in actions:
                    qValue = self.getQValue(state, action)
                    newValues.append(qValue)
                valuesCopy[state] = max(newValues, default=valuesCopy[state])
            # initialize values to vi+1
            self.values = valuesCopy

    def getQValue(self, state, action):
        outcomes = self.mdp.getTransitionStatesAndProbs(state, action)
        sumOutcomes = 0
        # sum of every possible action based on alg
        for nextState, prob in outcomes:
            reward = self.mdp.getReward(state, action, nextState)
            sumOutcomes += (prob * (reward + (self.discountRate * self.values[nextState])))
        return sumOutcomes

    def getPolicy(self, state):
        bestAction = None
        bestQValue = float('-inf')
        actions = self.mdp.getPossibleActions(state)
        # find best action corres. to best val
        # if none returns None
        for action in actions:
            qValue = self.getQValue(state, action)
            if qValue > bestQValue:
                bestQValue = qValue
                bestAction = action
        return bestAction

    def getValue(self, state):
        """
        Return the value of the state (computed in __init__).
        """

        return self.values[state]

    def getAction(self, state):
        """
        Returns the policy at the state (no exploration).
        """

        return self.getPolicy(state)
