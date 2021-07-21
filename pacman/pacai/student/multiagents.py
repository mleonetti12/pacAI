import random

from pacai.agents.base import BaseAgent
from pacai.agents.search.multiagent import MultiAgentSearchAgent
from pacai.core import distance

class ReflexAgent(BaseAgent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.
    You are welcome to change it in any way you see fit,
    so long as you don't touch the method headers.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index)

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        `ReflexAgent.getAction` chooses among the best options according to the evaluation function.

        Just like in the previous project, this method takes a
        `pacai.core.gamestate.AbstractGameState` and returns some value from
        `pacai.core.directions.Directions`.
        """

        # Collect legal moves.
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions.
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best.

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current `pacai.bin.pacman.PacmanGameState`
        and an action, and returns a number, where higher numbers are better.
        Make sure to understand the range of different values before you combine them
        in your evaluation function.
        """

        successorGameState = currentGameState.generatePacmanSuccessor(action)

        # Useful information you can extract.
        # newPosition = successorGameState.getPacmanPosition()
        # oldFood = currentGameState.getFood()
        # newGhostStates = successorGameState.getGhostStates()
        # newScaredTimes = [ghostState.getScaredTimer() for ghostState in newGhostStates]

        # *** Your Code Here ***

        newGhostStates = successorGameState.getGhostStates()
        ghostPositions = [ghost.getPosition() for ghost in newGhostStates]
        foodPositions = currentGameState.getFood().asList()
        newPosition = successorGameState.getPacmanPosition()

        def man(goal):
            return distance.manhattan(newPosition, goal)

        ghostDistances = list(map(man, ghostPositions))
        foodDistances = list(map(man, foodPositions))

        # manhattan dist estimate of closest ghost
        closestGhostDist = min(ghostDistances)
        if closestGhostDist == 0:
            closestGhostDist = 0.001

        # get true distance to closest food to avoid progress loss
        closestFood = newPosition
        closestFoodDist = min(foodDistances, default=0)
        for i in range(len(foodPositions)):
            if foodDistances[i] == closestFoodDist:
                closestFood = foodPositions[i]

        closestFoodDist = distance.maze(newPosition, closestFood, currentGameState)
        if closestFoodDist == 0:
            closestFoodDist = 0.001

        # return 1 - reciprocal of closest ghost loc + reciprocal of closest food
        return (1 - (1 / closestGhostDist)) + (1 / closestFoodDist)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    A minimax agent.

    Here are some method calls that might be useful when implementing minimax.

    `pacai.core.gamestate.AbstractGameState.getNumAgents()`:
    Get the total number of agents in the game

    `pacai.core.gamestate.AbstractGameState.getLegalActions`:
    Returns a list of legal actions for an agent.
    Pacman is always at index 0, and ghosts are >= 1.

    `pacai.core.gamestate.AbstractGameState.generateSuccessor`:
    Get the successor game state after an agent takes an action.

    `pacai.core.directions.Directions.STOP`:
    The stop direction, which is always legal, but you may not want to include in your search.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gamestate):

        numAgents = gamestate.getNumAgents()

        # minimax recursive function
        def minimax(self, state, agent, depth, prevAction):
            # if leaf node, return static eval fcn
            # leaf if game over or depth limit hit
            if state.isOver() or depth == self.getTreeDepth():
                return (self.getEvaluationFunction()(state), prevAction)

            # maxValue function for max nodes
            def maxValue(self, state, agent, depth):
                maximum = float('-inf')
                # default return action
                returnAction = 'STOP'
                # loop through actions to find max
                for action in state.getLegalActions(agentIndex=agent):
                    succ = state.generateSuccessor(agent, action)
                    cost, move = minimax(self, succ, agent + 1, depth, action)
                    maximum = max(maximum, cost)
                    if maximum == cost:
                        returnAction = action
                # return max, action pair
                return (maximum, returnAction)

            # minValue function for min nodes
            def minValue(self, state, agent, depth):
                minimum = float('inf')
                # default return action
                returnAction = 'STOP'
                # loop through actions to find min
                for action in state.getLegalActions(agentIndex=agent):
                    succ = state.generateSuccessor(agent, action)
                    if agent + 1 == numAgents:
                        cost, move = minimax(self, succ, 0, depth + 1, action)
                        minimum = min(minimum, cost)
                    else:
                        cost, move = minimax(self, succ, agent + 1, depth, action)
                        minimum = min(minimum, cost)
                    if minimum == cost:
                        returnAction = action
                # return min, action pair
                return (minimum, returnAction)

            if agent == 0:
                return maxValue(self, state, agent, depth)
            else:
                return minValue(self, state, agent, depth)

        # run maxValue if max, else run minValue
        evaluation, action = minimax(self, gamestate, 0, 0, 'STOP')
        return action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    A minimax agent with alpha-beta pruning.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gamestate):

        numAgents = gamestate.getNumAgents()

        # minimax recursive function, with alpha/beta pruning
        def minimax(self, state, agent, depth, prevAction, alpha, beta):
            # if leaf node, return static eval fcn
            # leaf if game over or depth limit hit
            if state.isOver() or depth == self.getTreeDepth():
                return (self.getEvaluationFunction()(state), prevAction)

            # maxValue function for max nodes
            def maxValue(self, state, agent, depth, alpha, beta):
                maximum = float('-inf')
                # default return action
                returnAction = 'STOP'
                # loop through actions to find max
                for action in state.getLegalActions(agentIndex=agent):
                    succ = state.generateSuccessor(agent, action)
                    cost, move = minimax(self, succ, agent + 1, depth, action, alpha, beta)
                    maximum = max(maximum, cost)
                    # set alpha if max > alpha
                    alpha = max(alpha, maximum)
                    if maximum == cost:
                        returnAction = action
                    # stop exploring if cond met
                    if alpha >= beta:
                        break
                # return max, action pair
                return (maximum, returnAction)

            def minValue(self, state, agent, depth, alpha, beta):
                minimum = float('inf')
                # default return action
                returnAction = 'STOP'
                # loop through actions to find min
                for action in state.getLegalActions(agentIndex=agent):
                    succ = state.generateSuccessor(agent, action)
                    # if last agent, go back to 0 increment depth, else go next
                    if agent + 1 == numAgents:
                        cost, move = minimax(self, succ, 0, depth + 1, action, alpha, beta)
                        minimum = min(minimum, cost)
                        # update beta if min < beta
                        beta = min(beta, minimum)
                    else:
                        cost, move = minimax(self, succ, agent + 1, depth, action, alpha, beta)
                        minimum = min(minimum, cost)
                        # update beta if min < beta
                        beta = min(beta, minimum)
                    if minimum == cost:
                        returnAction = action
                    # if cond met, stop exploring
                    if alpha >= beta:
                        break
                # return min action pair
                return (minimum, returnAction)

            # run maxValue if max, else run minValue
            if agent == 0:
                return maxValue(self, state, agent, depth, alpha, beta)
            else:
                return minValue(self, state, agent, depth, alpha, beta)

        evaluation, action = minimax(self, gamestate, 0, 0, 'STOP', float('-inf'), float('inf'))
        return action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    An expectimax agent.

    All ghosts should be modeled as choosing uniformly at random from their legal moves.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the expectimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gamestate):

        numAgents = gamestate.getNumAgents()

        # expectimax recursive function
        def expectimax(self, state, agent, depth, prevAction):
            # if leaf node, return static eval fcn
            # leaf if game over or depth limit hit
            if state.isOver() or depth == self.getTreeDepth():
                return (self.getEvaluationFunction()(state), prevAction)

            # maxValue function for max nodes
            def maxValue(self, state, agent, depth):
                maximum = float('-inf')
                # default placeholder action
                returnAction = 'STOP'
                # loop through actions calculate max
                for action in state.getLegalActions(agentIndex=agent):
                    succ = state.generateSuccessor(agent, action)
                    cost, move = expectimax(self, succ, agent + 1, depth, action)
                    maximum = max(maximum, cost)
                    if maximum == cost:
                        returnAction = action
                # return max, action pair
                return (maximum, returnAction)

            # chanceValue function for chance nodes
            def chanceValue(self, state, agent, depth):
                returnAction = 'STOP'
                # store actions and costs list for calculating avg
                # and what action to return
                actions = []
                costs = []
                # loop through actions to calulate avg prob
                for action in state.getLegalActions(agentIndex=agent):
                    succ = state.generateSuccessor(agent, action)
                    actions.append(action)
                    # if last agent, go back to 0 increment depth, else go next
                    if agent + 1 == numAgents:
                        cost, move = expectimax(self, succ, 0, depth + 1, action)
                        costs.append(cost)
                    else:
                        cost, move = expectimax(self, succ, agent + 1, depth, action)
                        costs.append(cost)
                # avg of action probs
                expectValue = sum(costs) / len(costs)
                # equal chance of all actions, return 1 at random
                returnAction = random.choice(actions)
                return (expectValue, returnAction)
            # run maxValue if max, else run chanceValue
            if agent == 0:
                return maxValue(self, state, agent, depth)
            else:
                return chanceValue(self, state, agent, depth)

        evaluation, action = expectimax(self, gamestate, 0, 0, 'STOP')
        return action


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable evaluation function.

    DESCRIPTION: <write something here so we know what you did>
    Very similar to reflex eval function,
    used manhattan distance to estimate closest food and ghost locations
    then used maze distance to calculate true distance to closest food and ghost
    return a linear function of:
        1 - reciprocal of distance of closest ghost to incentivize avoidance
            weight: 0.1 not as important as food, can make pacman too cautious
        reciprocal of closest food distance
            weight: 0.8 most important metric imo
        getScore() default return
            weight: 0.1 wanted to include it

    """
    ghostStates = currentGameState.getGhostStates()
    ghostPositions = [ghost.getPosition() for ghost in ghostStates]
    foodPositions = currentGameState.getFood().asList()
    currentPosition = currentGameState.getPacmanPosition()

    def man(goal):
        return distance.manhattan(currentPosition, goal)

    ghostDistances = list(map(man, ghostPositions))
    foodDistances = list(map(man, foodPositions))

    # manhattan dist estimate of closest ghost
    closestGhostDist = min(ghostDistances)
    if closestGhostDist == 0:
        closestGhostDist = 0.001

    # get true distance to closest food to avoid progress loss
    closestFood = currentPosition
    closestFoodDist = min(foodDistances, default=0)
    for i in range(len(foodPositions)):
        if foodDistances[i] == closestFoodDist:
            closestFood = foodPositions[i]

    closestFoodDist = distance.maze(currentPosition, closestFood, currentGameState)
    if closestFoodDist == 0:
        closestFoodDist = 0.001

    # return 1 - reciprocal of closest ghost loc + reciprocal of closest food + score(weighted)
    return ((0.1 * (1 - (1 / closestGhostDist)))
        + (0.8 * (1 / closestFoodDist)) + (0.1 * currentGameState.getScore()))

class ContestAgent(MultiAgentSearchAgent):
    """
    Your agent for the mini-contest.

    You can use any method you want and search to any depth you want.
    Just remember that the mini-contest is timed, so you have to trade off speed and computation.

    Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
    just make a beeline straight towards Pacman (or away if they're scared!)

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`
    """

    def __init__(self, index, **kwargs):
        super().__init__(index)
