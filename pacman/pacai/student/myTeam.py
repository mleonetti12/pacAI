from pacai.agents.capture.capture import CaptureAgent
import random
from pacai.util import counter
import time
import logging
from pacai.util import util
from pacai.core.directions import Directions

# main attack agent that was implemented in final submission
class AttackAgent(CaptureAgent):
    def __init__(self, index, **kwargs):
        super().__init__(index)

    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the agent and populates useful fields,
        such as the team the agent is on and the `pacai.core.distanceCalculator.Distancer`.

        IMPORTANT: If this method runs for more than 15 seconds, your agent will time out.
        """

        super().registerInitialState(gameState)

        # Your initialization code goes here, if you need any.

    def chooseAction(self, gameState):
        """
        Pick action based on evalFcn, default to random
        """
        actions = gameState.getLegalActions(self.index)

        start = time.time()
        values = [self.evalFcn(gameState, a) for a in actions]
        logging.debug('evaluate() time for agent %d: %.4f' % (self.index, time.time() - start))

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        return random.choice(bestActions)

    def evalFcn(self, state, action):
        weights = self.weightsGenerator(state, action)
        features = self.featureGenerator(state, action)
        return features * weights

    # unused features remain commented out
    def featureGenerator(self, state, action):

        successor = self.getSuccessor(state, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()
        features = counter.Counter()

        features['successorScore'] = self.getScore(successor)

        foodList = self.getFood(successor).asList()
        if (len(foodList) > 0):
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = minDistance

        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        guards = [a for a in enemies if a.isGhost() and a.getPosition() is not None]
        # features['numGuards'] = len(guards)

        guardDists = [self.getMazeDistance(myPos, a.getPosition()) for a in guards]
        if (len(guardDists) > 0):
            minGuardDistance = min(guardDists)
            # if minGuardDistance < 3:
            #     minGuardDistance *= 2
            features['guardDistance'] = minGuardDistance

        capsuleList = self.getCapsules(successor)
        if (len(capsuleList) > 0):
            minCapsuleDistance = min([self.getMazeDistance(myPos, c) for c in capsuleList])
            features['distanceToCapsule'] = minCapsuleDistance

        allies = [successor.getAgentState(i) for i in self.getTeam(successor)]
        if (len(allies) > 0):
            allyDists = [self.getMazeDistance(myPos, a.getPosition()) for a in allies]
            features['allyDistance'] = min(allyDists)

        # eat pill, then inverse guard for scaredghost timer
        # for a in enemies:
        #     if a.isScaredGhost() and self.getMazeDistance(myPos, a.getPosition()) < 6:
        #         features['scaredExists'] = 1

        # distance towards closest scared ghost
        # scaredGhosts = [a for a in enemies if a.isScaredGhost() and a.getPosition() is not None]
        # scaredDists = [self.getMazeDistance(myPos, a.getPosition()) for a in scaredGhosts]
        # if (len(scaredDists) > 0):
        #     minScaredDistance = min(scaredDists)
        #     if (minScaredDistance == 0):
        #         minScaredDistance = 0.001
        #     features['scaredDistance'] = self.reciprocal(minScaredDistance)

        return features

    def weightsGenerator(self, state, action):
        return {
            'successorScore': 100,
            'distanceToFood': -1,
            # 'numGuards': 0,
            'guardDistance': 0.75,
            'allyDistance': 0,
            # 'scaredDistance': -10,  # implement only if really close
            'distanceToCapsule': -0.2,  # -0.5 og
            # 'scaredExists': -3
        }

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """

        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()

        if (pos != util.nearestPoint(pos)):
            # Only half a grid position was covered.
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

# main defense agent that was implemented in final submission
class DefenseAgent(CaptureAgent):
    def __init__(self, index, **kwargs):
        super().__init__(index)

    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the agent and populates useful fields,
        such as the team the agent is on and the `pacai.core.distanceCalculator.Distancer`.

        IMPORTANT: If this method runs for more than 15 seconds, your agent will time out.
        """

        super().registerInitialState(gameState)

        # Your initialization code goes here, if you need any.

    def chooseAction(self, gameState):
        """
        Pick action based on evalFcn, default to random
        """
        actions = gameState.getLegalActions(self.index)

        start = time.time()
        values = [self.evalFcn(gameState, a) for a in actions]
        logging.debug('evaluate() time for agent %d: %.4f' % (self.index, time.time() - start))

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        return random.choice(bestActions)

    def evalFcn(self, state, action):
        weights = self.weightsGenerator(state, action)
        features = self.featureGenerator(state, action)
        return features * weights

    def featureGenerator(self, state, action):
        features = counter.Counter()
        successor = self.getSuccessor(state, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0).
        features['onDefense'] = 1
        if (myState.isPacman()):
            features['onDefense'] = 0

        # Computes distance to invaders we can see.
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman() and a.getPosition() is not None]
        features['numInvaders'] = len(invaders)

        if (len(invaders) > 0):
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        if (len(enemies) > 0):
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in enemies]
            features['enemyDistance'] = min(dists)

        if (action == Directions.STOP):
            features['stop'] = 1

        rev = Directions.REVERSE[state.getAgentState(self.index).getDirection()]
        if (action == rev):
            features['reverse'] = 1

        return features

    def weightsGenerator(self, state, action):
        return {
            'numInvaders': -1000,
            'onDefense': 150,
            'invaderDistance': -100,
            'stop': 0,
            'reverse': 0,
            'enemyDistance': -10
        }

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()

        if (pos != util.nearestPoint(pos)):
            # Only half a grid position was covered.
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

# Attempt at a hybrid attack/defense agent, not used due to poor performance

class HybridAgent1(CaptureAgent):
    def __init__(self, index, **kwargs):
        super().__init__(index)

    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the agent and populates useful fields,
        such as the team the agent is on and the `pacai.core.distanceCalculator.Distancer`.

        IMPORTANT: If this method runs for more than 15 seconds, your agent will time out.
        """

        super().registerInitialState(gameState)
        # Your initialization code goes here, if you need any.
        self.startingPosition = gameState.getAgentState(self.index).getPosition()
        self.defensive = False

    def chooseAction(self, gameState):
        """
        Pick action based on evalFcn, default to random
        """

        if gameState.getAgentState(self.index).getPosition() == self.startingPosition:
            self.defensive = not self.defensive
            print(self.defensive)

        actions = gameState.getLegalActions(self.index)

        start = time.time()
        values = [self.evalFcn(gameState, a) for a in actions]
        logging.debug('evaluate() time for agent %d: %.4f' % (self.index, time.time() - start))

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        return random.choice(bestActions)

    def evalFcn(self, state, action):
        weights = self.weightsGenerator(state, action)
        features = self.featureGenerator(state, action)
        return features * weights

    def featureGenerator(self, state, action):
        successor = self.getSuccessor(state, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()
        features = counter.Counter()

        features['successorScore'] = self.getScore(successor)

        foodList = self.getFood(successor).asList()
        if (len(foodList) > 0):
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = minDistance

        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        guards = [a for a in enemies if a.isBraveGhost() and a.getPosition() is not None]
        features['numGuards'] = len(guards)

        guardDists = [self.getMazeDistance(myPos, a.getPosition()) for a in guards]
        if (len(guardDists) > 0):
            minGuardDistance = min(guardDists)
            features['guardDistance'] = minGuardDistance

        capsuleList = self.getCapsules(successor)
        if (len(capsuleList) > 0):
            minCapsuleDistance = min([self.getMazeDistance(myPos, c) for c in capsuleList])
            features['distanceToCapsule'] = minCapsuleDistance

        if (action == Directions.STOP):
            features['stop'] = 1

        rev = Directions.REVERSE[state.getAgentState(self.index).getDirection()]
        if (action == rev):
            features['reverse'] = 1

        allies = [successor.getAgentState(i) for i in self.getTeam(successor)]
        if (len(allies) > 0):
            allyDists = [self.getMazeDistance(myPos, a.getPosition()) for a in allies]
            features['allyDistance'] = min(allyDists)

        scaredGhosts = [a for a in enemies if a.isScaredGhost() and a.getPosition() is not None]
        scaredDists = [self.getMazeDistance(myPos, a.getPosition()) for a in scaredGhosts]
        if (len(scaredDists) > 0):
            minScaredDistance = min(scaredDists)
            features['scaredDistance'] = minScaredDistance

        # Computes whether we're on defense (1) or offense (0).
        features['onDefense'] = 1
        if (myState.isPacman()):
            features['onDefense'] = 0

        # Computes distance to invaders we can see.
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman() and a.getPosition() is not None]
        features['numInvaders'] = len(invaders)

        if (len(invaders) > 0):
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        if (len(enemies) > 0):
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in enemies]
            features['enemyDistance'] = min(dists)

        return features

    def weightsGenerator(self, state, action):
        if self.defensive:
            print('is it working')
            return {
                # offensive
                'successorScore': 0,
                'distanceToFood': 0,
                'numGuards': 0,
                'guardDistance': 0,
                'allyDistance': 0,
                'scaredDistance': 0,  # implement only if really close
                'distanceToCapsule': 0,
                # defensive
                'onDefense': -1000,
                'invaderDistance': 100,
                'numInvaders': -100,
                'enemyDistance': -10,
                # misc
                'stop': 0,
                'reverse': 0
            }
        else:
            return {
                # offensive
                'successorScore': 100,
                'distanceToFood': -1,
                'numGuards': 0,
                'guardDistance': 0.7,
                'allyDistance': 0,
                'scaredDistance': -10,  # implement only if really close
                'distanceToCapsule': -0.5,
                # defensive
                'onDefense': 0,
                'invaderDistance': 0,
                'numInvaders': 0,
                'enemyDistance': 0,
                # misc
                'stop': 0,
                'reverse': 0
            }

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """

        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()

        if (pos != util.nearestPoint(pos)):
            # Only half a grid position was covered.
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

def createTeam(firstIndex, secondIndex, isRed,
        first = 'pacai.agents.capture.dummy.DummyAgent',
        second = 'pacai.agents.capture.dummy.DummyAgent'):
    """
    This function should return a list of two agents that will form the capture team,
    initialized using firstIndex and secondIndex as their agent indexed.
    isRed is True if the red team is being created,
    and will be False if the blue team is being created.
    """

    firstAgent = AttackAgent
    secondAgent = DefenseAgent

    return [
        firstAgent(firstIndex),
        secondAgent(secondIndex),
    ]
