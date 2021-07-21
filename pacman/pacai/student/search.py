"""
In this file, you will implement generic search algorithms which are called by Pacman agents.
"""
from pacai.util.stack import Stack
from pacai.util.queue import Queue
from pacai.util.priorityQueue import PriorityQueue

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first [p 85].

    Your search algorithm needs to return a list of actions that reaches the goal.
    Make sure to implement a graph search algorithm [Fig. 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    ```
    print("Start: %s" % (str(problem.startingState())))
    print("Is the start a goal?: %s" % (problem.isGoal(problem.startingState())))
    print("Start's successors: %s" % (problem.successorStates(problem.startingState())))
    ```
    """

    # *** Your Code Here ***
    # initialize structures
    stack = Stack()
    visited = []
    actionList = []

    # initialize stack and visited list with start node,
    # parent node, and previous action
    stack.push((problem.startingState(), None, None))
    visited.append(problem.startingState())

    while (not stack.isEmpty()):
        currentState, sourceState, prevAction = stack.pop()
        # trace back ancestors to form path if goal reached
        if problem.isGoal(currentState):
            actionList.append(prevAction)
            while sourceState:
                s, ss, p = sourceState
                actionList.append(p)
                sourceState = ss
            break
        # push succesor nodes to stack
        for (state, action, cost) in problem.successorStates(currentState):
            if state not in visited:
                stack.push((state, (currentState, sourceState, prevAction), action))
                visited.append(state)

    # reverse list to get start->goal
    # remove first filler action (None)
    list.reverse(actionList)
    actionList.pop(0)

    return actionList

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first. [p 81]
    """

    # *** Your Code Here ***
    # initialize structures
    queue = Queue()
    visited = []
    actionList = []

    # initialize queue and visited list with start node,
    # parent node, and previous action
    queue.push((problem.startingState(), None, None))
    visited.append(problem.startingState())

    while (not queue.isEmpty()):
        currentState, sourceState, prevAction = queue.pop()
        # trace back ancestors to form path if goal reached
        if problem.isGoal(currentState):
            actionList.append(prevAction)
            while sourceState:
                s, ss, p = sourceState
                actionList.append(p)
                sourceState = ss
            break
        # push succesor nodes to queue
        for (state, action, cost) in problem.successorStates(currentState):
            if state not in visited:
                queue.push((state, (currentState, sourceState, prevAction), action))
                visited.append(state)

    # reverse list to get start->goal
    # remove first filler action (None)
    list.reverse(actionList)
    actionList.pop(0)
    return actionList

def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """

    # *** Your Code Here ***
    # initialize structures
    pQueue = PriorityQueue()
    visited = []
    actionList = []

    # initialize pQueue and visited list with start node,
    # parent node, previous action, path cost, and priority=0
    pQueue.push((problem.startingState(), None, None, 0), 0)
    visited.append(problem.startingState())

    while (not pQueue.isEmpty()):
        (currentState, sourceState, prevAction, currentCost) = pQueue.pop()
        # trace back ancestors to form path if goal reached
        if problem.isGoal(currentState):
            actionList.append(prevAction)
            while sourceState:
                s, ss, p = sourceState
                actionList.append(p)
                sourceState = ss
            break
        # push succesor nodes to pQueue
        for (state, action, cost) in problem.successorStates(currentState):
            if state not in visited:
                pQueue.push((state, (currentState, sourceState, prevAction),
                    action, cost + currentCost), cost + currentCost)
                visited.append(state)

    # reverse list to get start->goal
    # remove first filler action (None)
    list.reverse(actionList)
    actionList.pop(0)
    return actionList

def aStarSearch(problem, heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """

    # *** Your Code Here ***
    # initialize structures
    pQueue = PriorityQueue()
    visited = []
    actionList = []

    # initialize pQueue and visited list with start node,
    # parent node, previous action, path cost, and priority=heuristic
    pQueue.push((problem.startingState(), None, None,
        0), heuristic(problem.startingState(), problem))
    visited.append(problem.startingState())

    while (not pQueue.isEmpty()):
        currentState, sourceState, prevAction, currentCost = pQueue.pop()
        # trace back ancestors to form path if goal reached
        if problem.isGoal(currentState):
            actionList.append(prevAction)
            while sourceState:
                s, ss, p = sourceState
                actionList.append(p)
                sourceState = ss
            break
        # push succesor nodes to pQueue, priority = path cost + heuristic
        for (state, action, cost) in problem.successorStates(currentState):
            if state not in visited:
                pQueue.push((state, (currentState, sourceState, prevAction), action,
                    cost + currentCost),
                    cost + currentCost + heuristic(state, problem))
                visited.append(state)

    # reverse list to get start->goal
    # remove first filler action (None)
    list.reverse(actionList)
    actionList.pop(0)
    return actionList
