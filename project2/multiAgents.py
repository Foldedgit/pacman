# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        ' Evaluate the possible movements by the distance from foods, ghosts '

        newFood = newFood.asList() #remaining food dots
        minCost = float('inf')  #  min distance to closest food dot

        # find the nearest food dot
        for food in newFood:
            if manhattanDistance(newPos, food) < minCost:
                minCost = manhattanDistance(newPos, food)

        interest = 1 / (minCost + 0.1)  # value of the movement = reward

        for ghost in successorGameState.getGhostPositions():
            if manhattanDistance(ghost, newPos) < 1:  # will be eaten by ghost
                interest = -float('inf')              # penalty

            elif manhattanDistance(ghost, newPos) < 4:  # close to a ghost
                interest = - interest                   # penalty
                break

        return successorGameState.getScore() + interest

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        """
        if the state is a terminal state: return the state’s utility
        if the next agent is MAX: return max-value(state)
        if the next agent is MIN: return min-value(state)
        
        initialize v = -∞
        for each successor of state:
        v = max(v, value(successor))
        
        initialize v = +∞
        for each successor of state:
        v = min(v, value(successor))
        """
        def minMax(nAgent, state, depth):
            # go to the next layer and reset the number of agents
            if nAgent + 1 > state.getNumAgents():
                depth += 1
                nAgent = 0

            maxval = ["", -float('inf')]  # initialize maximum value
            minval = ["", float('inf')]  # initialize minimum value

            posbmov = state.getLegalActions(nAgent)  # All possible movements

            # no movement possible
            if not posbmov:
                return self.evaluationFunction(state)
            # set depth limit
            if depth == self.depth:
                return self.evaluationFunction(state)

            # calculate minval and maxval for all nodes corresponding possible movement, agents and ghosts
            for step in posbmov:
                cstate = state.generateSuccessor(nAgent, step)  # get the current successor game state

                cval = minMax(nAgent + 1, cstate, depth)  # get current min or max value
                try:
                    cval = cval[1]
                except:
                    cval = cval
                # compare current min/max value with previous ones
                if cval > maxval[1] and nAgent == 0:
                    maxval = [step, cval]
                    continue
                if cval < minval[1]:
                    minval = [step, cval]
            # return the result
            if nAgent == 0:
                return maxval
            return minval

        return minMax(0, gameState, 0)[0]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        """
        α: MAX’s best option on path to root
        β: MIN’s best option on path to root
        
        initialize v = -∞
        for each successor of state:
        v = max(v, value(successor, α, β))
        if v ≥ β return v
        α = max(α, v)
        
        initialize v = +∞
        for each successor of state:
        v = min(v, value(successor, α, β))
        if v ≤ α return v
        β = min(β, v)
        """
        def alphaBeta(nAgent, state, depth, alpha, beta):
            # go to the next layer and reset the number of agents
            if nAgent + 1 > state.getNumAgents():
                depth += 1
                nAgent = 0

            maxval = ["", -float('inf')]  # initialize maximum value
            minval = ["", float('inf')]  # initialize minimum value

            posbmov = state.getLegalActions(nAgent)  # All possible movements

            # no movement possible
            if not posbmov:
                return self.evaluationFunction(state)
            # set depth limit
            if depth == self.depth:
                return self.evaluationFunction(state)

            # calculate minval and maxval for all nodes corresponding possible movement, agents and ghosts
            for step in posbmov:
                cstate = state.generateSuccessor(nAgent, step)  # get the current successor game state

                cval = alphaBeta(nAgent + 1, cstate, depth, alpha, beta)  # get current min or max value
                try:
                    cval = cval[1]
                except:
                    cval = cval
                # compare current min/max value with previous ones
                if nAgent == 0:
                    if cval > maxval[1]:
                        maxval = [step, cval]
                    if cval > beta:
                        return [step, cval]
                    alpha = max(alpha, cval)
                    continue
                if cval < minval[1]:
                    minval = [step, cval]
                if cval < alpha:
                    return [step, cval]
                beta = min(beta, cval)
            # return the result
            if nAgent == 0:
                return maxval
            return minval

        return alphaBeta(0, gameState, 0, -float('inf'), float('inf'))[0]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        """
        if the state is a terminal state: return the state’s utility
        if the next agent is MAX: return max-value(state)
        if the next agent is EXP: return exp-value(state)
        
        initialize v = -∞
        for each successor of state:
        v = max(v, value(successor))
        
        initialize v = 0
        for each successor of state:
        p = probability(successor)
        v += p * value(successor)
        """
        def expectimax(nAgent, state, depth):
            # go to the next layer and reset the number of agents
            if nAgent + 1 > state.getNumAgents():
                depth += 1
                nAgent = 0

            posbmov = state.getLegalActions(nAgent)     # All possible movements
            # no movement possible
            if not posbmov:
                return self.evaluationFunction(state)
            maxval = ["", -float('inf')]        # initialize maximum value
            expcval = ["", 0]                   # initialize expected value
            prob = 1.0 / len(posbmov)
            # set depth limit
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            # calculate expected value and maxval for all nodes corresponding possible movement, agents and ghosts
            for step in posbmov:
                cstate = state.generateSuccessor(nAgent, step)      # get the current successor game state
                cval = expectimax(nAgent + 1, cstate, depth)        # get current expectimax value

                try:
                    cval = cval[1]
                except:
                    cval = cval
                # compare current expectimax value with previous ones
                if cval > maxval[1] and nAgent == 0:
                    maxval = [step, cval]
                    continue
                expcval[0] = step
                expcval[1] += cval * prob
            # return the result
            if nAgent == 0:
                return maxval
            return expcval

        return expectimax(0, gameState, 0)[0]

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
