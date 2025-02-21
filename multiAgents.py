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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, but please don't change the method
    headers.
    """


    def getAction(self, gameState: GameState):
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        # print(newPos)
        # print(newFood)
        # print(newScaredTimes)

        # Consider reducing the distance to the nearest food
        foods = newFood.asList()
        if len(foods) == 0:
            minFoodDistance = 0
        else:
            minFoodDistance = min([manhattanDistance(newPos, food) for food in foods])
        
        # Consider increasing the distance to the nearest ghost
        ghostDistances = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
        minGhostDistance = min(ghostDistances)

        # if the ghost is not scared and too close, return a low score
        if minGhostDistance < 3 and min(newScaredTimes) == 0:
            return -1000

        # else, return a score that is related to the distance to the nearest food
        score = successorGameState.getScore() + (1.0 / float(minFoodDistance + 1))

        # return successorGameState.getScore()
        return score

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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
        # util.raiseNotDefined()

        # get the number of agents in the game
        numAgents = gameState.getNumAgents()

        # define the recursive minimax function
        def minimax(agentIndex, depth, gameState):
            # check if win, lose or reached the depth limit
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            
            # get a list of legal moves for the current agent
            legalActions = gameState.getLegalActions(agentIndex)

            if agentIndex == 0:
                # maximizing player(pacman's turn)
                return maxValue(agentIndex, depth, gameState, legalActions)
            else:  
                # minimizing player(ghosts' turn)
                return minValue(agentIndex, depth, gameState, legalActions)
            
        def maxValue(agentIndex, depth, gameState, legalActions):
            value = float("-inf")
            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                value = max(value, minimax(agentIndex + 1, depth, successor))
            return value
        
        def minValue(agentIndex, depth, gameState, legalActions):
            value = float("inf")
            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                nextAgent = (agentIndex + 1) % gameState.getNumAgents()
                nextDepth = depth + 1 if nextAgent == 0 else depth
                value = min(value, minimax(nextAgent, nextDepth, successor))
            return value

        # get a list of legal moves for Pacman
        legalActions = gameState.getLegalActions(0)
        bestValue = -float('inf')
        bestAction = None

        # find the best action for Pacman
        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            # first call to the minimax function
            value = minimax(1, 0, successor)
            if value > bestValue:
                bestValue = value
                bestAction = action

        return bestAction




class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()

        def alphaBetaPruning(agentIndex, depth, gameState, alpha, beta):
            # check if win, lose or reached the depth limit
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            
            # get a list of legal moves for the current agent
            legalActions = gameState.getLegalActions(agentIndex)

            if agentIndex == 0:
                # maximizing player(pacman's turn)
                return maxValue(agentIndex, depth, gameState, alpha, beta, legalActions)
            else:  
                # minimizing player(ghosts' turn)
                return minValue(agentIndex, depth, gameState, alpha, beta, legalActions)
            
        def maxValue(agentIndex, depth, gameState, alpha, beta, legalActions):
            value = float("-inf")
            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                value = max(value, alphaBetaPruning(agentIndex + 1, depth, successor, alpha, beta))

                if value > beta:
                    # find a value greater than beta, prune the rest of the branch
                    return value
                # update alpha
                alpha = max(alpha, value)

            return value
        
        def minValue(agentIndex, depth, gameState, alpha, beta, legalActions):
            value = float("inf")
            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                nextAgent = (agentIndex + 1) % gameState.getNumAgents()
                nextDepth = depth + 1 if nextAgent == 0 else depth
                value = min(value, alphaBetaPruning(nextAgent, nextDepth, successor, alpha, beta))

                if value < alpha:
                    # find a value less than alpha, prune the rest of the branch
                    return value
                # update beta
                beta = min(beta, value)

            return value

        
        # get a list of legal moves for Pacman
        legalActions = gameState.getLegalActions(0)

        alpha = float("-inf")
        beta = float("inf")
        bestValue = -float('inf')
        bestAction = None

        # find the best action for Pacman
        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            # first call to the alphaBetaPruning function
            value = alphaBetaPruning(1, 0, successor, alpha, beta)

            if value > bestValue:
                bestValue = value
                bestAction = action

            # update alpha
            alpha = max(alpha, bestValue)
            # if the best value is greater than beta, directly return the best action
            if bestValue > beta:
                return bestAction

        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()

        def expectiMax(agentIndex, depth, gameState):
            # check if win, lose or reached the depth limit
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            
            # get a list of legal moves for the current agent
            legalActions = gameState.getLegalActions(agentIndex)

            if agentIndex == 0:
                # maximizing player(pacman's turn)
                return maxValue(agentIndex, depth, gameState, legalActions)
            else:  
                # minimizing player(ghosts' turn)
                return expectValue(agentIndex, depth, gameState, legalActions)
        
        def maxValue(agentIndex, depth, gameState, legalActions):
            value = float("-inf")
            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                value = max(value, expectiMax(agentIndex + 1, depth, successor))
            return value
        
        def expectValue(agentIndex, depth, gameState, legalActions):
            value = 0
            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                nextAgent = (agentIndex + 1) % gameState.getNumAgents()
                nextDepth = depth + 1 if nextAgent == 0 else depth
                value += expectiMax(nextAgent, nextDepth, successor)
            return value / len(legalActions)
        
        # get a list of legal moves for Pacman
        legalActions = gameState.getLegalActions(0)
        bestValue = -float('inf')
        bestAction = None
        
        # find the best action for Pacman
        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            # first call to the minimax function
            value = expectiMax(1, 0, successor)
            if value > bestValue:
                bestValue = value
                bestAction = action

        return bestAction
        

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    # The function return a score related to following factors:
    # 1. The score of the current game state (base score)
    # 2. The number of food left (the fewer food left, the higher the score)
    # 3. The distance to the nearest food (the closer to the food, the higher the score)
    # 4. The distance to the nearest ghost (the closer to the ghost, the lower the score)
    # The evaluation function returns a score that is related to the above factors
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()

    # get the current position of pacman
    pacmanPosition = currentGameState.getPacmanPosition()

    # get the current score
    score = currentGameState.getScore()

    # consider the food factor
    foods = currentGameState.getFood().asList()

    # the fewer food left, the higher the score
    score -= len(foods)

    # if there is food left, more closer to the food, the higher the score
    if len(foods) > 0:
        # get the distance to the nearest food
        minFoodDistance = min([manhattanDistance(pacmanPosition, food) for food in foods])
        score += 1.0 / float(minFoodDistance + 1.0)
        
    # consider the ghost factor 
    # get the current ghost states
    ghostStates = currentGameState.getGhostStates() 
    # get the distance to the nearest ghost
    ghostDistances = [manhattanDistance(pacmanPosition, ghost.getPosition()) for ghost in ghostStates]

    # consider the scared time of the ghosts
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    for i in range(len(ghostStates)):
        if scaredTimes[i] > 0:
            # If the ghost is scared, the closer to the ghost, the higher the score
            score += 1.0 / (ghostDistances[i] + 1.0)
        else:
            # If the ghost is not scared, the closer to the ghost, the lower the score
            score -= 1.0 / (ghostDistances[i] + 1.0)

    return score


# Abbreviation (don't modify existing, but you can add to them)
better = betterEvaluationFunction
score = scoreEvaluationFunction
