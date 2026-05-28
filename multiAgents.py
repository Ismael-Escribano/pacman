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

import torch
import numpy as np
from net import PacmanNet
import os
from util import manhattanDistance
from game import Directions
import random, util
random.seed(42)  # For reproducibility
from game import Agent
from pacman import GameState
from enum import Enum

class Evaluation(Enum):
    neural_and_heuristics = 0
    neural = 1
    heuristics = 2

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
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
        return successorGameState.getScore()

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
        def minimax(agentIndex, depth, gameState):
            """
            Recursive minimax function
            
            Args:
            - agentIndex: Current agent (0=Pacman, 1+=Ghosts)  
            - depth: Current depth in the game tree
            - gameState: Current state of the game
            
            Returns:
            - Best evaluation score for this state
            """
            # Base case: terminal state or maximum depth reached
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            # Pacman's turn (Maximizer)
            if agentIndex == 0:
                return maxValue(agentIndex, depth, gameState)
            # Ghost's turn (Minimizer)  
            else:
                return minValue(agentIndex, depth, gameState)
        
        def maxValue(agentIndex, depth, gameState):
            """
            Handles Pacman's moves (maximizing player)
            """
            v = float('-inf')  # Start with worst possible value
            legalActions = gameState.getLegalActions(agentIndex)
            
            # No legal actions available
            if not legalActions:
                return self.evaluationFunction(gameState)

            # Try each possible action and choose the best
            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                # After Pacman moves, first ghost plays (agent 1)
                v = max(v, minimax(1, depth, successor))
            return v

        def minValue(agentIndex, depth, gameState):
            """
            Handles Ghost moves (minimizing players)
            """
            v = float('inf')  # Start with best possible value for Pacman
            legalActions = gameState.getLegalActions(agentIndex)
            
            # No legal actions available
            if not legalActions:
                return self.evaluationFunction(gameState)

            # Determine next agent and depth
            nextAgent = agentIndex + 1
            nextDepth = depth
            
            # If all ghosts have moved, return to Pacman and increment depth
            if nextAgent == gameState.getNumAgents():
                nextAgent = 0      # Back to Pacman
                nextDepth = depth + 1  # New ply begins

            # Try each possible action and choose the worst for Pacman
            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                v = min(v, minimax(nextAgent, nextDepth, successor))
            return v

        # Main decision logic for Pacman
        bestAction = None
        bestScore = float('-inf')

        # Try each legal action for Pacman
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            # Start minimax with first ghost (agent 1) at current depth
            score = minimax(1, 0, successor)
            
            if score > bestScore:
                bestScore = score
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
        def alphabeta(gameState, depth, alpha, beta, agentIndex):
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            
            if agentIndex == 0: # Turno de Pacman (MAX)
                max_eval = float('-inf')
                actions = gameState.getLegalActions(agentIndex)
                if not actions:
                    return self.evaluationFunction(gameState)
                for action in actions:
                    successor = gameState.generateSuccessor(agentIndex, action)
                    eval_score = alphabeta(successor, depth, alpha, beta, agentIndex+1)
                    max_eval = max(max_eval, eval_score)
                    alpha = max(alpha, eval_score)
                    if beta <= alpha:
                        break
                return max_eval
            
            else: # Turno de los fantasmas (MIN)
                min_eval = float('inf')
                actions = gameState.getLegalActions(agentIndex)
                if not actions:
                    return self.evaluationFunction(gameState)
                
                nextAgent = agentIndex + 1
                nextDepth = depth
                if nextAgent == gameState.getNumAgents():
                    nextAgent = 0
                    nextDepth = depth + 1
                
                for action in actions:
                    successor = gameState.generateSuccessor(agentIndex, action)
                    eval_score = alphabeta(successor, nextDepth, alpha, beta, nextAgent)
                    min_eval = min(min_eval, eval_score)
                    beta = min(beta, eval_score)
                    if beta <= alpha:
                        break
                return min_eval
        bestAction = None
        bestScore = float('-inf')
        for action in gameState.getLegalActions(0):
            succesor = gameState.generateSuccessor(0, action)
            score = alphabeta(succesor, 0, float('-inf'), float('+inf'), 1)
            if score > bestScore:
                bestScore = score
                bestAction = action
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
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction


###########################################################################
# Ahmed
###########################################################################

class NeuralAgent(Agent):
    """
    Un agente de Pacman que utiliza una red neuronal para tomar decisiones
    basado en la evaluación del estado del juego.
    """
    def __init__(self, model_path="models/pacman_model.pth"):
        super().__init__()
        self.model = None
        self.input_size = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model(model_path)
        
        # Mapeo de índices a acciones
        self.idx_to_action = {
            0: Directions.STOP,
            1: Directions.NORTH,
            2: Directions.SOUTH,
            3: Directions.EAST,
            4: Directions.WEST
        }
        
        # Para evaluar alternativas
        self.action_to_idx = {v: k for k, v in self.idx_to_action.items()}
        
        # Contador de movimientos
        self.move_count = 0
        
        print(f"NeuralAgent inicializado, usando dispositivo: {self.device}")

    def load_model(self, model_path):
        """Carga el modelo desde el archivo guardado"""
        try:
            if not os.path.exists(model_path):
                print(f"ERROR: No se encontró el modelo en {model_path}")
                return False
                
            # Cargar el modelo
            checkpoint = torch.load(model_path, map_location=self.device)
            self.input_size = checkpoint['input_size']
            
            # Crear y cargar el modelo
            self.model = PacmanNet(self.input_size, 128, 5).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()  # Modo evaluación
            
            print(f"Modelo cargado correctamente desde {model_path}")
            print(f"Tamaño de entrada: {self.input_size}")
            return True
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            return False

    def state_to_matrix(self, state):
        """Convierte el estado del juego en una matriz numérica normalizada"""
        # Obtener dimensiones del tablero
        walls = state.getWalls()
        width, height = walls.width, walls.height
        
        # Crear una matriz numérica
        # 0: pared, 1: espacio vacío, 2: comida, 3: cápsula, 4: fantasma, 5: Pacman
        numeric_map = np.zeros((width, height), dtype=np.float32)
        
        # Establecer espacios vacíos (todo lo que no es pared comienza como espacio vacío)
        for x in range(width):
            for y in range(height):
                if not walls[x][y]:
                    numeric_map[x][y] = 1
        
        # Agregar comida
        food = state.getFood()
        for x in range(width):
            for y in range(height):
                if food[x][y]:
                    numeric_map[x][y] = 2
        
        # Agregar cápsulas
        for x, y in state.getCapsules():
            numeric_map[x][y] = 3
        
        # Agregar fantasmas
        for ghost_state in state.getGhostStates():
            ghost_x, ghost_y = int(ghost_state.getPosition()[0]), int(ghost_state.getPosition()[1])
            # Si el fantasma está asustado, marcarlo diferente
            if ghost_state.scaredTimer > 0:
                numeric_map[ghost_x][ghost_y] = 6  # Fantasma asustado
            else:
                numeric_map[ghost_x][ghost_y] = 4  # Fantasma normal
        
        # Agregar Pacman
        pacman_x, pacman_y = state.getPacmanPosition()
        numeric_map[int(pacman_x)][int(pacman_y)] = 5
        
        # Normalizar
        numeric_map = numeric_map / 6.0
        
        return numeric_map

    def evaluationFunction(self, state: GameState, evaluation: Evaluation = Evaluation.neural_and_heuristics):
        """
        Una función de evaluación basada en la red neuronal y en heurísticas adicionales.
        """
        if self.model is None:
            return 0  # Si no hay modelo, devolver 0
        
        # Obtener acciones legales
        legal_actions = state.getLegalActions()
        
        # Aplicar heurísticas adicionales, similar a betterEvaluationFunction
        score = state.getScore()

        if evaluation == Evaluation.neural or evaluation == Evaluation.neural_and_heuristics:
            # Convertir a matriz
            state_matrix = self.state_to_matrix(state)
            
            # Convertir a tensor
            state_tensor = torch.FloatTensor(state_matrix).unsqueeze(0).to(self.device)
            
            # Obtener predicciones
            with torch.no_grad():
                output = self.model(state_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]
        
        if evaluation == Evaluation.heuristics or evaluation == Evaluation.neural_and_heuristics:            
            # Mejorar la evaluación con conocimiento del dominio
            pacman_pos = state.getPacmanPosition()
            food = state.getFood().asList()
            ghost_states = state.getGhostStates()
            capsules = state.getCapsules()
            
            # Factor 1: Distancia a la comida más cercana
            if food:
                min_food_distance = min(manhattanDistance(pacman_pos, food_pos) for food_pos in food)
                score += 5 / (min_food_distance + 1)
            
            # Factor 2: Proximidad a fantasmas
            for ghost_state in ghost_states:
                ghost_pos = ghost_state.getPosition()
                ghost_distance = manhattanDistance(pacman_pos, ghost_pos)
                
                if ghost_state.scaredTimer > 0: # Si el fantasma está asustado
                    if ghost_distance <= 2:
                        score -= 200 
                    elif ghost_distance > ghost_state.scaredTimer:
                        score -= 50 / (ghost_distance + 1) # Cuanto más lejos esté el fantasma asustado, mejor score
                else:
                    # Si no está asustado, evitarlo
                    score -= 100 / (ghost_distance + 1)
            
            
            # Factor 3: comer más cuando el fantasma está asustado
            times = []
            for ghost_state in ghost_states:
                times.append(ghost_state.scaredTimer)
            
            if all(times) and food:
                min_food_distance = min(manhattanDistance(pacman_pos, food_pos) for food_pos in food)
                score += 200 / (min_food_distance + 1)

            # Factor 4: comer cápsulas cuando los fantasmas están cerca
            if capsules:
                min_capsule_distance = min(manhattanDistance(pacman_pos, capsule_pos) for capsule_pos in capsules)
                fantasmas_cerca = False
                for ghost_state in ghost_states:
                    ghost_pos = ghost_state.getPosition()
                    ghost_distance = manhattanDistance(pacman_pos, ghost_pos)

                    if ghost_distance <= 4 and ghost_state.scaredTimer == 0:
                        fantasmas_cerca = True
                        break
                    
                if fantasmas_cerca:
                    if min_capsule_distance <= 3:
                        score += 50 / (min_capsule_distance + 1)
                else:
                    if min_capsule_distance <= 3:
                        score += 5 / (min_capsule_distance + 1)

            # Factor 5: comer más cuando hay poca comida
            if food and len(food) <= 15:
                min_food_distance = min(manhattanDistance(pacman_pos, food_pos) for food_pos in food)
                score += 200 / (min_food_distance + 1)
            
            # Factor 6: penalizar comida lejana
            if food:
                min_food_distance = min(manhattanDistance(pacman_pos, food_pos) for food_pos in food)
                if min_food_distance >= 8:
                    score -= 5 * min_food_distance


        if evaluation == Evaluation.neural or evaluation == Evaluation.neural_and_heuristics:
            # Combinar la puntuación de la red con la heurística
            neural_score = 0
            for i, action in enumerate(self.idx_to_action.values()):
                if action in legal_actions:
                    neural_score += probabilities[i] * 100

        match evaluation:
            case Evaluation.neural_and_heuristics:
                return score + neural_score
            case Evaluation.neural:
                return neural_score
            case Evaluation.heuristics:
                return score

    def getAction(self, state):
        """
        Devuelve la mejor acción basada en la evaluación de la red neuronal
        y heurísticas adicionales.
        """
        self.move_count += 1
        
        # Si no hay modelo, hacer un movimiento aleatorio
        if self.model is None:
            print("ERROR: Modelo no cargado. Haciendo movimiento aleatorio.")
            exit()
            legal_actions = state.getLegalActions()
            return random.choice(legal_actions)
        
        # Obtener acciones legales
        legal_actions = state.getLegalActions()
        
        # Evaluación directa con la red neuronal
        state_matrix = self.state_to_matrix(state)
        state_tensor = torch.FloatTensor(state_matrix).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(state_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]
        
        # Mapear índices del modelo a acciones del juego
        action_probs = []
        for idx, prob in enumerate(probabilities):
            action = self.idx_to_action[idx]
            if action in legal_actions:
                action_probs.append((action, prob))
        
        # Ordenar por probabilidad (mayor a menor)
        action_probs.sort(key=lambda x: x[1], reverse=True)
        
        # Exploración: con una probabilidad decreciente, elegir aleatoriamente
        exploration_rate = 0.2 * (0.99 ** self.move_count)  # Disminuye con el tiempo
        if random.random() < exploration_rate:
            # Excluir STOP si es posible
            if len(legal_actions) > 1 and Directions.STOP in legal_actions:
                legal_actions.remove(Directions.STOP)
            return random.choice(legal_actions)
        
        # Evaluación alternativa: generar sucesores y evaluar cada uno
        successors = []
        for action in legal_actions:
            successor = state.generateSuccessor(0, action)
            eval_score = self.evaluationFunction(successor)
            neural_score = 0
            for a, p in action_probs:
                if a == action:
                    neural_score = p * 100
                    break
            # Combinar evaluación heurística con la predicción de la red
            combined_score = eval_score + neural_score
            
            # Penalizar STOP a menos que sea la única opción
            if action == Directions.STOP and len(legal_actions) > 1:
                combined_score -= 50
                
            successors.append((action, combined_score))
        
        # Ordenar por puntuación combinada
        successors.sort(key=lambda x: x[1], reverse=True)
        
        # Devolver la mejor acción
        return successors[0][0]

# Definir una función para crear el agente
def createNeuralAgent(model_path="models/pacman_model.pth"):
    """
    Función de fábrica para crear un agente neuronal.
    Útil para integrarse con la estructura de pacman.py.
    """
    return NeuralAgent(model_path)


class AlphaBetaNeuralAgent(NeuralAgent):
    def __init__(self, model_path="models/pacman_model.pth", depth = '4', w_trad = 0.25, w_neural = 0.75):
        super().__init__(model_path)

        self.depth = int(depth)
        
        self.w_trad = w_trad
        self.start_trad = w_trad
        self.end_trad = 1 - w_trad

        self.w_neural = w_neural
        self.start_neural = w_neural
        self.end_neural = 1 - w_neural

        self.total_food = None

    def combined_evaluation(self, state):
        trad_score = self.evaluationFunction(state, evaluation=Evaluation.heuristics)

        neural_score = self.evaluationFunction(state, evaluation=Evaluation.neural)
        
        return self.w_trad * trad_score + self.w_neural * neural_score

    def alphabeta(self, gameState, depth, alpha, beta, agentIndex):
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.combined_evaluation(gameState)
        
        if agentIndex == 0: # Turno de Pacman (MAX)
            max_eval = float('-inf')
            actions = gameState.getLegalActions(agentIndex)
            if not actions:
                return self.combined_evaluation(gameState)
            for action in actions:
                successor = gameState.generateSuccessor(agentIndex, action)
                eval_score = self.alphabeta(successor, depth, alpha, beta, agentIndex+1)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval
        
        else: # Turno de los fantasmas (MIN)
            min_eval = float('inf')
            actions = gameState.getLegalActions(agentIndex)
            if not actions:
                return self.combined_evaluation(gameState)
            
            nextAgent = agentIndex + 1
            nextDepth = depth
            if nextAgent == gameState.getNumAgents():
                nextAgent = 0
                nextDepth = depth + 1
            
            for action in actions:
                successor = gameState.generateSuccessor(agentIndex, action)
                eval_score = self.alphabeta(successor, nextDepth, alpha, beta, nextAgent)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval
    
    def getAction(self, state: GameState):
        best_score = float('-inf')
        best_action = None
        for action in state.getLegalActions(0):
            succesor = state.generateSuccessor(0, action)
            score = self.alphabeta(succesor, 0, float('-inf'), float('+inf'), 1)
            if action == Directions.STOP:
                score -= 200
            if score > best_score:
                best_score = score
                best_action = action
        # Obtenemos la comida total en el primer turno
        if self.total_food is None:
            self.total_food = state.getNumFood()
        
        # Obtenemos el progreso del juego (% de comida por comer)
        current_food = state.getNumFood()
        game_progress = current_food / self.total_food

        # Modificamos los pesos usando interpolación lineal
        self.w_neural = self.start_neural + (self.end_neural - self.start_neural) * game_progress
        self.w_trad = self.start_trad + (self.end_trad - self.start_trad) * game_progress
        return best_action