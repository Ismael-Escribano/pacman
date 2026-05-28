import layout
import os
import random
import contextlib
from pathlib import Path
from dataclasses import dataclass

from pacman import ClassicGameRules
from multiAgents import AlphaBetaNeuralAgent
from ghostAgents import DirectionalGhost
import textDisplay
import gamedata

######################################################
#               Configuración general
######################################################

LAYOUT_NAME = 'mediumClassic'                # Nombre del laberinto
DEPTH = 4                               # Profundidad máxima de Alpha-Beta
SEED_START = 0                          # Primera semilla
SEED_END = 10                            # Última semilla
OUTPUT_DIR_NAME = 'runner_data'              # Carpeta para guardar CSV
TIMEOUT = 300                           # Tiempo máximo por partida
START_W_TRAD = 0.25                     # Peso inicial para heurísticas tradicionales
START_W_NEURAL = 0.75                   # Peso inicial para la red neuronal


OUTPUT_DIR = Path(OUTPUT_DIR_NAME)
LAYOUT = layout.getLayout(LAYOUT_NAME)

######################################################
#               Funciones auxiliares
######################################################

@dataclass
class GameData:
    """Clase para guardar de forma sencilla los datos de una partida"""
    score: float
    win: bool
    seed: int
    csv_path: Path

def mostrar_progreso(start: int, end: int, it: int, length: int = 40) -> None:
    """Muestra una barra de progreso"""
    total = end - start
    actual = it - start + 1

    relleno = length * actual // total

    barra = '█' * relleno + '-' * (length - relleno)

    print(f'\r |{barra}| {actual} / {total}', end="\r", flush=True)
    if actual == total:
        print()

######################################################
#               Funciones principales
######################################################

def run_game(seed: int) -> GameData:
    """
    Ejecuta una partida con la semilla indicada

    Parametros
    ----------
    seed: int
        Semilla a utilizar para la partida
    
    Retorna
    -------
    GameData
        Datos de la partida, si ocurre algun error, los datos
        no conseguidos son 'None'
    """
    # Fijamos la semilla
    random.seed(str(seed))

    # Creamos el agente Pacman (con la salida estándar suprimida)
    with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
        pacman = AlphaBetaNeuralAgent(
                depth=str(DEPTH),
                w_trad=START_W_TRAD,
                w_neural=START_W_NEURAL,
        )

    # Creamos los fantasmas
    ghosts = [DirectionalGhost(i) for i in range(1, LAYOUT.getNumGhosts() + 1)]

    # Creamos el juego de sin gráficos
    rules = ClassicGameRules(timeout=TIMEOUT)
    rules.quiet = True

    game = rules.newGame(LAYOUT, pacman, ghosts, textDisplay.NullGraphics(), quiet=True, catchExceptions=True)

    # Crear colector de datos
    data_collector = gamedata.GameDataCollector(output_dir=OUTPUT_DIR)
    game.data_collector = data_collector

    # Ejecución de la partida
    try:
        game.run()
    except Exception as e:
        print(f'La partida con {seed=} Falló:\n{str(e)}')
        return GameData(None, False, seed, None)

    # Obtenemos los datos finales y lo guardamos a CSV
    final_score = game.state.getScore()
    win = game.state.isWin()

    with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
        data_collector.save_game_data(seed)

    # Obtenemos el último CSV creado
    files = [f for f in OUTPUT_DIR.iterdir() if f.suffix == '.csv']
    if not files:
        csv_path = None
    else:
        csv_path = max(files, key=lambda x: x.stat().st_mtime)

    return GameData(final_score, win, seed, csv_path)


def main() -> tuple[list[GameData], GameData]:
    """
    Realiza el bucle de partidas según la configuración dada

    Retorna
    -------
    tuple[list[GameData], GameData]
        Lista de todas las partidas y la mejor partida aparte
    """
    games = []

    max_score = float('-inf')
    print(f'Ejecutando {SEED_END - SEED_START} partidas...')
    mostrar_progreso(SEED_START, SEED_END, SEED_START - 1)
    try:
        for seed in range(SEED_START, SEED_END):
            game = run_game(seed)

            games.append(game)

            if game.score > max_score:
                max_score = game.score
                best_game = game

            mostrar_progreso(SEED_START, SEED_END, seed)
    except KeyboardInterrupt:
        return games, best_game
    else:
        return games, best_game


def summary(games: list[GameData], best_game: GameData) -> None:
    """
    Muestra los resultados de todas las partidas:
        - Resultados individuales
        - Resultados globales
        - Mejor partida encontrada
    
    Parametros
    ----------
    games: List[GameData]
        Lista con todas las partidas jugadas
    best_game: GameData
        Mejor partida encontrada
    """
    print(f'\nResultados de {len(games)} partidas:')
    scores = ' '.join([str(g.score) for g in games])
    wins = ' '.join(['WIN' if g.win else 'LOSE' for g in games])
    seeds = ' '.join([str(g.seed) for g in games])

    print(f'Scores: {scores}')
    print(f'Wins: {wins}')
    print(f'Seeds: {seeds}')

    print(f'Puntuación media: {sum(g.score for g in games)/len(games):.2f}')
    print(f'% de victorias: {sum(g.win for g in games)/len(games) * 100:.2f}%\n')

    print(f'MEJOR PARTIDA:\n\tScore: {best_game.score}\n\tSeed: {best_game.seed}\n\tWin: {best_game.win}\n\tCSV file: {best_game.csv_path}')
    print(f'COMANDO PARA VER LA MEJOR PARTIDA:\n---->\tpython pacman.py -l {LAYOUT_NAME} --csv {best_game.csv_path} -s {best_game.seed}')

if __name__ == "__main__":
    games, best_game = main()
    summary(games, best_game)