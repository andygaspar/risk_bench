from game import Game
import agent
import torch

NUM_PLAYERS = 7
BOARD_SIZE = 40

game = Game()
game.set_game(NUM_PLAYERS, BOARD_SIZE)

inner, outer = agent.get_inner_outer(game, game.players[0])

x, y = agent.set_xy(game)
