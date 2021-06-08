from game import Game
import numpy as np
import torch


def set_xy(game: Game):
    x = torch.tensor(game.boardMat)
    y = torch.zeros(game.boardSize)
    if game.solution.shape[0] == 1:
        y[game.solution] = 1
    else:
        val = 1 / game.solution.shape[0]
        y[game.solution] = val

    return x, y


def get_inner_outer(game: Game, player: int):
    inner = torch.tensor(game.boardMat[game.playersVect == player])
    outer = torch.tensor(game.boardMat[game.playersVect != player])
    return inner, outer
