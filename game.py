import numpy as np



class Game:
    def __init__(self):
        self.players = None
        self.playersVect = None
        self.board = None
        self.boardMat = None

        self.numPlayers = None
        self.boardSize = None
        self.solution = None

    def set_game(self, num_players, board_size):
        self.numPlayers = num_players
        self.boardSize = board_size
        self.playersVect = np.random.choice(range(num_players), size=board_size)
        self.players = np.unique(self.playersVect)
        self.board = np.random.choice(range(1, 10), size=board_size)
        self.boardMat = np.diag(self.board)
        self.solution = self.solve()

    def solve(self):
        unique, counts = np.unique(self.playersVect, return_counts=True)
        minimal_players = unique[np.nonzero(counts == np.min(counts))[0]]
        if len(minimal_players) > 1:
            sums = [np.sum(self.board[np.nonzero(self.playersVect == i)[0]]) for i in minimal_players]
            minimal_players = minimal_players[np.nonzero(sums == np.min(sums))[0]]

        return np.hstack([np.nonzero((self.board == np.min(self.board[self.playersVect == i])) * (self.playersVect == i))[0]
                         for i in minimal_players])








