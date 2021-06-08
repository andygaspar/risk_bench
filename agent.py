from game import Game
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


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


sMax = nn.Softmax(dim=1)


class attentionNet(nn.Module):
    def __init__(self, hidden_dim, discretisation_size, n_airlines, n_trades, l_rate,
                 weight_decay=1e-4):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        self.hidden_dim = hidden_dim
        self.discretisationSize = discretisation_size
        self.numAirlines = n_airlines
        self.singleFlightSize = self.discretisationSize + self.numAirlines

        self.loss = 0
        self.bestLoss = 100_000_000

        torch.cuda.current_device()
        print("Running on GPU:", torch.cuda.is_available())

        self.action_embedding = nn.Sequential(nn.Linear(self.singleFlightSize, self.hidden_dim, bias=False),
                                              nn.ReLU(),
                                              nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
                                              nn.ReLU(),
                                              nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)).to(self.device)

        self.schedule_embedding = nn.Sequential(nn.Linear(self.singleFlightSize, self.hidden_dim, bias=False),
                                                nn.ReLU(),
                                                nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
                                                nn.ReLU(),
                                                nn.Linear(self.hidden_dim, 1, bias=False)).to(self.device)

        # self.value_net = nn.Sequential(nn.Linear(3*self.hidden_dim, self.hidden_dim),
        self.value_net = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim * 2),
                                       nn.ReLU(),
                                       nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                                       nn.ReLU(),
                                       nn.Linear(self.hidden_dim, 16),
                                       nn.ReLU(),
                                       nn.Linear(16, 16),
                                       nn.ReLU(),
                                       nn.Linear(16, 1)).to(self.device)

        # self.value_net[-1].weight.data = torch.abs(self.value_net[-1].weight.data)
        # self.value_net[-1].bias.data = torch.abs(self.value_net[-1].bias.data)

        params = list(self.schedule_embedding.parameters()) + list(self.value_net.parameters())

        self.optimizer = optim.Adam(params, weight_decay=weight_decay, lr=l_rate)

    def forward(self, state, actions, num_flights, num_trades):
        state = state.reshape((-1, state.shape[-1]))
        actions = actions.reshape((-1, actions.shape[-1]))  # /302200
        schedule_len = self.singleFlightSize * num_flights

        schedules = state[:, : schedule_len].to(self.device)
        # trades = state[:, schedule_len: -num_flights].to(self.device)
        # current_trade = state[:, -num_flights:].to(self.device)

        schedules = schedules.reshape((schedules.shape[0], num_flights, self.discretisationSize + self.numAirlines))
        schedules = self.schedule_embedding(schedules)

        actions = self.action_embedding(actions)
        QT = torch.matmul(actions.unsqueeze(-1), schedules.transpose(1, 2))
        QT = sMax(QT)

        h = torch.matmul(QT, schedules).squeeze(-1)

        # airline_eval = [s[torch.nonzero(schedules[:, :, i], as_tuple=True)] for i in range(4)]

        # trades = [trades[:, i*self.singleTradeSize: (i+1)*self.singleTradeSize] for i in range(num_trades)]
        #
        # actions = self.action_embedding(actions)
        # trades = torch.cat([self.trade_embedding(trade) for trade in trades], dim=-1)*100
        # current_trade = self.trade_embedding(current_trade)*100
        #
        # # trades_w = torch.matmul(trades, current_trade.unsqueeze(2))
        # # trades_w = sMax(trades_w)
        # # trades_attention = torch.matmul(trades.transpose(1, 2), trades_w).to(self.device)
        #
        # joint = torch.cat([p, s, trades, current_trade], dim=-1)
        result = self.value_net(h)

        return result

    def pick_action(self, state, mask, num_flights, len_episode):
        with torch.no_grad():
            scores = self.forward(state.to(self.device), mask.to(self.device), num_flights, len_episode)
        return scores

    def update_weights(self, batch: tuple, gamma: float = 1.0):
        criterion = torch.nn.MSELoss()

        states, next_states, masks, actions, rewards, dones = (element.to(self.device) for element in batch)

        self.zero_grad()
        curr_Q = self.forward(states)
        curr_Q = curr_Q.gather(1, actions.argmax(dim=1).view(-1, 1)).flatten()

        with torch.no_grad():
            next_Q = self.forward(next_states)
            next_Q[masks == 0] = -float('inf')
            max_next_Q = torch.max(next_Q, 1)[0]
            expected_Q = (rewards.flatten() + (1 - dones.flatten()) * gamma * max_next_Q)

        loss = criterion(curr_Q, expected_Q)
        self.loss = self.loss * 0.9 + 0.1 * loss.item()
        self.optimizer.zero_grad()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 10)
        self.optimizer.step()

        if self.loss < self.bestLoss:
            torch.save(self.state_dict(), "air.pt")

        return loss

    def update_weights_episode(self, batch: tuple, num_flights, num_trades, gamma: float = 1.0):
        criterion = torch.nn.MSELoss()

        states, actions, rewards = (element.to(self.device) for element in batch)
        actions_tensor = states[:, :self.singleFlightSize * num_flights]
        actions_tensor = actions_tensor.reshape((actions_tensor.shape[0], num_flights, self.singleFlightSize))

        actions_tensor = actions_tensor[torch.nonzero(actions, as_tuple=True)]

        self.zero_grad()
        Q = self.forward(states, actions_tensor, num_flights, num_trades)
        rewards = rewards.reshape((rewards.shape[0], -1))
        loss = criterion(Q, rewards)
        self.loss = self.loss * 0.9 + 0.1 * loss.item()
        self.optimizer.zero_grad()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 10)
        self.optimizer.step()

        # if self.loss < self.bestLoss:
        #     torch.save(self.state_dict(), "air.pt")

        return loss

        # schedules = schedules.reshape((schedules.shape[0], -1, self.singleFlightSize)).to(self.device)
        # trades = trades.reshape((trades.shape[0], -1, self.tradeSize)).to(self.device)
        #
        # schedules_e = self.schedule_embedding(schedules).to(self.device)
        # trades_e = self.trade_embedding(trades).to(self.device)
        # current_trades_e = self.trade_embedding(current_trades).to(self.device)
        #
        # trades_w = torch.matmul(trades_e, current_trades_e.unsqueeze(2)).to(self.device)
        # trades_w = sMax(trades_w)
        # trades_attention = torch.matmul(trades_e.transpose(1, 2), trades_w).to(self.device)
        #
        # schedules_w = torch.matmul(schedules_e, trades_attention).to(self.device)
        # schedules_w = sMax(schedules_w)
        # schedules_attention = torch.matmul(schedules_e.transpose(1, 2), schedules_w).to(self.device)

        # value_in = torch.cat((schedules_attention, trades_attention), dim=1).transpose(1, 2).squeeze(1).to(self.device)
        # value_in = torch.cat((value_in, current_trades_e), dim = 1).to(self.device)

