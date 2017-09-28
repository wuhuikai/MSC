import os
import json
from collections import namedtuple

import numpy as np
from scipy import sparse

from tqdm import tqdm

class BatchEnv(object):
    def __init__(self):
        pass

    def init(self, path, root, race, enemy_race, step_mul=8, n_replays=4, n_steps=5, epochs=10, seed=None):
        np.random.seed(seed)

        with open(path) as f:
            replays = json.load(f)

        self.replays = self.__generate_replay_list__(replays, root, race)

        self.race = race
        self.enemy_race = enemy_race

        self.step_mul = step_mul
        self.n_replays = n_replays
        self.n_steps = n_steps

        self.epochs = epochs
        self.epoch = -1
        self.steps = 0

        self.replay_idx = -1
        self.replay_list = [None for _ in range(self.n_replays)]

        ## Display Progress Bar
        self.epoch_pbar = tqdm(total=self.epochs, desc='Epoch')
        self.replay_pbar = None

        self.__post_init__()

    def __generate_replay_list__(self, replays, race):
        raise NotImplementedError

    def __init_epoch__(self):
        self.epoch += 1
        if self.epoch > 0:
            self.epoch_pbar.update(1)
        if self.epoch == self.epochs:
            return False

        np.random.shuffle(self.replays)
        ## Display Progress Bar
        if self.replay_pbar is not None:
            self.replay_pbar.close()
        self.replay_pbar = tqdm(total=len(self.replays), desc='  Replays')
        return True

    def __reset__(self):
        self.replay_idx += 1
        if self.replay_idx % len(self.replays) == 0:
            has_more = self.__init_epoch__()
            if not has_more:
                return None

        path = self.replays[self.replay_idx%len(self.replays)]

        return self.__load_replay__(path)

    def __load_replay__(self, path):
        raise NotImplementedError

    def step(self, **kwargs):
        require_init = [False for _ in range(self.n_replays)]
        for i in range(self.n_replays):
            if self.replay_list[i] is None or self.replay_list[i]['done']:
                if self.replay_list[i] is not None:
                    keys = set(self.replay_list[i].keys())
                    for k in keys:
                        del self.replay_list[i][k]
                self.replay_list[i] = self.__reset__()
                require_init[i] = True
            if self.replay_list[i] is None:
                return None

        result = []
        for step in range(self.n_steps):
            result_per_step = []
            for i in range(self.n_replays):
                replay_dict = self.replay_list[i]

                features = self.__one_step__(replay_dict, replay_dict['done'])

                result_per_step.append(features)

            result.append(result_per_step)

        return self.__post_process__(result, **kwargs), require_init

    def __one_step__(self, replay_dict, done):
        raise NotImplementedError

    def __post_process__(self, result, **kwargs):
        raise NotImplementedError

    def step_count(self):
        return self.steps

    def close(self):
        if self.epoch_pbar is not None:
            self.epoch_pbar.close()
        if self.replay_pbar is not None:
            self.replay_pbar.close()

class BatchGlobalFeatureEnv(BatchEnv):
    n_features_dic = {'Terran':  {'Terran': 738,  'Protoss': 648,  'Zerg': 1116},
                      'Protoss': {'Terran': 638,  'Protoss': 548,  'Zerg': 1016},
                      'Zerg':    {'Terran': 1106, 'Protoss': 1016, 'Zerg': 1484}}
    n_actions_dic = {'Terran': 75, 'Protoss': 61, 'Zerg': 74}

    def __post_init__(self):
        self.n_features = self.n_features_dic[self.race][self.enemy_race]
        self.n_actions = self.n_actions_dic[self.race]

    def __generate_replay_list__(self, replays, root, race):
        result = []
        for path_dict in replays:
            for player_path in path_dict[race]:
                result.append(os.path.join(root, player_path['global_path']))

        return result

    def __load_replay__(self, path):
        replay_dict = {}
        replay_dict['ptr'] = 0
        replay_dict['done'] = False
        replay_dict['states'] = np.asarray(sparse.load_npz(path).todense())

        return replay_dict

    def __one_step__(self, replay_dict, done):
        states = replay_dict['states']
        feature_shape = states.shape[1:]
        if done:
            return np.zeros(feature_shape)

        self.steps += 1
        state = states[replay_dict['ptr']]
        replay_dict['ptr'] += 1
        if replay_dict['ptr'] == states.shape[0]:
            self.replay_pbar.update(1)
            replay_dict['done'] = True

        return state

    def __post_process__(self, result, reward=True, action=False, score=False):
        result = np.asarray(result)

        result_return = [result[:, :, 15:]]
        if reward:
            result_return.append(result[:, :, 0:1])
        if action:
            result_return.append(result[:, :, 1:2])
        if score:
            result_return.append(result[:, :, 2:15])

        return result_return

class BatchSpatialEnv(BatchEnv):
    n_channels = 5
    n_features = 11
    n_actions_dic = {'Terran': 75, 'Protoss': 61, 'Zerg': 74}
    Feature = namedtuple('Feature', ['S', 'G'])

    def __post_init__(self):
        self.n_actions = self.n_actions_dic[self.race]

    def __generate_replay_list__(self, replays, root, race):
        result = []
        for path_dict in replays:
            for player_path in path_dict[race]:
                result.append([os.path.join(root, player_path['spatial_path_S']),
                               os.path.join(root, player_path['spatial_path_G'])])
        return result

    def __load_replay__(self, path):
        replay_dict = {}
        replay_dict['ptr'] = 0
        replay_dict['done'] = False
        replay_dict['states_S'] = np.asarray(sparse.load_npz(path[0]).todense()).reshape([-1, 13, 64, 64])
        replay_dict['states_G'] = np.asarray(sparse.load_npz(path[1]).todense())

        return replay_dict

    def __one_step__(self, replay_dict, done):
        states_S = replay_dict['states_S']
        states_G = replay_dict['states_G']
        feature_shape_S = states_S.shape[1:]
        feature_shape_G = states_G.shape[1:]
        if done:
            return self.Feature(np.zeros(feature_shape_S), np.zeros(feature_shape_G))

        self.steps += 1
        state_S = states_S[replay_dict['ptr']]
        state_G = states_G[replay_dict['ptr']]
        replay_dict['ptr'] += 1
        if replay_dict['ptr'] == states_S.shape[0]:
            self.replay_pbar.update(1)
            replay_dict['done'] = True

        return self.Feature(state_S, state_G)

    def __post_process__(self, result, reward=True, action=False, score=False):
        result = self.Feature(*zip(*[self.Feature(*zip(*result_per_step)) for result_per_step in result]))

        S = np.asarray(result.S)
        G = np.asarray(result.G)

        result_return = [S[:, :, 8:13, :, :], G[:,:, :11]]
        if reward:
            result_return.append(G[:, :, 24:25])
        if action:
            result_return.append(G[:, :, 25:26])
        if score:
            result_return.append(G[:, :, 11:24])

        return result_return

if __name__ == '__main__':
    env = BatchSpatialEnv()
    env.init('../train_val_test/Terran_vs_Terran/train.json', '../', 'Terran', 'Terran')
    while True:
        r = env.step()
        if r is None:
            break
