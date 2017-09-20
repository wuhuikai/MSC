import os
import json
import stream
import numpy as np
from collections import namedtuple

from tqdm import tqdm

from google.protobuf.json_format import Parse
from s2clientprotocol import sc2api_pb2 as sc_pb

class BatchEnv(object):
    def __init__(self, path, race, enemy_race, parsed_replays='parsed_replays',
                 step_mul=8, n_replays=4, n_steps=5, epoches=10, seed=None):
        np.random.seed(seed)

        with open(path) as f:
            replays = json.load(f)

        self.replays = self.__generate_replay_list__(replays, race)

        self.parsed_replays = parsed_replays
        self.race = race
        self.enemy_race = enemy_race

        self.step_mul = step_mul
        self.n_replays = n_replays
        self.n_steps = n_steps

        self.epoches = epoches
        self.epoch = -1
        self.steps = 0

        self.replay_idx = -1
        self.replay_list = [None for _ in range(self.n_replays)]

        ## Display Progress Bar
        self.epoch_pbar = tqdm(total=self.epoches, desc='Epoch')
        self.replay_pbar = None

    def __generate_replay_list__(self, replays, race):
        """
        Each replay: (info_path, sampled_action_path,
                        (global_info_path, action_path, sampled_observation_path),
                        (global_info_path, action_path, sampled_observation_path)/None)
        """
        result = []
        for path_dict in replays:
            info_path = path_dict['info_path']
            sampled_action_path = path_dict['sampled_action_path']
            for player_path in path_dict[race]:
                result.append((info_path, sampled_action_path,
                               (player_path['global_info_path'], player_path['action_path'],
                                player_path['sampled_observation_path']), None))

        return result

    def __init_epoch__(self):
        self.epoch += 1
        if self.epoch > 0:
            self.epoch_pbar.update(1)
        if self.epoch == self.epoches:
            return False

        np.random.shuffle(self.replays)
        ## Display Progress Bar
        if self.replay_pbar is not None:
            self.replay_pbar.close()
        self.replay_pbar = tqdm(total=len(self.replays), desc='  Replays')
        return True

    def __load_replay_info__(self, path):
        with open(path) as f:
            info = json.load(f)
        return Parse(info['info'], sc_pb.ResponseReplayInfo())

    def __load_global_info__(self, path):
        if path is None:
            return None
        with open(path) as f:
            global_info = json.load(f)
        global_info['game_info'] = Parse(global_info['game_info'], sc_pb.ResponseGameInfo())
        global_info['data_raw'] = Parse(global_info['data_raw'], sc_pb.ResponseData())

        return global_info

    def __load_action__(self, path, sampled_action_path):
        if path is None:
            return None
        with open(sampled_action_path) as f:
            sampled_action = json.load(f)
        sampled_action = [id//self.step_mul+1 for id in sampled_action]

        with open(path) as f:
            actions = json.load(f)

        result = []
        for idx in sampled_action:
            action = None
            for action_str in actions[idx]:
                action = Parse(action_str, sc_pb.Action())
                break
            result.append(action)

        return iter(result)

    def __load_observation__(self, path):
        if path is None:
            return None

        return stream.parse(path, sc_pb.ResponseObservation)

    def __reset__(self):
        self.replay_idx += 1
        if self.replay_idx % len(self.replays) == 0:
            has_more = self.__init_epoch__()
            if not has_more:
                return None

        path = self.replays[self.replay_idx%len(self.replays)]
        replay_info_path, sampled_action_path,  player_a_paths, player_b_paths = path

        replay_dict = {}
        # Info
        replay_dict['info'] = self.__load_replay_info__(replay_info_path)

        ## Player
        player_a_global_info_path, player_a_action_path, player_a_sampled_observation_path = player_a_paths
        player_b_global_info_path, player_b_action_path, player_b_sampled_observation_path = None, None, None
        if player_b_paths is not None:
            player_b_global_info_path, player_b_action_path, player_b_sampled_observation_path = player_b_paths
        # Global Info
        replay_dict['global_info'] = (self.__load_global_info__(player_a_global_info_path),
                                      self.__load_global_info__(player_b_global_info_path))
        # Actions
        replay_dict['action'] = (self.__load_action__(player_a_action_path, sampled_action_path),
                                 self.__load_action__(player_b_action_path, sampled_action_path))
        # Observations
        replay_dict['observation'] = (self.__load_observation__(player_a_sampled_observation_path),
                                      self.__load_observation__(player_b_sampled_observation_path))

        replay_dict['player_id'] = int(os.path.basename(player_a_action_path).split('@')[0])
        replay_dict['done'] = False

        return self.__process_replay__(replay_dict)

    def __process_replay__(self, replay_dict):
        return replay_dict

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

from game_state import GameState

class BatchGlobalFeatureEnv(BatchEnv):
    n_features = 738
    n_actions = 75
    Feature = namedtuple('Feature', ['state', 'reward', 'action', 'score'])

    def __process_replay__(self, replay_dict):
        replay_dict['game_state'] = GameState(os.path.join(self.parsed_replays, 'Stat',
                                                           '{}.json'.format(self.race)),
                                              os.path.join(self.parsed_replays, 'Stat',
                                                           '{}.json'.format(self.enemy_race))
                                              )
        return replay_dict

    def __load_replay_info__(self, path):
        return None

    def __load_global_info__(self, path):
        return None

    def __load_observation__(self, path):
        if path is None:
            return None
        path = path.replace('SampledObservations', 'GlobalFeatures')
        with open(path) as f:
            states = json.load(f)
        return iter(states)

    def __load_action__(self, path, sampled_action_path):
        return None

    def __one_step__(self, replay_dict, done):
        if done:
            return self.Feature(np.zeros(self.n_features), np.zeros(1), np.zeros(1), np.zeros(13))

        self.steps += 1
        try:
            state = next(replay_dict['observation'][0])
        except StopIteration:
            self.replay_pbar.update(1)
            replay_dict['done'] = True
            return self.Feature(np.zeros(self.n_features), np.zeros(1), np.zeros(1), np.zeros(13))

        game_state = replay_dict['game_state']
        game_state.update(state)

        state_vec = np.asarray(game_state.to_vector())
        reward_vec = np.asarray([game_state.reward])
        action_vec = np.asarray([game_state.get_action()])
        score_vec = np.asarray(game_state.score)

        return self.Feature(state_vec, reward_vec, action_vec, score_vec)

    def __post_process__(self, result, reward=True, action=False, score=False):
        results = {k: [] for k in self.Feature._fields}
        for result_per_step in result:
            feature = self.Feature(*zip(*result_per_step))
            for k in self.Feature._fields:
                results[k].append(getattr(feature, k))
        for k, v in results.items():
            results[k] = np.asarray(v)

        result_list = [results['state']]
        if reward:
            result_list.append(results['reward'])
        if action:
            result_list.append(results['action'])
        if score:
            result_list.append(results['score'])
        return result_list

from pysc2.lib import FUNCTIONS
from game_state import load_stat
from SpatialFeatures import SpatialFeatures

class BatchSpatialEnv(BatchEnv):
    n_actions = 75
    Feature = namedtuple('Feature', ['minimap', 'game_loop', 'screen',
                                     'player', 'score', 'reward', 'action'])

    def __load_observation__(self, path):
        if path is None:
            return None
        return iter([obs for obs in stream.parse(path, sc_pb.ResponseObservation)])

    def __process_replay__(self, replay_dict):
        replay_dict['features'] = SpatialFeatures(replay_dict['global_info'][0]['game_info'])
        info, player_id = replay_dict['info'], replay_dict['player_id']
        for player_info in info.player_info:
            if player_id == player_info.player_info.player_id:
                replay_dict['reward'] = 2-player_info.player_result.result
                break

        return replay_dict

    def __one_step__(self, replay_dict, done):
        if not hasattr(self, 'stat'):
            self.stat = load_stat(os.path.join(self.parsed_replays,
                                               'Stat', '{}.json'.format(self.race)))

        spec = replay_dict['features'].observation_spec()
        if done:
            return self.Feature(**{k:np.zeros(v) for k, v in spec.items()},
                                reward=np.zeros(1), action=np.zeros(1))

        self.steps += 1
        try:
            state = next(replay_dict['observation'][0])
            action = next(replay_dict['action'][0])
            action_id = -1
            if action is not None:
                try:
                    func_id = replay_dict['features'].reverse_action(action).function
                    func_name = FUNCTIONS[func_id].name
                    if func_name.split('_')[0] in {'Build', 'Train', 'Research', 'Morph', 'Cancel', 'Halt', 'Stop'}:
                        action_id = func_id
                except:
                    pass
        except StopIteration:
            self.replay_pbar.update(1)
            replay_dict['done'] = True
            return self.Feature(**{k: np.zeros(v) for k, v in spec.items()},
                                reward=np.zeros(1), action=np.zeros(1))

        feature = replay_dict['features'].transform_obs(state.observation)
        return self.Feature(**feature, reward=replay_dict['reward'], action=self.stat['action_id'][action_id])

    def __post_process__(self, result):
        results = {k:[] for k in self.Feature._fields}
        for result_per_step in result:
            feature = self.Feature(*zip(*result_per_step))
            for k in self.Feature._fields:
                results[k].append(getattr(feature, k))

        for k, v in results.items():
            results[k] = np.asarray(v)

        return results

if __name__ == '__main__':
    env = BatchSpatialEnv('train_val_test/Terran_vs_Terran/train.json', 'Terran', 'Terran')
    while True:
        r = env.step()
        if r is None:
            break