from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
from absl import flags

import numpy as np
from scipy import sparse

from tqdm import tqdm

from google.protobuf.json_format import Parse

from s2clientprotocol import sc2api_pb2 as sc_pb

from game_state import GameState

FLAGS = flags.FLAGS
flags.DEFINE_string(name='hq_replay_set', default='../high_quality_replays/Terran_vs_Terran.json',
                    help='File storing replays list')
flags.DEFINE_string(name='parsed_replay_path', default='../parsed_replays',
                    help='Path storing parsed replays')

def parse_replay(replay_player_path, reward, race, enemy_race):
    with open(os.path.join(FLAGS.parsed_replay_path, 'GlobalFeatures', replay_player_path)) as f:
        states = json.load(f)

    states_np = []
    game_state = GameState(os.path.join(FLAGS.parsed_replay_path, 'Stat', '{}.json'.format(race)),
                           os.path.join(FLAGS.parsed_replay_path, 'Stat', '{}.json'.format(enemy_race)))
    for state in states:
        game_state.update(state)
        states_np.append(np.hstack([game_state.reward, game_state.get_action(),
                                    game_state.score, game_state.to_vector()]))
    states_np = np.asarray(states_np)

    sparse.save_npz(os.path.join(FLAGS.parsed_replay_path, 'GlobalFeatureVector',
                                 replay_player_path), sparse.csc_matrix(states_np))

def main():
    with open(FLAGS.hq_replay_set) as f:
        replay_list = sorted(json.load(f))

    race_vs_race = os.path.basename(FLAGS.hq_replay_set).split('.')[0]
    global_feature_vec_path = os.path.join(FLAGS.parsed_replay_path, 'GlobalFeatureVector', race_vs_race)
    races = set(race_vs_race.split('_vs_'))
    for race in races:
        path = os.path.join(global_feature_vec_path, race)
        if not os.path.isdir(path):
            os.makedirs(path)

    pbar = tqdm(total=len(replay_list), desc='#Replay')
    for replay_path, replay_info_path in replay_list:
        with open(replay_info_path) as f:
            info = json.load(f)
        info = Parse(info['info'], sc_pb.ResponseReplayInfo())

        replay_name = os.path.basename(replay_path)
        for player_info in info.player_info:
            race = sc_pb.Race.Name(player_info.player_info.race_actual)
            player_id = player_info.player_info.player_id
            reward = player_info.player_result.result

            replay_player_path = os.path.join(race_vs_race, race, '{}@{}'.format(player_id, replay_name))
            parse_replay(replay_player_path, reward, race, race if len(races) == 1 else list(races - {race})[0])

        pbar.update()

if __name__ == '__main__':
    main()
