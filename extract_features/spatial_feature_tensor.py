from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import stream
import gflags as flags

import numpy as np
from scipy import sparse

from tqdm import tqdm

from google.protobuf.json_format import Parse

from pysc2.lib import app
from pysc2.lib import FUNCTIONS
from s2clientprotocol import sc2api_pb2 as sc_pb

from game_state import load_stat
from SpatialFeatures import SpatialFeatures

FLAGS = flags.FLAGS
flags.DEFINE_string(name='hq_replay_set', default='../high_quality_replays/Terran_vs_Terran.json',
                    help='File storing replays list')
flags.DEFINE_string(name='parsed_replay_path', default='../parsed_replays',
                    help='Path storing parsed replays')
flags.DEFINE_integer(name='step_mul', default=8,
                     help='step size')

def parse_replay(replay_player_path, sampled_action_path, reward, race, enemy_race, stat):
    with open(os.path.join(FLAGS.parsed_replay_path, 'GlobalInfos', replay_player_path)) as f:
        global_info = json.load(f)
    feat = SpatialFeatures(Parse(global_info['game_info'], sc_pb.ResponseGameInfo()))

    states = [obs for obs in stream.parse(os.path.join(FLAGS.parsed_replay_path,
                    'SampledObservations', replay_player_path), sc_pb.ResponseObservation)]

    # Sampled Actions
    with open(sampled_action_path) as f:
        sampled_action = json.load(f)
    sampled_action_id = [id // FLAGS.step_mul + 1 for id in sampled_action]
    # Actions
    with open(os.path.join(FLAGS.parsed_replay_path, 'Actions', replay_player_path)) as f:
        actions = json.load(f)
    actions = [None if len(actions[idx]) == 0 else Parse(actions[idx][0], sc_pb.Action())
               for idx in sampled_action_id]

    assert len(states) == len(actions)

    spatial_states_np, global_states_np = [], []
    for state, action in zip(states, actions):
        action_id = -1
        if action is not None:
            try:
                func_id = feat.reverse_action(action).function
                func_name = FUNCTIONS[func_id].name
                if func_name.split('_')[0] in {'Build', 'Train', 'Research', 'Morph', 'Cancel', 'Halt', 'Stop'}:
                    action_id = func_id
            except:
                pass

        obs = feat.transform_obs(state.observation)
        spatial_states_np.append(np.concatenate([obs['screen'], obs['minimap']], axis=0))

        global_states_np.append(np.hstack([obs['player']/(stat['max']+1e-5), obs['score'], [reward],
                                           [stat['action_id'][action_id]]]))

    spatial_states_np = np.asarray(spatial_states_np)
    global_states_np = np.asarray(global_states_np)

    spatial_states_np = spatial_states_np.reshape([len(states), -1])
    sparse.save_npz(os.path.join(FLAGS.parsed_replay_path, 'SpatialFeatureTensor',
                                 replay_player_path+'@S'), sparse.csc_matrix(spatial_states_np))
    sparse.save_npz(os.path.join(FLAGS.parsed_replay_path, 'SpatialFeatureTensor',
                                 replay_player_path+'@G'), sparse.csc_matrix(global_states_np))

max_keys = ['frame_id', 'minerals', 'vespene', 'food_cap',
                    'food_cap', 'food_cap', 'food_cap', 'idle_worker_count',
                        'army_count', 'warp_gate_count', 'larva_count']

def main(_):
    with open(FLAGS.hq_replay_set) as f:
        replay_list = sorted(json.load(f))

    race_vs_race = os.path.basename(FLAGS.hq_replay_set).split('.')[0]
    global_feature_vec_path = os.path.join(FLAGS.parsed_replay_path, 'SpatialFeatureTensor', race_vs_race)
    races = set(race_vs_race.split('_vs_'))
    stats = {}
    for race in races:
        path = os.path.join(global_feature_vec_path, race)
        if not os.path.isdir(path):
            os.makedirs(path)

        stat = load_stat(os.path.join(FLAGS.parsed_replay_path, 'Stat', '{}.json'.format(race)))
        stats[race] = {'max': np.asarray([stat['max_'+k] for k in max_keys]),
                       'action_id': stat['action_id']}

    pbar = tqdm(total=len(replay_list), desc='#Replay')
    for replay_path, replay_info_path in replay_list:
        with open(replay_info_path) as f:
            info = json.load(f)
        info = Parse(info['info'], sc_pb.ResponseReplayInfo())

        replay_name = os.path.basename(replay_path)
        sampled_action_path = os.path.join(FLAGS.parsed_replay_path, 'SampledActions', race_vs_race, replay_name)
        for player_info in info.player_info:
            race = sc_pb.Race.Name(player_info.player_info.race_actual)
            player_id = player_info.player_info.player_id
            reward = player_info.player_result.result

            replay_player_path = os.path.join(race_vs_race, race, '{}@{}'.format(player_id, replay_name))
            parse_replay(replay_player_path, sampled_action_path, reward, race,
                                race if len(races) == 1 else list(races - {race})[0], stats[race])

        pbar.update()

if __name__ == '__main__':
    app.run()
