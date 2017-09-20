from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import stream
import gflags as flags

from tqdm import tqdm

from google.protobuf.json_format import Parse

from pysc2.lib import app
from pysc2.lib import features
from pysc2.lib import FUNCTIONS
from pysc2.lib import static_data
from s2clientprotocol import sc2api_pb2 as sc_pb

FLAGS = flags.FLAGS
flags.DEFINE_string(name='hq_replay_set', default='../high_quality_replays/Terran_vs_Terran.json',
                    help='File storing replays list')
flags.DEFINE_string(name='parsed_replay_path', default='../parsed_replays',
                    help='Path storing parsed replays')
flags.DEFINE_integer(name='step_mul', default=8,
                     help='step size')

def process_replay(sampled_action, actions, observations, feat, units_info, reward):
    states = []

    for frame_id, action, obs in zip(sampled_action, actions, observations):
        state = {}
        # actions
        state['action'] = None
        if action is not None:
            try:
                func_id = feat.reverse_action(action).function
                func_name = FUNCTIONS[func_id].name
                if func_name.split('_')[0] in {'Build', 'Train', 'Research', 'Morph', 'Cancel', 'Halt', 'Stop'}:
                    state['action'] = (func_id, func_name)
            except ValueError:
                pass

        observation = obs.observation
        #####################################################
        # frame_id
        assert frame_id == observation.game_loop-1
        state['frame_id'] = frame_id
        # reward
        state['reward'] = reward

        state['score_cumulative'] = [
            observation.score.score,
            observation.score.score_details.idle_production_time,
            observation.score.score_details.idle_worker_time,
            observation.score.score_details.total_value_units,
            observation.score.score_details.total_value_structures,
            observation.score.score_details.killed_value_units,
            observation.score.score_details.killed_value_structures,
            observation.score.score_details.collected_minerals,
            observation.score.score_details.collected_vespene,
            observation.score.score_details.collection_rate_minerals,
            observation.score.score_details.collection_rate_vespene,
            observation.score.score_details.spent_minerals,
            observation.score.score_details.spent_vespene,
        ]
        # resources
        resources = observation.player_common
        state['minerals'] = resources.minerals
        state['vespene'] = resources.vespene
        state['food_cap'] = resources.food_cap
        state['food_used'] = resources.food_used
        state['food_army'] = resources.food_army
        state['food_workers'] = resources.food_workers
        state['idle_worker_count'] = resources.idle_worker_count
        state['army_count'] = resources.army_count
        state['warp_gate_count'] = resources.warp_gate_count
        state['larva_count'] = resources.larva_count
        #####################################################
        # alert
        state['alert'] = list(observation.alerts)

        #####################################################
        ### raw data
        raw_data = observation.raw_data
        ## player
        player = raw_data.player
        # upgrades
        state['upgrades'] = list(player.upgrade_ids)
        # power
        state['n_power_source'] = len(player.power_sources)
        #####################################################
        ## units
        state['friendly_units'] = {}
        state['enemy_units'] = {}
        for unit in raw_data.units:
            if unit.display_type == 3:
                continue
            if unit.alliance != 1 and unit.alliance != 4:
                continue
            # Friendly or Enemy
            units = state['friendly_units'] if unit.alliance == 1 else state['enemy_units']
            # Already have this unit_type ?
            unit_type = unit.unit_type
            if unit_type not in units:
                units[unit_type] = {'units': [], 'name': units_info[unit_type]}
            # Basic info
            unit_info = {'tag': unit.tag,
                         'build_progress': unit.build_progress}

            units[unit_type]['units'].append(unit_info)

        states.append(state)

    return states

def parse_replay(replay_player_path, sampled_action_path, reward):
    # Global Info
    with open(os.path.join(FLAGS.parsed_replay_path, 'GlobalInfos', replay_player_path)) as f:
        global_info = json.load(f)
    units_info = static_data.StaticData(Parse(global_info['data_raw'], sc_pb.ResponseData())).units
    feat = features.Features(Parse(global_info['game_info'], sc_pb.ResponseGameInfo()))

    # Sampled Actions
    with open(sampled_action_path) as f:
        sampled_action = json.load(f)
    sampled_action_id = [id // FLAGS.step_mul + 1 for id in sampled_action]

    # Actions
    with open(os.path.join(FLAGS.parsed_replay_path, 'Actions', replay_player_path)) as f:
        actions = json.load(f)
    actions = [None if len(actions[idx]) == 0 else Parse(actions[idx][0], sc_pb.Action())
                for idx in sampled_action_id]

    # Observations
    observations =  [obs for obs in stream.parse(os.path.join(FLAGS.parsed_replay_path,
                            'SampledObservations', replay_player_path), sc_pb.ResponseObservation)]

    assert len(sampled_action) == len(sampled_action_id) == len(actions) == len(observations)

    states = process_replay(sampled_action, actions, observations, feat, units_info, reward)

    with open(os.path.join(FLAGS.parsed_replay_path, 'GlobalFeatures', replay_player_path), 'w') as f:
        json.dump(states, f)

def main(_):
    with open(FLAGS.hq_replay_set) as f:
        replay_list = sorted(json.load(f))

    race_vs_race = os.path.basename(FLAGS.hq_replay_set).split('.')[0]
    global_feature_path = os.path.join(FLAGS.parsed_replay_path, 'GlobalFeatures', race_vs_race)
    for race in set(race_vs_race.split('_vs_')):
        path = os.path.join(global_feature_path, race)
        if not os.path.isdir(path):
            os.makedirs(path)

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
            parse_replay(replay_player_path, sampled_action_path, reward)

        pbar.update()

if __name__ == '__main__':
    app.run()
