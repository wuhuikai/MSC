from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import json
import gflags as flags
from tqdm import tqdm

from google.protobuf.json_format import Parse

from pysc2.lib import features
from pysc2.lib import  FUNCTIONS
from s2clientprotocol import sc2api_pb2 as sc_pb

FLAGS = flags.FLAGS
flags.DEFINE_string(name='hq_replay_set', default='high_quality_replays/Terran_vs_Terran.json',
                    help='File storing replays list')
flags.DEFINE_string(name='parsed_replays', default='parsed_replays',
                    help='Path for parsed actions')
flags.DEFINE_string(name='infos_path', default='replays_infos',
                    help='Paths for infos of replays')
flags.DEFINE_integer(name='step_mul', default=96,
                     help='step size')

def sample_action_from_player(global_info_path, action_path):
    with open(global_info_path) as f:
        global_info = json.load(f)
    feat = features.Features(Parse(global_info['game_info'], sc_pb.ResponseGameInfo()))

    with open(action_path) as f:
        actions = json.load(f)

    frame_id = 0
    result_frames = []
    for step_mul, action_strs in actions:
        action_name = None
        for action_str in action_strs:
            action = Parse(action_str, sc_pb.Action())
            try:
                func_id = feat.reverse_action(action).function
            except:
                pass
            func_name = FUNCTIONS[func_id].name
            if func_name.split('_')[0] in {'Build', 'Train', 'Research', 'Morph', 'Cancel', 'Halt', 'Stop'}:
                action_name = func_name
                break
        if action_name is not None or (frame_id > 0 and frame_id%FLAGS.step_mul == 0):
            result_frames.append(frame_id-step_mul)

        frame_id += step_mul

    return result_frames


def sample_action(replay_path, global_info_path, action_path, sampled_path):
    replay_info = os.path.join(FLAGS.infos_path, replay_path)
    if not os.path.isfile(replay_info):
        return
    with open(replay_info) as f:
        info = json.load(f)

    result = []
    proto = Parse(info['info'], sc_pb.ResponseReplayInfo())
    for p in proto.player_info:
        player_id = p.player_info.player_id
        race = sc_pb.Race.Name(p.player_info.race_actual)
        global_info = os.path.join(global_info_path, race, 'G_{}_{}'.format(player_id, replay_path))
        if not os.path.isfile(global_info):
            return
        action = os.path.join(action_path, race, 'A_{}_{}'.format(player_id, replay_path))
        if not os.path.isfile(action):
            return

        result.append(sample_action_from_player(global_info, action))

    assert len(result) == 2
    player_1, player_2 = result

    sampled_actions = sorted(set(player_1) | set(player_2))

    with open(os.path.join(sampled_path, replay_path), 'w') as f:
        json.dump(sampled_actions, f)

def main():
    with open(FLAGS.hq_replay_set) as f:
        replay_list = json.load(f)
    replay_list = sorted([p for p, _ in replay_list])

    race_vs_race = os.path.basename(FLAGS.hq_replay_set).split('.')[0]
    FLAGS.parsed_replays = os.path.join(FLAGS.parsed_replays, race_vs_race)

    sampled_path = os.path.join(FLAGS.parsed_replays, 'SampledActions')
    if not os.path.isdir(sampled_path):
        os.makedirs(sampled_path)

    global_info_path = os.path.join(FLAGS.parsed_replays, 'GlobalInfos')
    action_path = os.path.join(FLAGS.parsed_replays, 'Actions')

    pbar = tqdm(total=len(replay_list), desc='#Replay')
    for replay_path in replay_list:
        sample_action(os.path.basename(replay_path), global_info_path, action_path, sampled_path)
        pbar.update()

if __name__ == '__main__':
    FLAGS(sys.argv)
    main()