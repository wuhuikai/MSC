import os
import json
import numpy as np
import gflags as flags

from google.protobuf.json_format import Parse
from pysc2.lib import app
from s2clientprotocol import sc2api_pb2 as sc_pb

FLAGS = flags.FLAGS
flags.DEFINE_string(name='hq_replay_set', default='high_quality_replays/Terran_vs_Terran.json',
                    help='File storing replays list')
flags.DEFINE_string(name='parsed_replay_path', default='parsed_replays',
                    help='Path for parsed replays')
flags.DEFINE_string(name='save_path', default='train_val_test',
                    help='Path for saving results')
flags.DEFINE_string(name='ratio', default='7:1:2',
                    help='train:val:test')
flags.DEFINE_integer(name='seed', default=1,
                    help='random seed')

def save(replays, prefix, folder):
    print('{}/{}: {}'.format(folder, prefix, len(replays)))
    with open(os.path.join(folder, prefix+'.json'), 'w') as f:
        json.dump(replays, f)

def main(_):
    np.random.seed(FLAGS.seed)
    ratio = np.asarray([float(i) for i in FLAGS.ratio.split(':')])
    ratio /= np.sum(ratio)

    if not os.path.isdir(FLAGS.save_path):
        os.makedirs(FLAGS.save_path)

    with open(FLAGS.hq_replay_set) as f:
        replays_list = json.load(f)

    result = []
    race_vs_race = os.path.basename(FLAGS.hq_replay_set).split('.')[0]
    for replay_path, info_path in replays_list:
        replay_path_dict = {}
        replay_name = os.path.basename(replay_path)

        ## Sampled Action
        sampled_action_path = os.path.join(FLAGS.parsed_replay_path, 'SampledActions', race_vs_race, replay_name)
        if not os.path.isfile(sampled_action_path):
            continue
        replay_path_dict['sampled_action_path'] = sampled_action_path

        ## Replay Info
        if not os.path.isfile(info_path):
            continue
        replay_path_dict['info_path'] = info_path

        ## Parsed Replay
        for race in set(race_vs_race.split('_vs_')):
            replay_path_dict[race] = []

        broken = False
        with open(info_path) as f:
            info = json.load(f)
        proto = Parse(info['info'], sc_pb.ResponseReplayInfo())
        for p in proto.player_info:
            player_id = p.player_info.player_id
            race = sc_pb.Race.Name(p.player_info.race_actual)

            parsed_replays_info = {}
            ## Global Info
            global_info_path = os.path.join(FLAGS.parsed_replay_path, 'GlobalInfos', race_vs_race, race,
                                            '{}@{}'.format(player_id, replay_name))
            if not os.path.isfile(global_info_path):
                broken = True
                break
            parsed_replays_info['global_info_path'] = global_info_path
            ## Global Info
            action_path = os.path.join(FLAGS.parsed_replay_path, 'Actions', race_vs_race, race,
                                            '{}@{}'.format(player_id, replay_name))
            if not os.path.isfile(action_path):
                broken = True
                break
            parsed_replays_info['action_path'] = action_path
            ## Observations
            sampled_observation_path = os.path.join(FLAGS.parsed_replay_path, 'SampledObservations', race_vs_race, race,
                                            '{}@{}'.format(player_id, replay_name))
            if not os.path.isfile(sampled_observation_path):
                broken = True
                break
            parsed_replays_info['sampled_observation_path'] = sampled_observation_path

            replay_path_dict[race].append(parsed_replays_info)

        if broken:
            continue

        result.append(replay_path_dict)

    FLAGS.save_path = os.path.join(FLAGS.save_path, race_vs_race)
    if not os.path.isdir(FLAGS.save_path):
        os.makedirs(FLAGS.save_path)

    train_end = int(len(result)*ratio[0])
    val_end = int(len(result)*(ratio[0]+ratio[1]))

    np.random.shuffle(result)
    save(result[:train_end], 'train', FLAGS.save_path)
    save(result[train_end:val_end], 'val', FLAGS.save_path)
    save(result[val_end:], 'test', FLAGS.save_path)

if __name__ == '__main__':
    app.run()