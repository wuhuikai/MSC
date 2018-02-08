import os
import json
import numpy as np
from absl import flags

from google.protobuf.json_format import Parse
from s2clientprotocol import sc2api_pb2 as sc_pb

FLAGS = flags.FLAGS
flags.DEFINE_string(name='hq_replay_set', default='../high_quality_replays/Terran_vs_Terran.json',
                    help='File storing replays list')
flags.DEFINE_string(name='root', default='..',
                    help='Root for parsed replays')
flags.DEFINE_string(name='parsed_replay_path', default='parsed_replays',
                    help='Path for parsed replays')
flags.DEFINE_string(name='save_path', default='../train_val_test',
                    help='Path for saving results')
flags.DEFINE_string(name='ratio', default='7:1:2',
                    help='train:val:test')
flags.DEFINE_integer(name='seed', default=1,
                    help='random seed')

def save(replays, prefix, folder):
    print('{}/{}: {}'.format(folder, prefix, len(replays)))
    with open(os.path.join(folder, prefix+'.json'), 'w') as f:
        json.dump(replays, f)

def main():
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

        ## Parsed Replay
        for race in set(race_vs_race.split('_vs_')):
            replay_path_dict[race] = []

        with open(info_path) as f:
            info = json.load(f)
        proto = Parse(info['info'], sc_pb.ResponseReplayInfo())
        for p in proto.player_info:
            player_id = p.player_info.player_id
            race = sc_pb.Race.Name(p.player_info.race_actual)

            parsed_replays_info = {}
            ## Global Feature
            global_path = os.path.join(FLAGS.parsed_replay_path, 'GlobalFeatureVector', race_vs_race, race,
                                            '{}@{}.npz'.format(player_id, replay_name))

            if os.path.isfile(os.path.join(FLAGS.root, global_path)):
                parsed_replays_info['global_path'] = global_path
            ## Spatial Feature
            spatial_path_S = os.path.join(FLAGS.parsed_replay_path, 'SpatialFeatureTensor', race_vs_race, race,
                                            '{}@{}@S.npz'.format(player_id, replay_name))
            if os.path.isfile(os.path.join(FLAGS.root, spatial_path_S)):
                parsed_replays_info['spatial_path_S'] = spatial_path_S

            spatial_path_G = os.path.join(FLAGS.parsed_replay_path, 'SpatialFeatureTensor', race_vs_race, race,
                                          '{}@{}@G.npz'.format(player_id, replay_name))
            if os.path.isfile(os.path.join(FLAGS.root, spatial_path_G)):
                parsed_replays_info['spatial_path_G'] = spatial_path_G

            replay_path_dict[race].append(parsed_replays_info)

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
    main()