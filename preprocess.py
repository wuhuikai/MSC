from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import json
import glob
import gflags as flags

from google.protobuf.json_format import Parse

from pysc2 import run_configs
from s2clientprotocol import sc2api_pb2 as sc_pb

FLAGS = flags.FLAGS
flags.DEFINE_string(name='infos_path', default='replays_infos',
                    help='Paths for infos of replays')
flags.DEFINE_string(name='save_path', default='high_quality_replays',
                    help='Path for saving results')

flags.DEFINE_integer(name='min_duration', default=10000,
                     help='Min duration')
flags.DEFINE_integer(name='max_duration', default=None,
                     help='Max duration')
flags.DEFINE_integer(name='min_apm', default=10,
                     help='Min APM')
flags.DEFINE_integer(name='min_mmr', default=1000,
                     help='Min MMR')

def valid_replay(info, ping):
    """Make sure the replay isn't corrupt, and is worth looking at."""
    if info.HasField("error"):
        return False
    if info.base_build != ping.base_build:
        return False
    if info.game_duration_loops < FLAGS.min_duration:
        return False
    if FLAGS.max_duration is not None and info.game_duration_loops > FLAGS.max_duration:
        return  False
    if len(info.player_info) != 2:
        return False

    for p in info.player_info:
        if p.player_apm < FLAGS.min_apm or p.player_mmr < FLAGS.min_mmr:
            # Low APM = player just standing around.
            # Low MMR = corrupt replay or player who is weak.
            return False
        if p.player_result.result not in {1, 2}:
            return False

    return True

def main():
    if not os.path.isdir(FLAGS.save_path):
        os.makedirs(FLAGS.save_path)
    replay_infos = glob.glob(os.path.join(FLAGS.infos_path, '*.SC2Replay'))

    run_config = run_configs.get()
    with run_config.start() as controller:
        ping = controller.ping()

    result = {}

    for info_path in replay_infos:
        with open(info_path) as f:
            info = json.load(f)

        proto = Parse(info['info'], sc_pb.ResponseReplayInfo())
        if valid_replay(proto, ping):
            players_info = proto.player_info
            races = '_vs_'.join(sorted(sc_pb.Race.Name(player_info.player_info.race_actual)
                                       for player_info in players_info))
            if races not in result:
                result[races] = []
            result[races].append((info['path'], info_path))

    for k, v in result.items():
        with open(os.path.join(FLAGS.save_path, k+'.json'), 'w') as f:
            json.dump(v, f)

if __name__ == '__main__':
    FLAGS(sys.argv)
    main()