from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import pprint
import numpy as np

import os
import json
from absl import flags

from tqdm import tqdm

FLAGS = flags.FLAGS
flags.DEFINE_string(name='hq_replay_path', default='../high_quality_replays',
                    help='Path for replays lists')
flags.DEFINE_string(name='parsed_replay_path', default='../parsed_replays',
                    help='Path storing parsed replays')
flags.DEFINE_string(name='race', default='Terran',
                     help='Race name')
# MAX stat
max_keys = {'frame_id', 'minerals', 'vespene', 'food_cap',
                'idle_worker_count', 'army_count', 'warp_gate_count',
                    'larva_count', 'n_power_source'}
# SET stat
set_keys = {'alert', 'upgrades'}

def update(replay_path, stat):
    with open(replay_path) as f:
        states = json.load(f)

    research_count = {}
    for state in states:
        # MAX stat
        for key in max_keys:
            stat['max_' + key] = max(state[key], stat['max_'+key])
        # SET stat
        for key in set_keys:
            stat[key].update(set(state[key]))
        # max_score_cumulative
        stat['max_score_cumulative'] = max(stat['max_score_cumulative'], state['score_cumulative'][0])
        ## Units stat
        units = state['friendly_units']
        for unit_type, unit in units.items():
            stat['units_type'].add(unit_type)
            stat['units_name'][unit_type] = unit['name']
            stat['max_unit_num'] = max(stat['max_unit_num'], len(unit['units']))
        ## Actions
        if state['action'] is None:
            continue

        id, name = state['action']
        stat['action_id'].add(id)
        stat['action_name'][id] = name

        if name.startswith('Research_'):
            stat['research_id'].add(id)
            if id not in research_count:
                research_count[id] = 0
            research_count[id] += 1
    if len(research_count) > 0:
        stat['max_research_num'] = max(stat['max_research_num'], max(research_count.values()))

def post_process(stat):
    for key in set_keys | {'action_id', 'research_id', 'units_type'}:
        values = np.asarray(list(stat[key]))
        idx = np.argsort(values)
        values = values[idx]
        stat[key] = {k: v for k, v in zip(values, idx)}

    # Turn all keys into str
    def dict_key_to_str(obj):
        if not isinstance(obj, dict):
            return str(obj)
        return {str(k):dict_key_to_str(v) for k, v in obj.items()}

    return dict_key_to_str(stat)

def main():
    replay_lists = sorted(glob.glob(os.path.join(FLAGS.hq_replay_path, '*{}*.json'.format(FLAGS.race))))
    save_path = os.path.join(FLAGS.parsed_replay_path, 'Stat')
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    ## Init Stat Dict
    stat = {}
    # MAX stat
    for key in max_keys:
        stat['max_'+key] = 0
    # SET stat
    for key in set_keys:
        stat[key] = set()
    # score_cumulative
    stat['max_score_cumulative'] = 0
    ## Units stat
    stat['units_type'] = set()
    stat['units_name'] = {}
    stat['max_unit_num'] = 0
    ## Actions
    stat['action_id'] = set()
    stat['action_name'] = {}
    stat['research_id'] = set()
    stat['max_research_num'] = 0

    set_bar = tqdm(total=len(replay_lists), desc='#SET')
    for replay_list_path in replay_lists:
        race_vs_race = os.path.basename(replay_list_path).split('.')[0]
        replays = glob.glob(os.path.join(FLAGS.parsed_replay_path, 'GlobalFeatures',
                                         race_vs_race, FLAGS.race, '*.SC2Replay'))

        pbar = tqdm(total=len(replays), desc='\t#Replay')
        for replay in replays:
            update(replay, stat)
            pbar.update()

        set_bar.update()

    stat = post_process(stat)

    with open(os.path.join(save_path, '{}_human.json'.format(FLAGS.race)), 'w') as f:
        f.write(pprint.pformat(stat))
    with open(os.path.join(save_path, '{}.json'.format(FLAGS.race)), 'w') as f:
        json.dump(stat, f)

if __name__ == '__main__':
    main()
