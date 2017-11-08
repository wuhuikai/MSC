import os
import json
import pprint
import numpy as np

def load_stat(path):
    def dict_key_to_int(obj):
        def str2int(s):
            try:
                return int(s)
            except:
                return s

        if not isinstance(obj, dict):
            return str2int(obj)

        return {str2int(k): dict_key_to_int(v) for k, v in obj.items()}

    with open(path) as f:
        stat = json.load(f)
    stat = dict_key_to_int(stat)
    stat['action_id'][-1] = len(stat['action_id'])

    return stat

class GameState(object):
    ## Starcraft Stat
    eps = 1e-9

    stat_path, stat = None, None
    enemy_stat_path, enemy_stat = None, None

    max_vars = ['frame_id', 'minerals', 'vespene', 'food_cap',
                    'food_used', 'food_army', 'food_workers', 'idle_worker_count',
                        'army_count', 'warp_gate_count', 'larva_count', 'n_power_source']
    max_keys = ['frame_id', 'minerals', 'vespene', 'food_cap',
                    'food_cap', 'food_cap', 'food_cap', 'idle_worker_count',
                        'army_count', 'warp_gate_count', 'larva_count', 'n_power_source']

    int_vars = max_vars

    def __init__(self, stat_path, enemy_stat_path):
        if self.stat_path is None or not os.path.samefile(self.stat_path, stat_path):
            self.stat_path = stat_path
            self.stat = load_stat(stat_path)
        if self.enemy_stat_path is None or not os.path.samefile(self.enemy_stat_path, enemy_stat_path):
            self.enemy_stat_path = enemy_stat_path
            self.enemy_stat = load_stat(enemy_stat_path)

        for k in self.int_vars:
            setattr(self, k, -1)
        # Reward
        self.reward = -1
        self.score = None
        # Alerts
        self.alert = set()
        # Upgrades
        self.upgrades = set()
        # Actions
        self.action = -1
        self.research = {}
        # Units
        self.friendly_units = {}
        self.enemy_units = {}

    def update(self, state):
        for k in self.int_vars:
            setattr(self, k, state[k])
        # Reward
        self.reward = 2 - state['reward']
        self.score = state['score_cumulative']
        # Alert
        self.alert = set(state['alert'])
        # Upgrades
        self.upgrades = set(state['upgrades'])
        # Actions
        self.__set_action__(-1 if state['action'] is None else state['action'][0])
        ## Units
        # Friendly units
        self.friendly_units = self.__set_units__(state['friendly_units'])
        # Enemy units
        self.enemy_units    = self.__set_units__(state['enemy_units'])

    def get_action(self):
        return self.stat['action_id'][self.action]

    def __set_units__(self, units):
        results = {}
        for unit_type_id, unit in units.items():
            unit_type_id = int(unit_type_id)
            if unit_type_id not in results:
                results[unit_type_id] = {'built': [], 'building': []}
            for unit_instance in unit['units']:
                if unit_instance['build_progress'] >= 1:
                    results[unit_type_id]['built'].append(unit_instance)
                else:
                    results[unit_type_id]['building'].append(unit_instance)
        return results

    def __set_action__(self, action):
        self.action = action
        if action == -1:
            return
        if self.stat['action_name'][action].startswith('Research'):
            if action not in self.research:
                self.research[action] = 0
            self.research[action] += 1

    def __units2vec__(self, units, stat):
        name2id = {'total_num': 0, 'finished_num': 1, 'building_num': 2,
                   'max_building_progress': 3, 'min_building_progress': 4,
                   'avg_building_progress': 5}
        units_stat = stat['units_type']

        result = np.zeros(len(units_stat)*len(name2id))
        for k, unit in units.items():
            if k not in units_stat:
                continue
            start = units_stat[k] * len(name2id)
            result[start + name2id['total_num']] = len(unit['built'])+len(unit['building'])
            if result[start + name2id['total_num']] == 0:
                continue

            result[start + name2id['finished_num']] = len(unit['built'])
            result[start + name2id['building_num']] = len(unit['building'])

            if result[start + name2id['building_num']] > 0:
                ## Init min value
                result[start+name2id['min_building_progress']] = 1.0

                for u in unit['building']:
                    result[start + name2id['max_building_progress']] = \
                        max(result[start + name2id['max_building_progress']], u['build_progress'])
                    result[start + name2id['min_building_progress']] = \
                        min(result[start + name2id['min_building_progress']], u['build_progress'])
                    result[start + name2id['avg_building_progress']] += u['build_progress']

                result[start+name2id['avg_building_progress']] /= result[start+name2id['building_num']]

            for name in {'total_num', 'finished_num', 'building_num'}:
                result[start+name2id[name]] /= stat['max_unit_num']

        return result

    def __set_to_array__(self, set_var, key2id):
        result = np.zeros(len(key2id))
        for key in set_var:
            result[key2id[key]] = 1
        return result

    def __dict_to_array__(self, dict_var, key2id, scale=1):
        result = np.zeros(len(key2id))
        for key, value in dict_var.items():
            result[key2id[key]] = value/scale
        return result

    def to_vector(self):
        result = []
        # Frame_id; Resources
        max_result = []
        for k, v in zip(self.max_keys, self.max_vars):
            max_result.append(self.__dict__[v]/(self.stat['max_'+k]+self.eps))
        result.append(max_result)
        # Alerts
        result.append(self.__set_to_array__(self.alert, self.stat['alert']))
        # Upgrades
        result.append(self.__set_to_array__(self.upgrades, self.stat['upgrades']))
        # Research
        result.append(self.__dict_to_array__(self.research, self.stat['research_id'],
                                             self.stat['max_research_num']))
        ## Units
        result.append(self.__units2vec__(self.friendly_units, self.stat))
        result.append(self.__units2vec__(self.enemy_units, self.enemy_stat))

        return np.hstack(result)

    def __str__(self):
        return pprint.pformat(self.__dict__)
