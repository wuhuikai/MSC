from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import json
import time
import signal
import threading
import queue as Queue
import multiprocessing
import gflags as flags
from future.builtins import range

from google.protobuf.json_format import MessageToJson

from pysc2 import run_configs
from pysc2.lib import app
from pysc2.lib import point
from s2clientprotocol import sc2api_pb2 as sc_pb

FLAGS = flags.FLAGS
flags.DEFINE_string(name='hq_replay_set', default='high_quality_replays/Terran_vs_Terran.json',
                    help='File storing replays list')
flags.DEFINE_string(name='save_path', default='parsed_replays',
                    help='Path for saving results')

flags.DEFINE_integer(name='n_instance', default=16,
                     help='# of processes to run')
flags.DEFINE_integer(name='step_mul', default=8,
                     help='step size')
flags.DEFINE_integer(name='batch_size', default=300,
                     help='# of replays to process in one iter')
flags.DEFINE_integer(name='width', default=24,
                     help='World width')
flags.DEFINE_integer(name='map_size', default=64,
                     help='Map size')

FLAGS(sys.argv)
size = point.Point(FLAGS.map_size, FLAGS.map_size)
interface = sc_pb.InterfaceOptions(raw=True, score=False,
                                   feature_layer=sc_pb.SpatialCameraSetup(width=FLAGS.width))
size.assign_to(interface.feature_layer.resolution)
size.assign_to(interface.feature_layer.minimap_resolution)

class ReplayProcessor(multiprocessing.Process):
    """A Process that pulls replays and processes them."""
    def __init__(self, run_config, replay_queue, counter, total_num):
        super(ReplayProcessor, self).__init__()
        self.run_config = run_config
        self.replay_queue = replay_queue
        self.counter = counter
        self.total_num = total_num

    def run(self):
        signal.signal(signal.SIGTERM, lambda a, b: sys.exit())  # Exit quietly.
        while True:
            with self.run_config.start() as controller:
                for _ in range(FLAGS.batch_size):
                    try:
                        replay_path = self.replay_queue.get()
                    except Queue.Empty:
                        return

                    try:
                        with self.counter.get_lock():
                            self.counter.value += 1
                            print('Processing {}/{} ...'.format(self.counter.value, self.total_num))

                        replay_data = self.run_config.replay_data(replay_path)
                        info = controller.replay_info(replay_data)
                        map_data = None
                        if info.local_map_path:
                            map_data = self.run_config.map_data(info.local_map_path)

                        for player_info in info.player_info:
                            race = sc_pb.Race.Name(player_info.player_info.race_actual)
                            player_id = player_info.player_info.player_id

                            if os.path.isfile(os.path.join(FLAGS.save_path, race,
                                                           '{}@{}'.format(player_id, os.path.basename(replay_path)))):
                                continue

                            self.process_replay(controller, replay_data, map_data, player_id, race, replay_path)
                    except Exception as e:
                        print(e)
                    finally:
                        self.replay_queue.task_done()

    def process_replay(self, controller, replay_data, map_data, player_id, race, replay_path):
        controller.start_replay(sc_pb.RequestStartReplay(
            replay_data=replay_data,
            map_data=map_data,
            options=interface,
            observed_player_id=player_id))

        save_folder = os.path.join(FLAGS.save_path, race)
        actions = []
        controller.step()
        while True:
            obs = controller.observe()
            actions.append([MessageToJson(a) for a in obs.actions])

            if obs.player_result:
                with open(os.path.join(save_folder, '{}@{}'.format(
                        player_id, os.path.basename(replay_path))), 'w') as f:
                    json.dump(actions, f)
                return
            controller.step(FLAGS.step_mul)

def replay_queue_filler(replay_queue, replay_list):
    """A thread that fills the replay_queue with replay filenames."""
    for replay_path in replay_list:
        replay_queue.put(replay_path)

def main(unused_argv):
    race_vs_race = os.path.basename(FLAGS.hq_replay_set).split('.')[0]
    FLAGS.save_path = os.path.join(FLAGS.save_path, 'Actions', race_vs_race)

    for race in set(race_vs_race.split('_vs_')):
        path = os.path.join(FLAGS.save_path, race)
        if not os.path.isdir(path):
            os.makedirs(path)

    run_config = run_configs.get()
    try:
        with open(FLAGS.hq_replay_set) as f:
            replay_list = json.load(f)
        replay_list = sorted([p for p, _ in replay_list])

        replay_queue = multiprocessing.JoinableQueue(FLAGS.n_instance * 10)
        replay_queue_thread = threading.Thread(target=replay_queue_filler,
                                           args=(replay_queue, replay_list))
        replay_queue_thread.daemon = True
        replay_queue_thread.start()

        counter = multiprocessing.Value('i', 0)
        for i in range(FLAGS.n_instance):
            p = ReplayProcessor(run_config, replay_queue, counter, len(replay_list))
            p.daemon = True
            p.start()
            time.sleep(1)   # Stagger startups, otherwise they seem to conflict somehow

        replay_queue.join() # Wait for the queue to empty.
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, exiting.")

if __name__ == '__main__':
    app.run()