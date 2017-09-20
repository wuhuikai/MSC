python extract_actions.py --hq_replay_set $1 --n_instance $2 &&
python sample_actions.py --hq_replay_set $1 &&
python parse_replay.py --hq_replay_set $1 --n_instance $2 &&
python replay2global_features.py --hq_replay_set $1