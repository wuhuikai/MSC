python global_feature_vector.py --hq_replay_set $1 &&
python spatial_feature_tensor.py --hq_replay_set $1 &&
python split.py --hq_replay_set $1