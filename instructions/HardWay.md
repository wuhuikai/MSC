# Step-by-Step: Details
## Preprocessing Replays
```sh
cd preprocess
```
### Parse Replay Info
```sh
python parse_replay_info.py
  --replays_paths $REPLAY_FOLDER_PATH_1$;$REPLAY_FOLDER_PATH_2$;...;$REPLAY_FOLDER_PATH_N$
  --save_path $SAVE_PATH$
  --n_instance [N_PROCESSES]
  --batch_size [BATCH_SIZE]
```
- **Code for reading processed files:**
    ```python
    import json
    from google.protobuf.json_format import Parse
    from s2clientprotocol import sc2api_pb2 as sc_pb

    with open(REPLAY_INFO_PATH) as f:
        info = json.load(f)
    REPLAY_PATH = info['path']
    REPLAY_INFO_PROTO = Parse(info['info'], sc_pb.ResponseReplayInfo())
    ```
    **ResponseReplayInfo** is defined [Here](https://github.com/Blizzard/s2client-proto/blob/4028f80aac30120f541e0e103efd63e921f1b7d5/s2clientprotocol/sc2api.proto#L398).
### Filter Replays
```sh
python preprocess.py
  --infos_path $REPLAY_INFO_PATH$
  --save_path $SAVE_PATH$
  --min_duration [MIN_DURATION]
  --max_duration [MAX_DURATION]
  --min_apm [MIN_APM]
  --min_mmr [MIN_MMR]
```
- **Format of processed files [JSON]:**
    ```python
    [[REPLAY_PATH_1, REPLAY_INFO_PATH_1],
     [REPLAY_PATH_2, REPLAY_INFO_PATH_2],
     ...,
     [REPLAY_PATH_N, REPLAY_INFO_PATH_N]]
    ```
## Parsing Replays
```sh
cd parse_replay
```
### Extract Actions
```sh
python extract_actions.py
  --hq_replay_set $PREFILTERED_REPLAY_LIST$
  --save_path $SAVE_PATH$
  --n_instance [N_PROCESSES]
  --batch_size [BATCH_SIZE]
  --step_mul [STEP_SIZE]
  --width [WORLD_WIDTH]
  --map_size [MAP_SIZE]
```
- **Code for reading processed files:**
    ```python
    import json
    from google.protobuf.json_format import Parse
    from s2clientprotocol import sc2api_pb2 as sc_pb

    with open(ACTION_PATH) as f:
        actions = json.load(f)

    for actions_per_frame in actions:
        for action_str in actions_per_frame:
            action = Parse(action_str, sc_pb.Action())
    ```
    **Action** is defined [Here](https://github.com/Blizzard/s2client-proto/blob/4028f80aac30120f541e0e103efd63e921f1b7d5/s2clientprotocol/sc2api.proto#L553).
### Sample Actions
```sh
python sample_actions.py
  --hq_replay_set $PREFILTERED_REPLAY_LIST$
  --parsed_replays $PARSED_REPLAYS$
  --infos_path $REPLAY_INFOS$
  --step_mul [STEP_SIZE]
  --skip [SKIP_FRAMES]
```
- **Format of processed files [JSON]:**
    ```python
    [FRAME_ID_1, FRAME_ID_2, ..., FRAME_ID_N]
    ```
### Extract Sampled Observations
```sh
python parse_replay.py
  --hq_replay_set $PREFILTERED_REPLAY_LIST$
  --save_path $SAVE_PATH$
  --n_instance [N_PROCESSES]
  --batch_size [BATCH_SIZE]
  --width [WORLD_WIDTH]
  --map_size [MAP_SIZE]
```
- **Code for reading GlobalInfos files:**
    ```python
    import json
    from google.protobuf.json_format import Parse
    from s2clientprotocol import sc2api_pb2 as sc_pb

    with open(GLOBAL_INFO_PATH) as f:
        global_info = json.load(f)
    GAME_INFO = Parse(global_info['game_info'], sc_pb.ResponseGameInfo())
    DATA_RAW  = Parse(global_info['data_raw'], sc_pb.ResponseData())
    ```
    **ResponseGameInfo** is defined [Here](https://github.com/Blizzard/s2client-proto/blob/4028f80aac30120f541e0e103efd63e921f1b7d5/s2clientprotocol/sc2api.proto#L315) while **ResponseData** is defined [Here](https://github.com/Blizzard/s2client-proto/blob/4028f80aac30120f541e0e103efd63e921f1b7d5/s2clientprotocol/sc2api.proto#L367).
- **Code for reading SampledObservations files**
    ```python
    import stream
    from s2clientprotocol import sc2api_pb2 as sc_pb

    OBS =  [obs for obs in stream.parse(SAMPLED_OBSERVATION_PATH), sc_pb.ResponseObservation)]
    ```
    **ResponseObsevation** is defined [Here](https://github.com/Blizzard/s2client-proto/blob/4028f80aac30120f541e0e103efd63e921f1b7d5/s2clientprotocol/sc2api.proto#L329).
### Extract Global Features
```sh
python replay2global_features.py
  --hq_replay_set $PREFILTERED_REPLAY_LIST$
  --parsed_replay_path: $PARSED_REPLAYS$
  --step_mul [STEP_SIZE]
```
- **Format of processed files [JSON]:**
    ```python
    [state_1, state_2, ..., state_N]
    state_t = {...} [READ THE CODE or PRINT]
    ```
## Build Dataset
```sh
cd extract_features
```
### Compute Stat
```sh
python replay_stat.py
    --hq_replay_path $PREFILTERED_REPLAY_FOLDER$
    --parsed_replay_path $PARSED_REPLAYS$
    --race [RACE]
```
The stat files with postfix **_human.json** is human-readable.
### Extract Features
- Global Feature Vector
    ```sh
    python global_feature_vector.py
        --hq_replay_set $PREFILTERED_REPLAY_LIST$
        --parsed_replay_path: $PARSED_REPLAYS$
    ```
- Spatial Feature Tensor
    ```sh
    python spatial_feature_tensor.py
        --hq_replay_set $PREFILTERED_REPLAY_LIST$
        --parsed_replay_path: $PARSED_REPLAYS$
        --step_mul [STEP_SIZE]
        --n_workers [#PROCESSES]
    ```
### Split Training, Validation and Test sets
```sh
python split.py
  --hq_replay_set $PREFILTERED_REPLAY_LIST$
  --root $ROOT_PARSED_REPLAYS$
  --parsed_replay_path $PARSED_REPLAYS$
  --save_path $SAVE_PATH$
  --ratio [TRAIN:VAL:TEST]
  --seed [RANDOM_SEED]
```
- **Format of processed files [JSON]:**
    ```python
    [{RACE_1: [{"global__path": GLOBAL_FEATURE_PATH,
                "spatial_path_S": SPATIAL_FEATURE_PATH_S,
                "spatial_path_G": SPATIAL_FEATURE_PATH_G}, ...],
      RACE_2: [{...}, ...]}, {...}, ...]
    ```
- **NOTE:** The pre-split training, validation and test sets are available in [**Here**](https://github.com/wuhuikai/MSC/tree/master/train_val_test).