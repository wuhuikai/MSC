# MSC
MSC: A Dataset for Macro-Management in StarCraft II
## Step-by-Step Instructions on Ubuntu
### Install [SC2LE](https://github.com/Blizzard/s2client-proto)
1. Download and **unzip** (Password: iagreetotheeula) StarCraft II Linux Packages [3.16.1](http://blzdistsc2-a.akamaihd.net/Linux/SC2.3.16.1.zip) into **$STAR_CRAFT$**.
2. Download and **unzip** (Password: iagreetotheeula) Replay Packs ([3.16.1 - Pack 1](http://blzdistsc2-a.akamaihd.net/ReplayPacks/3.16.1-Pack_1-fix.zip), [3.16.1 - Pack 2](http://blzdistsc2-a.akamaihd.net/ReplayPacks/3.16.1-Pack_2.zip) [Currently not used]) into **$STAR_CRAFT$**.

After step 1 and step 2, the folder structure is as follows:
```
$STAR_CRAFT$
    ├── Battle.net
    ├── Libs
    ├── Maps
    ├── Replays
    ├── SC2Data
    └── Versions
```
- **NOTE:**
    1. **$STAR_CRAFT$/Replays** contains all ***.SC2Replay** files from **3.16.1 - Pack 1** and **3.16.1 - Pack 2** [Currently not used]
    2. **$STAR_CRAFT$/Battle.net** contains all contents from the folder **Battle.net** in **3.16.1 - Pack 1** and **3.16.1 - Pack 2** [Currently not used]
### Preprocessing Replays
#### Parse Replay Info
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
- **NOTE:** Pre-parsed replay infos are available [HERE](https://drive.google.com/open?id=0Bybnpq8dvwudX1Z5MVp3THFnTlk).
#### Filter Replays
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
- **NOTE:** Pre-filtered replay lists are available [HERE](https://drive.google.com/open?id=0Bybnpq8dvwudLWVlU1QtMmNyeE0).
### Parsing Replays
#### Extract Actions
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
- **NOTE:** The pre-extracted actions are available **NOW**.
    
    [Terran v.s. Terran](https://drive.google.com/open?id=0Bybnpq8dvwudY2x1aEdqc05KMXM) **|** [Terran v.s. Zerg]() **|** [Terran v.s. Protoss]() **|** [Zerg v.s. Zerg]() **|** [Zerg v.s. Protoss]() **|** [Protoss v.s. Protoss](https://drive.google.com/open?id=0Bybnpq8dvwudMTBsM1FCMmlVSHc)
#### Sample Actions
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
- **NOTE:** The pre-sampled actions are available **NOW**.
   
   [Terran v.s. Terran](https://drive.google.com/open?id=0Bybnpq8dvwudWGVWMFFmdmhmdWM) **|** [Terran v.s. Zerg]() **|** [Terran v.s. Protoss]() **|** [Zerg v.s. Zerg]() **|** [Zerg v.s. Protoss]() **|** [Protoss v.s. Protoss](https://drive.google.com/open?id=0Bybnpq8dvwudMFJxNlotM1NWQ0U)
#### Extract Sampled Observations
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
- **NOTE:** The pre-parsed global infos are available **NOW**.
    
    [Terran v.s. Terran]() **|** [Terran v.s. Zerg]() **|** [Terran v.s. Protoss]() **|** [Zerg v.s. Zerg]() **|** [Zerg v.s. Protoss]() **|** [Protoss v.s. Protoss]()
- **Code for reading SampledObservations files**
    ```python
    import stream
    from s2clientprotocol import sc2api_pb2 as sc_pb
    
    OBS =  [obs for obs in stream.parse(SAMPLED_OBSERVATION_PATH), sc_pb.ResponseObservation)]
    ```
    **ResponseObsevation** is defined [Here](https://github.com/Blizzard/s2client-proto/blob/4028f80aac30120f541e0e103efd63e921f1b7d5/s2clientprotocol/sc2api.proto#L329).
- **NOTE:** The pre-sampled observations are available **NOW**.

    [Terran v.s. Terran]() **|** [Terran v.s. Zerg]() **|** [Terran v.s. Protoss]() **|** [Zerg v.s. Zerg]() **|** [Zerg v.s. Protoss]() **|** [Protoss v.s. Protoss]()
### Split Training, Validation and Test sets
```sh
python split.py
  --hq_replay_set $PREFILTERED_REPLAY_LIST$
  --parsed_replay_path $PARSED_REPLAYS$
  --save_path $SAVE_PATH$
  --ratio [TRAIN:VAL:TEST]
  --seed [RANDOM_SEED]
```
- **Format of processed files [JSON]:**
    ```python
    [{"info_path": INFO_PATH,
      "sampled_action_path": SAMPLED_ACTION_PATH,
      RACE_1: [{"global_info_path": GLOBAL_INFO_PATH,
                "action_path": ACTION_PATH,
                "sampled_observation_path": SAMPLED_OBSERVATION_PATH}, ...],
      RACE_2: [{...}, ...]}, {...}, ...]
    ```
- **NOTE:** The pre-split training, validation and test sets are [**available**](https://github.com/wuhuikai/MSC/tree/master/train_val_test).
### Turn Parsed Replays into Global Feature Vectors
#### Extract Global Features
```sh
python replay2global_features.py:
  --hq_replay_set $PREFILTERED_REPLAY_LIST$
  --parsed_replay_path: $PARSED_REPLAYS$
  --step_mul [STEP_SIZE]
```
- **Format of processed files [JSON]:**
    ```python
    [state_1, state_2, ..., state_N]
    state_t = {...} [READ THE CODE or PRINT]
    ```
- **NOTE:** The pre-parsed global features are available **NOW**.

    [Terran v.s. Terran]() **|** [Terran v.s. Zerg]() **|** [Terran v.s. Protoss]() **|** [Zerg v.s. Zerg]() **|** [Zerg v.s. Protoss]() **|** [Protoss v.s. Protoss]()