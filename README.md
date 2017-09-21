# MSC
MSC: A Dataset for Macro-Management in StarCraft II.
## Download
**[Global]:** [Terran v.s. Terran]() **|** [Terran v.s. Zerg]() **|** [Terran v.s. Protoss]() **|** [Zerg v.s. Zerg]() **|** [Zerg v.s. Protoss]() **|** [Protoss v.s. Protoss]()
**[Spatial]:** [Terran v.s. Terran]() **|** [Terran v.s. Zerg]() **|** [Terran v.s. Protoss]() **|** [Zerg v.s. Zerg]() **|** [Zerg v.s. Protoss]() **|** [Protoss v.s. Protoss]()
**[Global+Spatial]:** [Terran v.s. Terran]() **|** [Terran v.s. Zerg]() **|** [Terran v.s. Protoss]() **|** [Zerg v.s. Zerg]() **|** [Zerg v.s. Protoss]() **|** [Protoss v.s. Protoss]()
## Dataset: Global Feature Vector
Each replay is a **(T, M)** matrix **F**, where **F[t, :]** is the feature vector for time step **t**.

Each **row** of **F** is a **M**-dimensional vector, with **M** varying as **[RACE] v.s. [RACE]**.

The **M**-dimensional vector is orgnized as follows:
1. **[0]:** reward, i.e. final result of the game. **0**: DEFEAT, **1:** WIN.
2. **[1]:** ground truth action, ranging from **[0, #ACTION]**.
3. **[2-15):** cumulative score **[NOT NORMALIZED]**, which is defined in [Here](https://github.com/wuhuikai/MSC/blob/1947d3e17dde13890ec5ba03c1f616d7bbcd175e/SpatialFeatures.py#L113).
4. **[15-M):** observation feature vector, which is normalized into **[0, 1]**.
    1. **[15]:** frame id.
    2. **[16-27):** player info, including various [resources]((https://github.com/wuhuikai/MSC/blob/1947d3e17dde13890ec5ba03c1f616d7bbcd175e/replay2global_features.py#L68)) and **n_power_source**.
    3. **[27-#1):** alerts, **boolean**.
    4. **[#1-#2):** upgrades, **boolean**.
    5. **[#2-#3):** research count.
    6. **[#3-#4):** friendly units info, which is defined in [Here](https://github.com/wuhuikai/MSC/blob/1947d3e17dde13890ec5ba03c1f616d7bbcd175e/game_state.py#L110).
    7. **[#4-M):** enemy units info, where **M = #4 + #[ENEMY RACE]**.
        
        | RACE | #1 | #2 | #3 | #4 | #ACTION | #[RACE] |
        | - | - | - | - | - | - | - |
        | Terran | - | - | - | - | - | -|
        | Protoss | - | - | - | - | - | - |
        | Zerg | - | - | - | - | - | - |
Code for loading **F**:
```python
import numpy as np
from scipy import sparse
F = np.asarray(sparse.load_npz(PATH).todense())
```
## Dataset: Spatial Feature Tensor
Each replay contains a **(T, 13, 64, 64)** tensor **S** and a **(T, 26)** matrix **G**.

The specifics for **S[t, :, :, :]** is as follows:
1. **S[t, 0:8, :, :]:** screen features, normalized into **[0-255]**, which is defined in [Here](https://github.com/wuhuikai/MSC/blob/1947d3e17dde13890ec5ba03c1f616d7bbcd175e/SpatialFeatures.py#L45).
2. **S[t, 8:13, :, :]:** minimap features, normalized into **[0-255]**, which is defined in [Here](https://github.com/wuhuikai/MSC/blob/1947d3e17dde13890ec5ba03c1f616d7bbcd175e/SpatialFeatures.py#L58).

**WARNING**[Cheat Layer]: The last layer **S[t, 12, :, :]** refers to **unit_type**, which could only be obtained in replays.

Code for loading **S**:
```python
import numpy as np
from scipy import sparse
S = np.asarray(sparse.load_npz(PATH).todense()).reshape([-1, 13, 64, 64])
```
The specifics for **G[t, :]** is as follows:
1. **[0-11):** frame id + player info, normalized into **[0, 1]**, which is defined [Here](https://github.com/wuhuikai/MSC/blob/1947d3e17dde13890ec5ba03c1f616d7bbcd175e/SpatialFeatures.py#L99).
2. **[11-24):** cumulative score **[NOT NORMALIZED]**, which is defined in [Here](https://github.com/wuhuikai/MSC/blob/1947d3e17dde13890ec5ba03c1f616d7bbcd175e/SpatialFeatures.py#L113).
3. **[24]:** reward, i.e. final result of the game. **0**: DEFEAT, **1:** WIN.
4. **[25]:** ground truth action, ranging from **[0, #ACTION]**.

Code for loading **G**:
```python
import numpy as np
from scipy import sparse
G = np.asarray(sparse.load_npz(PATH).todense())
```
## Baselines
### Global State Evaluation

## Install [SC2LE](https://github.com/Blizzard/s2client-proto)
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
## Step-by-Step Instructions (The Easy Way)
### Preprocess
```sh
cd preprocess
bash preprocess.sh
```
### Parse Replay
```sh
cd parse_replay
bash parse_replay.sh $HQ_REPLAY_LIST$ [N_PROCESSES]
```
For example:
```sh
cd parse_replay
bash parse_replay.sh ../high_quality_replays/Protoss_vs_Terran.json 32
```
### Build Dataset
```sh
cd extract_features
```
#### Compute Stat
```sh
bash compute_stat.sh [RACE]
```
For example:
```sh
bash compute_stat.sh Terran
```
#### Extract Features
```sh
bash extract_features.sh $HQ_REPLAY_LIST$
```
For example:
```sh
bash extract_features.sh ../high_quality_replays/Protoss_vs_Terran.json
```
## Step-by-Step Instructions on Ubuntu
### Preprocessing Replays
```sh
cd preprocess
```
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
### Parsing Replays
```sh
cd parse_replay
```
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
- **Code for reading SampledObservations files**
    ```python
    import stream
    from s2clientprotocol import sc2api_pb2 as sc_pb
    
    OBS =  [obs for obs in stream.parse(SAMPLED_OBSERVATION_PATH), sc_pb.ResponseObservation)]
    ```
    **ResponseObsevation** is defined [Here](https://github.com/Blizzard/s2client-proto/blob/4028f80aac30120f541e0e103efd63e921f1b7d5/s2clientprotocol/sc2api.proto#L329).
#### Extract Global Features
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
### Build Dataset
```sh
cd extract_features
```
#### Compute Stat
```sh
python replay_stat.py
    --hq_replay_path $PREFILTERED_REPLAY_FOLDER$
    --parsed_replay_path $PARSED_REPLAYS$
    --race [RACE]
```
The stat files with postfix **_human.json** is human-readable.
#### Extract Features
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
    ```
#### Split Training, Validation and Test sets
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
