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
**NOTE:**
1. **$STAR_CRAFT$/Replays** contains all ***.SC2Replay** files from **3.16.1 - Pack 1** and **3.16.1 - Pack 2**
2. **$STAR_CRAFT$/Battle.net** contains all contents from the folder **Battle.net** in **3.16.1 - Pack 1** and **3.16.1 - Pack 2**
### Preprocessing Replays
#### Parse Replay Info
```sh
python parse_replay_info.py
  --replays_paths $REPLAY_FOLDER_PATH_1$;$REPLAY_FOLDER_PATH_2$;...;$REPLAY_FOLDER_PATH_n$
  --save_path $SAVE_PATH$
  --n_instance [N_PROCESSES]
  --batch_size [BATCH_SIZE]
```
**NOTE:** Preparsed replay infos are available [HERE](https://drive.google.com/open?id=0Bybnpq8dvwudX1Z5MVp3THFnTlk).
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
**NOTE:** Prefiltered replay lists are available [HERE](https://drive.google.com/open?id=0Bybnpq8dvwudLWVlU1QtMmNyeE0).
#### Split Training, Validation and Test sets
```sh
python split.py
  --hq_replays_path $PREFILTERED_REPLAY_LIST_PATH$
  --save_path $SAVE_PATH$
  --ratio [TRAIN:VAL:TEST]
  --seed [RANDOM_SEED]
```