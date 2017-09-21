# Step-by-Step: The Easy Way
## Preprocess
```sh
cd preprocess
bash preprocess.sh
```
## Parse Replay
```sh
cd parse_replay
bash parse_replay.sh $HQ_REPLAY_LIST$ [N_PROCESSES]
```
For example:
```sh
cd parse_replay
bash parse_replay.sh ../high_quality_replays/Protoss_vs_Terran.json 32
```
## Build Dataset
```sh
cd extract_features
```
### Compute Stat
```sh
bash compute_stat.sh [RACE]
```
For example:
```sh
bash compute_stat.sh Terran
```
### Extract Features
```sh
bash extract_features.sh $HQ_REPLAY_LIST$
```
For example:
```sh
bash extract_features.sh ../high_quality_replays/Protoss_vs_Terran.json
```