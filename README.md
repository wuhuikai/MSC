# MSC
MSC: A Dataset for Macro-Management in StarCraft II.
## Download
**FTP:** [ftp://surveillance.idealtest.org/](ftp://surveillance.idealtest.org/) (User Name: msc; Password: msc)

- **[Stat]:** ftp://msc:msc@surveillance.idealtest.org/Stat.tar.gz

- **[TRAIN|VAL|TEST]：** ftp://msc:msc@surveillance.idealtest.org/TRAIN-VAL-TEST.tar.gz

- **[Global]:** ftp://msc:msc@surveillance.idealtest.org/GlobalFeatureVector.tar.gz

- **[Spatial]:**
    - TvT: ftp://msc:msc@surveillance.idealtest.org/TvT.tar.gz
    - TvP: ftp://msc:msc@surveillance.idealtest.org/PvT.tar.gz
    - TvZ: ftp://msc:msc@surveillance.idealtest.org/TvZ/TvZ.tar.gz
    - PvP: ftp://msc:msc@surveillance.idealtest.org/PvP.tar.gz
    - PvZ: ftp://msc:msc@surveillance.idealtest.org/PvZ.tar.gz
    - ZvZ: ftp://msc:msc@surveillance.idealtest.org/ZvZ.tar.gz

For **Linux** and **MacOS**:
```sh
wget ftp://msc:msc@surveillance.idealtest.org/[FILE_NAME]
```

**NOTE:**
- **Global** features are also available [HERE](https://drive.google.com/open?id=0Bybnpq8dvwudNUVOX1FCWnZoSGM).
- **[TRAIN|VAL|TEST]** split is also available [HERE](train_val_test).
- **[Stat]** is also available [HERE](parsed_replays/Stat). The stat files with postfix **_human.json** are human-readable.

## Baselines
### Global State Evaluation
| Method | TvT:T | TvZ:T | TvZ:Z | TvP:T | TvP:P | ZvZ:Z | ZvP:Z | ZvP:P | PvP:P |
| - | - | - | - | - | - | - | - | - | - |
| Baseline[Global] | 61.09 | 58.89 | 60.61 | 57.21 | 60.95 | 59.91 | 59.95 | 59.35 | 51.36 |
### Build Order Prediction
| Method | TvT:T | TvZ:T | TvZ:Z | TvP:T | TvP:P | ZvZ:Z | ZvP:Z | ZvP:P | PvP:P |
| - | - | - | - | - | - | - | - | - | - |
| - | - | - | - | - | - | - | - | - | - |
## Dataset: Global Feature Vector
Each replay is a **(T, M)** matrix **F**, where **F[t, :]** is the feature vector for time step **t**.

Each **row** of **F** is a **M**-dimensional vector, with **M** varying as **[RACE] v.s. [RACE]**.

The **M**-dimensional vector is orgnized as follows:
1. **[0]:** reward, i.e. final result of the game. **0**: DEFEAT, **1:** WIN.
2. **[1]:** ground truth action, ranging from **[0, #ACTION]**.
3. **[2-15):** cumulative score **[NOT NORMALIZED]**, which is defined in [Here](https://github.com/wuhuikai/MSC/blob/ebb1a722206e594e1c3a1da7cf21df8c514e5040/parse_replay/replay2global_features.py#L52).
4. **[15-M):** observation feature vector, which is normalized into **[0, 1]**.
    1. **[15]:** frame id.
    2. **[16-27):** player info, including various [resources](https://github.com/wuhuikai/MSC/blob/ebb1a722206e594e1c3a1da7cf21df8c514e5040/parse_replay/replay2global_features.py#L68) and **n_power_source**.
    3. **[27-#1):** alerts, **boolean**.
    4. **[#1-#2):** upgrades, **boolean**.
    5. **[#2-#3):** research count.
    6. **[#3-#4):** friendly units info, which is defined in [Here](https://github.com/wuhuikai/MSC/blob/ebb1a722206e594e1c3a1da7cf21df8c514e5040/extract_features/game_state.py#L110).
    7. **[#4-M):** enemy units info, where **M = #4 + #[ENEMY RACE]**.
         
        | V.S. | TvT:T | TvZ:T | TvZ:Z | TvP:T | TvP:P | ZvZ:Z | ZvP:Z | ZvP:P | PvP:P |
        | - | - | - | - | - | - | - | - | - | - |
        | M | 753 | 1131 | 1121 | 663 | 653 | 1499 | 1031 | 1031 | 563 |

        | RACE | #1 | #2 | #3 | #4 | #ACTION | #RACE |
        | - | - | - | - | - | - | - |
        | Terran | 29 | 60 | 81 | 417 | 75 | 336|
        | Protoss | 29 | 55 | 71 | 317 | 61 | 246 |
        | Zerg | 29 | 55 | 71 | 785 | 74 | 714 |
Code for loading **F**:
```python
import numpy as np
from scipy import sparse
F = np.asarray(sparse.load_npz(PATH).todense())
```
## Dataset: Spatial Feature Tensor
Each replay contains a **(T, 13, 64, 64)** tensor **S** and a **(T, 26)** matrix **G**.

The specifics for **S[t, :, :, :]** is as follows:
1. **S[t, 0:8, :, :]:** screen features, normalized into **[0-255]**, which is defined in [Here](https://github.com/wuhuikai/MSC/blob/ebb1a722206e594e1c3a1da7cf21df8c514e5040/extract_features/SpatialFeatures.py#L45).
2. **S[t, 8:13, :, :]:** minimap features, normalized into **[0-255]**, which is defined in [Here](https://github.com/wuhuikai/MSC/blob/ebb1a722206e594e1c3a1da7cf21df8c514e5040/extract_features/SpatialFeatures.py#L58).

**WARNING**[Cheat Layer]: The last layer **S[t, 12, :, :]** refers to **unit_type**, which could only be obtained in replays.

Code for loading **S**:
```python
import numpy as np
from scipy import sparse
S = np.asarray(sparse.load_npz(PATH).todense()).reshape([-1, 13, 64, 64])
```
The specifics for **G[t, :]** is as follows:
1. **[0-11):** frame id + player info, normalized into **[0, 1]**, which is defined [Here](https://github.com/wuhuikai/MSC/blob/ebb1a722206e594e1c3a1da7cf21df8c514e5040/extract_features/SpatialFeatures.py#L97).
2. **[11-24):** cumulative score **[NOT NORMALIZED]**, which is defined in [Here](https://github.com/wuhuikai/MSC/blob/ebb1a722206e594e1c3a1da7cf21df8c514e5040/extract_features/SpatialFeatures.py#L111).
3. **[24]:** reward, i.e. final result of the game. **0**: DEFEAT, **1:** WIN.
4. **[25]:** ground truth action, ranging from **[0, #ACTION]**.

Code for loading **G**:
```python
import numpy as np
from scipy import sparse
G = np.asarray(sparse.load_npz(PATH).todense())
```
## Build the Dataset Yourself Step by Step
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
### Step-by-Step Instructions
#### [The Easy Way](instructions/EasyWay.md)
#### [The Hard Way [Step-by-Step in Details]](instructions/HardWay.md)
### Requirements
```
future == 0.16.0

numpy == 1.13.0
scipy == 0.19.0

python_gflags == 3.1.1

tqdm == 4.14.0

protobuf == 3.4.0
pystream_protobuf == 1.4.4

PySC2 == 1.0
s2clientprotocol == 1.1
```
