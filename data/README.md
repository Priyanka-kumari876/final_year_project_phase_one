# PAMAP2 Dataset Organization

This directory stores raw and processed data for the research project:
**"P10: Multimodal Acquisition with Sensor Wake-Up Costs and Outages"**.

## Dataset Origin
- **Name**: PAMAP2 Physical Activity Monitoring Dataset
- **Authors**: Attila Reiss and Didier Stricker (DFKI / University of Kaiserslautern, 2012)
- **Source**: UCI Machine Learning Repository
  `https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring`

## Local Storage Layout
The local workspace contains the original extracted files located in:
`pamap2+physical+activity+monitoring/PAMAP2_Dataset/PAMAP2_Dataset/`
- `Protocol/`: Contains `subject101.dat` through `subject109.dat`.
- `Optional/`: Contains optional activity recordings (`subject101.dat`, `105`, `106`, `108`, `109`).
- Documentation PDFs: `readme.pdf`, `DataCollectionProtocol.pdf`, `DescriptionOfActivities.pdf`, `PerformedActivitiesSummary.pdf`, `subjectInformation.pdf`.

## Raw Data Columns (54 Columns)
1. `timestamp` (s)
2. `activityID` (integer, 0 to 24)
3. `heart_rate` (bpm, sampled at ~9 Hz; missing values recorded as NaN)
4-20. `IMU hand` (17 channels: temp, 3D-acc ±16g, 3D-acc ±6g, 3D-gyro, 3D-mag, 4D-orientation [invalid])
21-37. `IMU chest` (17 channels)
38-54. `IMU ankle` (17 channels)

## Physical Sensor Groups Selected for P10
We study 3 physical sensor modalities ($2^3 = 8$ subsets):
1. **Hand IMU** (`hand_acc_x, y, z`, `hand_gyro_x, y, z`, `hand_mag_x, y, z`, `hand_temp`)
2. **Ankle IMU** (`ankle_acc_x, y, z`, `ankle_gyro_x, y, z`, `ankle_mag_x, y, z`, `ankle_temp`)
3. **Heart-Rate Monitor** (`heart_rate`)

*Note: Chest IMU is retained in raw format but excluded from the 3 primary sensor groups in accordance with the project specification.*

## Valid Activities (12 Protocol Activities)
- 1: Lying
- 2: Sitting
- 3: Standing
- 4: Walking
- 5: Running
- 6: Cycling
- 7: Nordic walking
- 12: Ascending stairs
- 13: Descending stairs
- 16: Vacuum cleaning
- 17: Ironing
- 24: Rope jumping

*Activity ID 0 represents transient/unlabelled activity and is strictly discarded before analysis.*
