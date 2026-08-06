# Corrected Farm Group Classification (2026-07-30)

## Issue
Previous data report used non-exclusive groups: "Europe within 500km (37)" and "UK/Nordic beyond 400km (40)".
These groups overlap because 14 UK farms counted in "37" are also in "40".

## Corrected mutually-exclusive groups (total = 171)

| Group | Farms | Capacity | Description |
|-------|-------|----------|-------------|
| Europe (direction data) | 37 | 44.1 GW | Within 500km of nlhrw/deess/behel (3 radars with extracted direction signatures) |
| Europe (no direction data) | 41 | 24.1 GW | Beyond 500km from data radars: UK(17), SE(8), DK(6), FR(4), DE(2), FI(1), IT(1), IE(1), PT(1) |
| East Asia | 88 | 81.8 GW | China(66), Vietnam(13), Taiwan(4), Japan(3), South Korea(2) |
| Americas | 5 | 0.8 GW | USA(5) |

## Note on "37"
- 37 farms are within 500km of the 3 radars with direction data (nlhrw, deess, behel)
- 55 farms are within 500km of ANY of the 10 VPTS stations
- The difference (18) are near stations whose data files exist but haven't been processed for direction extraction

## File
farm_groups_corrected.csv contains all 171 farms with:
- data_radar: nearest radar WITH direction data
- data_dist: distance to that radar (km)
- all_radar: nearest of all 10 VPTS stations
- all_dist: distance to nearest VPTS station
- group: mutually-exclusive regional group
