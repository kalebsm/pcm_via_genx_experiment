# NYISO Test System for GenX

This repository contains data and scripts to create a detailed model of the New York Independent System Operator (NYISO) transmission grid for capacity expansion modeling in GenX.

## System Overview

The NYISO test system represents the 11-zone transmission network (zones A through K) with actual generation resources, transmission constraints, and hourly variability based on 2019 data. This model can be used for studying renewable integration, transmission expansion, storage deployment, and policy analysis in the New York power system.

## Network Structure

- **Zones**: 11 zones (A-K) representing NYISO control areas
- **Transmission Lines**: 12 interconnections between zones with maximum transfer limits (no DCOPF, transmission expansion)
- **Data Source**: Original transfer limits from Hibbard et al. reference case, with adjustments for separating zones H and I
- 
| Zone | Transfer Limit (MW) |
|------|---------------------|
| A<>B | 5133                |
| B<>C | 1600                |
| C<>D | 8432                |
| D<>E | 4161                |
| E<>F | 3600                |
| F<>G | 9279                |
| G<>H | 4600                |
| HI<>J| 14,713              |
| J<>K | 8675                |
| K<>L | 4520                |
| L<>M | 300                 |

## Resources by Category
Starting Capacities: 
| Renewable Resource  | A    | B    | C    | D    | E    | F    | G    | H    | I    | J    | K    |
|---------------------|------|------|------|------|------|------|------|------|------|------|------|
| Land-Based Wind     | 2692 | 390  | 1923 | 1935 | 1821 | 1864 | 606  | 303  | 0    | 0    | 121  |
| Offshore Wind       | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 0    | 8250 | 6488 |
| BTM Solar           | 1297 | 402  | 1098 | 127  | 1240 | 2154 | 2270 | 202  | 299  | 1676 | 2883 |
| Utility Solar       | 14440| 1648 | 9006 | 0    | 5698 | 15647| 3353 | 0    | 0    | 0    | 1441 |
| Hydro               | 2460 | 63.8 | 109.4| 909.8| 376.3| 269.6| 75.8 | 0    | 0    | 0    | 0    |
| Battery             | 2479 | 10   | 2538 | 2562 | 892  | 4727 | 150  | 140  | 140  | 4263 | 1924 |
| Nuclear             | 0    | 0    | 581.7| 2783 | 0    | 0    | 0    | 0    | 0    | 0    | 0    |

| Thermal Resource  | A       | B      | C      | D      | E      | F      | G      | H    | I    | J      | K      |
|---------------------|---------|--------|--------|--------|--------|--------|--------|------|------|--------|--------|
| Combined Cycle      | 882.9   | 134.3  | 5477.9 | 673.3  | 236.0  | 5773.0 | 770.0  | 0    | 0    | 3261.2 | 945.2  |
| Combustion Turbine  | 47.3    | 0      | 0      | 0      | 0      | 0      | 0      | 0    | 0    | 470.0  | 500.9  |
| Steam Turbine       | 655.1   | 0      | 322.5  | 0      | 0      | 0      | 0      | 0    | 0    | 1526.0 | 376.0  |
| Steam Turbine (Coal)| 50.0    | 0      | 1803.6 | 0      | 0      | 0      | 0      | 0    | 0    | 0      | 0      |
| Steam Turbine (Refuse)| 50.0  | 0      | 0      | 0      | 0      | 0      | 0      | 0    | 0    | 0      | 0      |
| Steam Turbine (Fuel Oil 6)| 0  | 0      | 0      | 0      | 0      | 0      | 1242.0 | 0    | 0    | 0      | 0      |
| Steam Turbine (Natural Gas) | 0  | 0      | 0      | 0      | 0      | 0      | 1774.0 | 0    | 0    | 1447.9 | 1924.0 |
| Jet Engine (Natural Gas)  | 0     | 0      | 0      | 0      | 0      | 0      | 88.4   | 0    | 0    | 0      | 60.5   |
| Jet Engine (Kerosene)    | 0     | 0      | 0      | 0      | 0      | 0      | 0      | 0    | 0    | 1116.0 | 60.5   |
| Jet Engine (Fuel Oil 2)  | 0     | 0      | 0      | 0      | 0      | 0      | 0      | 0    | 0    | 0      | 621.0  |

Note: Steam Turbine (Refuse) and Jet Engine (Kerosene) currently excluded due to lack of fuel price data
### Thermal Resources

- **Data Source**: thermalGenMatched_2019.csv, thermalGenHourly_2019.csv, groupedThermals.csv
- **Technologies**: Combined Cycle, Combustion Turbine, Steam Turbine, Jet Engine
- **Fuels**: Natural Gas, Coal, Fuel Oil 2, Fuel Oil 6, Kerosene
- **Variability**: Based on actual 2019 hourly generation data, aggregated by technology, fuel type, and zone
- **Costs**: Investment and O&M costs from ATB data with zonal scaling factors from NYISO Net CONE curves :
  - Base (NYCA, Zones A-F): 1.0
  - G-J (Zones G-I): 1.31
  - NYC (Zone J): 1.72
  - Long Island (Zone K): 1.40

### Nuclear Resources

- **Data Source**: nuclearGenDaily_2019.csv
- **Plants**: Fitzpatrick, Ginna, Indian Point 2 & 3 (excluded), Nine Mile Point 1 & 2
- **Zones**: B (Ginna), C (Fitzpatrick, Nine Mile), H (Indian Point - excluded)
- **Variability**: Based on daily capacity factors expanded to hourly resolution

### VRE Resources

- **Data Source**: merged_2019.csv (wind and solar)
- **Technologies**: Land-based Wind, Offshore Wind, Utility Solar, BTM Solar
- **Variability**: Hourly capacity factors from 2019 data


### Hydro Resources

- **Data Source**: hydroGenMonthly_2019.csv
- **Plants**: Robert Moses Niagara (Zone A), St. Lawrence (Zone D)
- **Variability**: Monthly capacity factors expanded to hourly resolution
- **Parameters**: Min power 10%, ramp rates 50%, energy-to-power ratio 12 hours

### Storage Resources

- **Data Source**: System data, ATB_Data.csv (costs)
- **Technology**: Battery Energy Storage Systems (4-hour duration)
- **Parameters**: 92% round-trip efficiency, 0.04% hourly self-discharge

### Allowed New Builds (New_Build = 1)
- **Land-based Wind** in all zones where present (zones A-H, K)
- **Offshore Wind** (zones J and K)
- **Utility Solar** in all zones where present (zones A-C, E-G, K)
- **BTM Solar** in all zones
- **Battery storage** in all zones (4-hour duration systems)
- **Combined Cycle** plants (in all zones)
- **Combustion Turbine** plants (where present)

### Resources NOT Allowed New Builds (New_Build = 0):
- **Nuclear** plants
- **Coal** plants
- **Steam Turbine** plants
- **Hydro** plants

### Build Limits:
- The build limits for renewable and storage resources were typically set to **3x the existing capacity**, allowing for significant expansion while maintaining reasonable constraints.
- For thermal resources, **Combined Cycle** technology was specifically configured to allow new builds, while older technologies like **coal** and **nuclear** were restricted to retirement-only options, reflecting realistic future planning scenarios.

## Fuel Data

- **Data Source**: fuelPriceWeekly_2019.csv
- **Fuels**: Natural gas (by region), fuel oil, coal
- **Resolution**: Weekly prices expanded to hourly (8760 hours)
- **CO2 Intensity**: Specific emissions factors for each fuel type (tons/MMBtu)

## Key Assumptions and Augmentations

1. **Thermal Generators**:
   - Aggregated to technology-fuel-zone combinations
   - Heat rates based on actual 2019 performance data
   - Technical parameters (min power, ramp rates) differentiated by technology

2. **Renewable Resources**:
   - Wind and solar capacity factors normalized from actual 2019 data
   - Build limits set to 3x existing capacity for expansion

3. **Storage**:
   - 4-hour battery systems with symmetric charge/discharge
   - Costs split between power component (60%) and energy component (40%)
   - Duration limits between 2-8 hours

4. **Transmission**:
   - Connection between Zones H and I estimated at 5,000 MW
   - G-H connection estimated at 7,356 MW (half of original G-HI capacity)

5. **Cost Scaling**:
   - All resource costs scaled by zone to reflect regional cost differences
   - Based on normalized gross CONE values with NYCA as baseline

6. **Variability Data**:
   - Resources without specific time series data default to 100% availability
   - Thermal variability weighted based on actual hourly generation divided by nameplate capacity
   - Hydro and nuclear capacity factors expanded from coarser time resolution data

## Directory Structure

```
NYISO_Case
├── settings/
│   ├── genx_settings.yml
│   ├── gurobi_settings.yml
├── system/
│   ├── Demand_data.csv
│   ├── Fuel_data.csv
│   ├── Generators_variability.csv
│   └── Network.csv
├── resources/
│   ├── Thermal.csv
│   ├── Storage.csv
│   ├── Vre.csv
│   └── Hydro.csv
└── README.md
```

## Usage

To use this model with GenX:

1. Clone this repository to your local machine
2. Ensure you have Julia and GenX installed
3. Run the model using the provided Run.jl script

## Data Sources

- Baseline data from [Anderson Lab NYISO case](https://github.com/AndersonEnergyLab-Cornell/NYgrid)
- Load, Wind, Solar forecasts and actuals from NREL ARPA-E database
- Investment cost data from NREL ATB (ATB_Data.csv)
- Starting capacities for renewables from Liu et. al. (2024) with nuclear capacities from Hibbard et al. reference case.
- Starting Capacities for thermal resources aggregated from NYCA 2019 nameplate capacities

## Acknowledgments

This test system was developed for capacity expansion modeling research. The data processing scripts and model configuration were created to facilitate reproducible energy system modeling studies.
