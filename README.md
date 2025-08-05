
DOI: 10.5281/zenodo.16748244
# lastname-etal_year_journal (to be included when published)

**your Paper Title here (once published, include a link to the text)**

First Last<sup>1\*</sup>, First Last<sup>1</sup>,  and First Last<sup>1, 2</sup>

<sup>1 </sup>Pacific Northwest National Laboratory, Richland, WA, USA.

<sup>2 </sup> Institute for Energy Analysis, Oak Ridge Associated Universities, Washington, DC, USA

\* corresponding author:  email@myorg.gov


# Abstract
This paper considers the impact of operational uncertainty on spot price distributions and resource valuation in power markets as systems transition to high levels of wind, solar, and storage resources. In systems dominated by thermal resources with minimal storage, marginal costs and therefore market clearing prices are driven by fuel costs, enabling modelers to rely on simplified deterministic formulations to approximate price behavior. Future price formation may instead be increasingly driven by storage opportunity costs, necessitating a more sophisticated representation of uncertainty and intertemporal constraints. In this paper we describe the mechanics of price formation with storage, employ a stochastic production cost model to simulate prices in modeled future systems, and highlight the differences between the simulated revenues and those that arise in deterministic simplifications. We perform numerical tests using a plausible representation of short-term forecast uncertainty and assess the economic impact of this uncertainty across cases with different battery durations and quantities.

# Journal reference
(to be included when published)

## Code reference
Bonaldo, L., S. Chakrabarti, F. Cheng, Y. Ding, J. Jenkins, Q. Luo, R. Macdonald, D. Mallapragada, A. Manocha, G. Mantegna, J. Morris, N. Patankar, F. Pecci, A. Schwartz, J. Schwartz, G. Schivley, N. Sepulveda, and Q. Xu (2024, April). GenX. [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15865702.svg)](https://doi.org/10.5281/zenodo.15865702)

Mirletz, Brian, Bannister, Michael, Vimmerstedt, Laura, Stright, Dana, and Heine, Matthew. ATB-calc (Annual Technology Baseline Calculators) [SWR-23-60]. Computer Software. https://github.com/NREL/ATB-calc. USDOE Office of Strategic Programs. 02 Aug. 2023. Web. doi:10.11578/dc.20230914.2.

Note that a forked version of GenX v0.4.4 is used to develop the lookahead production cost model experiments and is given the name PCMviaGenX and is included in the DOI for the entire experiment.


## Data reference

### Input data

Krall, E., M. Higgins, and R. O’Neill (2012, July). RTO unit commitment test system. Technical report.

National Renewable Energy Laboratory (2024). 2024 annual technology baseline. Available at https://atb.nrel.gov/.

## Contributing modeling software
| Model | Version | Repository Link | DOI |
|-------|---------|-----------------|-----|
| ATB-calc | v0.0 | [link to code repository](https://github.com/NREL/ATB-calc) | [link to DOI dataset ](https://doi.org/10.11578/dc.20230914.2)|
| GenX | v0.4.4 | [link to code repository](https://github.com/kalebsm/GenX/tree/caee5563bc1118c24fe99a30e1ed72e052191478) \ [link to original code repository](https://github.com/GenXProject/GenX.jl) | [link to DOI dataset](https://zenodo.org/records/15865702) |
| SequentialNorta | v0.0 | [link to code repository](https://github.com/kalebsm/sequential_norta.git) | no DOI currently |

## Program requirements
Running the scripts and computations will require the installation of the following:
1. A python version 3.10 or higher
2. A julia version 1.11 or higher
3. Gurobi license 9.4 or higher

Additionally, these experiments were originally run on an Intel i7-10700K CPU, 3.7 Ghz machine with 8 GB RAM. The total time for all 11 cases of the experiment to run on that machine was approximately 33 hours, or 3 hours per experiment (without parallelization).

## Reproduce my experiment
1. open a git bash terminal in the desired folder and enter: `git clone --recurse-submodules https://github.com/kalebsm/pcm_via_genx_experiment.git`
2. download the following ERCOT data from NREL ARPA-E PERFORM dataset and save in `spcm_genx_experiment\SPCM\src\scenario_generation\sequential_norta\data`

| #   | File Name                                | Data Type | Forecast Type         | Location Path                                               |
|-----|-------------------------------------------|-----------|------------------------|-------------------------------------------------------------|
| 1   | BA_load_actuals_2018.h5                    | Load      | Actuals                | ERCOT/2018/Load/Actuals/BA_level/                           |
| 2   | BA_load_day-ahead_fcst_2018.h5            | Load      | Day-ahead Forecast     | ERCOT/2018/Load/Forecast/Day-ahead/BA_level/               |
| 3   | BA_solar_actuals_Existing_2018.h5         | Solar     | Actuals                | ERCOT/2018/Solar/Actuals/BA-level/                          |
| 4   | BA_solar_2day-ahead_fcst_Existing_2018.h5 | Solar     | 2-Day Ahead Forecast   | ERCOT/2018/Solar/2Day_ahead/BA_level/                      |
| 5   | BA_solar_day-ahead_fcst_Existing_2018.h5  | Solar     | Day-ahead Forecast     | ERCOT/2018/Solar/Day-ahead/BA_level/                       |
| 6   | BA_wind_actuals_Existing_2018.h5         | Wind     | Actuals                | ERCOT/2018/Wind/Actuals/BA-level/                          |
| 7   | BA_wind_2day-ahead_fcst_Existing_2018.h5  | Wind      | 2-Day Ahead Forecast   | ERCOT/2018/Wind/2Day_ahead/BA_level/                       |
| 8   | BA_wind_day-ahead_fcst_Existing_2018.h5   | Wind      | Day-ahead Forecast     | ERCOT/2018/Wind/Day-ahead/BA_level/                        |

3. Save the ATB data (2024 v2 Annual Technology Baseline Workbook Errata 7-19-2024.xlsx) to the folder `data/` from the following link: (https://atb.nrel.gov/electricity/2024/data) or in the ATB archives.

4. Download Gurobi Academic license https://www.gurobi.com/account

5. Set up Python virtual environment using Git Bash with the following

| Step | Git Bash Command                                   | Description                  |
|------|----------------------------------------------------|------------------------------|
| 1  | `python -m venv venv`                              | Create a virtual environment |
| 2  | `chmod +x venv_setup.sh`                           | Make the setup script executable |
| 3  | `./venv_setup.sh`                                  | Run the setup script         |
| 4  | `source ./venv/Scripts/activate`                   | Activate the environment     |
| 5  | `python scripts/sge_model_setup/sge_model_setup.py`| Run the experiment setup script   |
| 6  | `deactivate`                                       | Exit the environment         |


6. Run CEM and LAC simulations using Julia. Note that running all 11 cases on a single thread will take 33+ hours, but running individual cases individually will take 3 hours, each.

| Step | Git Bash Command                       | Description                    |
|------|----------------------------------------|--------------------------------|
| 1  | `julia scripts/sge_run_cem_lac.jl`     | Run both CEM and LAC simulations |


7. Reproduce my figures

| Step | Git Bash Command                        | Description                    |
|------|-----------------------------------------|--------------------------------|
| 1  | `source ./venv/Scripts/activate`         | Activate the virtual environment |
| 2  | `python figures/run_all_figures.py`      | Generate all figure outputs    |




