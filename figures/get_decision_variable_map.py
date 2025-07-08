import pandas as pd
import numpy as np


def get_decision_variable_map():
    """
    This function creates a mapping of decision variable names to their corresponding
    file names in the CEM and Policies models. It returns a DataFrame with the decision
    variable names as the index and two columns: 'CEM' and 'Policies', which contain
    the respective file names for each decision variable in the CEM and Policies models.
    """
    # Define the decision variable names and their corresponding file names
    decision_variable_names = [
                                'discharge', 
                                'charge',
                                'state of charge', 
                                'startup',
                                'shutdown',
                                'commitment',
                                'unmet reserves',
                                'non-served energy',
                                'regulation',
                                'reserve',
                                'energy prices',
                                'regulation prices',
                                'reserve prices',
                                'fuel costs',
                                'startup costs',
                                'charge costs',
                                'energy revenues',
                                'regulation revenues',
                                'reserve revenues',
                                'non-served energy costs',
                                'unmet reserves costs',
                            ]


    cem_decision_file_names = [
                                'power', 
                                'charge',
                                'storage', 
                                'start',
                                'shutdown',
                                'commit',
                                'reserves',
                                'nse',
                                'reg',
                                'reserves',
                                'prices',
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None,
                            ]


    policies_decision_file_names = [
                                'unit_pgen', 
                                'unit_charge',
                                'unit_state_of_charge', 
                                'unit_start',
                                'unit_shut',
                                'unit_commit',
                                'zone_unmet_rsv',
                                'zone_nse',
                                'unit_reg',
                                'unit_rsv',
                                'price_electricity',
                                'prices_reg',
                                'prices_rsv',
                                'revenue_fuel_costs_dp',
                                'revenue_start_costs_dp',
                                'revenue_charge_costs_dp',
                                'revenue_energy_revs_dp',
                                'revenue_reg_revs_dp',
                                'revenue_rsv_revs_dp',
                                'revenue_nse_cost',
                                'revenue_unmet_rsv_cost',
                            ]

    # Create a DataFrame with decision_variable_names as the index
    decision_variable_map = pd.DataFrame({
        'CEM': cem_decision_file_names,
        'Policies': policies_decision_file_names
    }, index=decision_variable_names)

    return decision_variable_map
