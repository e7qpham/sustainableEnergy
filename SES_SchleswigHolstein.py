# Load libaries
from oemof import solph
from oemof.tools import logger
from oemof.tools import economics

import logging
import pandas as pd
import matplotlib.pyplot as plt

import pprint as pp
from collections import OrderedDict

#%% Initialization
# Logger: initiate it (see the API docs for more information)
logger.define_logging(logfile='model.log', 
                      screen_level=logging.INFO,
                      file_level=logging.DEBUG)
logging.info('Necessary packages have been imported.')

#%% Read input data
# Input Data Reading
logging.info('Read input data.')
timeseries = pd.read_excel('../data/input_data_exercise2.xls', 
                           sheet_name='timeseries', 
                           index_col=[0], 
                           parse_dates=True)

# Add timestep (oemof model needs time increment)
timeseries.index.freq = '1H'

capacities = pd.read_excel('../data/input_data_exercise2.xls', 
                           sheet_name='capacity', 
                           index_col=[0], 
                           parse_dates=True)
tech = pd.read_excel('../data/input_data_exercise2.xls', 
                           sheet_name='tech', 
                           index_col=[0], 
                           parse_dates=True)
costs = pd.read_excel('../data/input_data_exercise2.xls', 
                           sheet_name='costs', 
                           index_col=[0], 
                           parse_dates=True)

#%% Initialize the energy system and read/calculate necessary parameters
logger.define_logging()
logging.info('Initialize the energy system')

energysystem = solph.EnergySystem(timeindex=timeseries.index)

#%% Create oemof Buses

logging.info('Create oemof objects')

# create electricity bus
bus_electricity = solph.Bus(label='bus_electricity_l')
# create heat bus
bus_heat = solph.Bus(label='bus_heat_l')
# create biomass bus
bus_biomass = solph.Bus(label='bus_biomass_l')

# add buses to energy model
energysystem.add(bus_electricity, bus_heat, bus_biomass)

#%% Create oemof Sinks

# create excess component for the electricity bus to allow overproduction
electricity_excess = solph.Sink(label='electricty_excess_l', 
                                inputs={bus_electricity: solph.Flow()})

# create simple sink object representing the electrical demand
electricity_demand = solph.Sink(label='electricity_demand_l',
                                inputs={bus_electricity: solph.Flow(
                                        fix=timeseries['electricity'], 
                                        nominal_value=capacities['electricity']['amount'])},)

# create excess component for the heat bus to allow overproduction
heat_excess = solph.Sink(label='heat_excess_l', 
                         inputs={bus_heat: solph.Flow()})

# create simple sink object representing the heat demand (space heat and hot water demand)
heat_space_demand = solph.Sink(label='heat_space_demand_l',
                               inputs={bus_heat: solph.Flow(
                                       fix=timeseries['space_heat'], 
                                       nominal_value=capacities['space_heat']['amount'])},)

heat_dhw_demand = solph.Sink(label='heat_dhw_demand_l',
                             inputs={bus_heat: solph.Flow(
                                 fix=timeseries['dhw_heat'], 
                                 nominal_value=capacities['dhw_heat']['amount'])},)

#%% Economic caluclation (for dispatch/sizing optimization) 

## Capital costs
# Annuities
a_onshore = economics.annuity(capex=costs['onshore']['capex'], 
                              n=costs['onshore']['lifetime'],
                              wacc=costs['onshore']['wacc'])
a_offshore = economics.annuity(capex=costs['offshore']['capex'], 
                               n=costs['offshore']['lifetime'],
                               wacc=costs['offshore']['wacc'])
a_pv = economics.annuity(capex=costs['pv']['capex'], 
                         n=costs['pv']['lifetime'],
                         wacc=costs['pv']['wacc'])
a_ror = economics.annuity(capex=costs['ror']['capex'], 
                          n=costs['ror']['lifetime'],
                          wacc=costs['ror']['wacc'])

a_chp = economics.annuity(capex=costs['biomass']['capex'], 
                            n=costs['biomass']['lifetime'],
                            wacc=costs['biomass']['wacc'])
a_hp = economics.annuity(capex=costs['hp']['capex'], 
                           n=costs['hp']['lifetime'],
                           wacc=costs['hp']['wacc'])

a_battery_energy = economics.annuity(capex=costs['battery']['capex_energy'], 
                                     n=costs['battery']['lifetime'],
                                     wacc=costs['battery']['wacc'])
a_battery = economics.annuity(capex=costs['battery']['capex'], 
                              n=costs['battery']['lifetime'],
                              wacc=costs['battery']['wacc'])
a_hydrogen_energy = economics.annuity(capex=costs['hydrogen']['capex_energy'], 
                                      n=costs['hydrogen']['lifetime'],
                                      wacc=costs['hydrogen']['wacc'])
a_hydrogen = economics.annuity(capex=costs['hydrogen']['capex'], 
                               n=costs['hydrogen']['lifetime'],
                               wacc=costs['hydrogen']['wacc'])
a_acaes_energy = economics.annuity(capex=costs['acaes']['capex_energy'], 
                                     n=costs['acaes']['lifetime'],
                                     wacc=costs['acaes']['wacc'])
a_acaes = economics.annuity(capex=costs['acaes']['capex'], 
                              n=costs['acaes']['lifetime'],
                              wacc=costs['acaes']['wacc'])
a_tes_energy = economics.annuity(capex=costs['tes']['capex_energy'], 
                                   n=costs['tes']['lifetime'],
                                       wacc=costs['tes']['wacc'])


# Capital costs
cc_onshore = (a_onshore + costs['onshore']['fom'])
cc_offshore = (a_offshore + costs['offshore']['fom'])
cc_pv = (a_pv + costs['pv']['fom'])
cc_ror = (a_ror + costs['ror']['fom'])

cc_chp = (a_chp + costs['biomass']['fom'])
cc_hp = (a_hp + costs['hp']['fom'])

cc_battery_energy = (a_battery_energy + costs['battery']['fom'])
cc_battery = (a_battery)
cc_hydrogen_energy = (a_hydrogen_energy + costs['hydrogen']['fom'])
cc_hydrogen = (a_hydrogen)
cc_acaes_energy = (a_acaes_energy + costs['acaes']['fom'])
cc_acaes = (a_acaes)
cc_tes_energy = (a_tes_energy + costs['tes']['fom'])


## Marginal costs
mc_onshore = costs['onshore']['vom']
mc_offshore = costs['offshore']['vom']
mc_pv = costs['pv']['vom']
mc_ror = costs['ror']['vom']

mc_chp = costs['biomass']['vom']
mc_hp = costs['hp']['vom']

mc_battery = costs['battery']['vom']
mc_hydrogen = costs['hydrogen']['vom']
mc_acaes = costs['acaes']['vom']
mc_tes = costs['tes']['vom']

#%%Create oemof Sources

# create fixed source object representing wind power plants offshore
wind_offshore = solph.Source(label='wind_offshore_l',
                             outputs={bus_electricity: solph.Flow(   
                                     fix=timeseries['offshore'], 
                                     variable_costs=mc_offshore,
                                     investment=solph.Investment(
                                                ep_costs=cc_offshore,
                                                maximum=capacities['offshore']['capacity_potential'],
                                                existing=capacities['offshore']['capacity_existing']))
                                    },)

# create fixed source object representing wind power plants onshore
wind_onshore = solph.Source(label='wind_onshore_l',
                            outputs={bus_electricity: solph.Flow(
                                    fix=timeseries['onshore'], 
                                    variable_costs=mc_onshore,
                                    investment=solph.Investment(
                                               ep_costs=cc_onshore,
                                               maximum=capacities['onshore']['capacity_potential'],
                                               existing=capacities['onshore']['capacity_existing']))
                                    },)

# create fixed source object representing pv power plants
pv = solph.Source(label='pv_l',
                  outputs={bus_electricity: solph.Flow(
                           fix=timeseries['pv'], 
                           variable_costs=mc_pv,
                           investment=solph.Investment(
                                      ep_costs=cc_pv,
                                      maximum=capacities['pv']['capacity_potential'],
                                      existing=capacities['pv']['capacity_existing']))
                           },)

# create fixed source object representing hydro run of river plant
ror = solph.Source(label='ror_l',
                   outputs={bus_electricity: solph.Flow(
                            fix=timeseries['ror'], 
                            variable_costs=mc_ror,
                            investment=solph.Investment(
                                       ep_costs=cc_ror,
                                       maximum=capacities['ror']['capacity_potential'],
                                       existing=capacities['ror']['capacity_existing']))
                            },)

# create fixed source object representing biomass ressource
biomass = solph.Source(label='biomass_l',
                       outputs={bus_biomass: solph.Flow(
                               nominal_value=capacities['biomass']['capacity_potential'],
                               summed_max=1)
                       },)
                                   	
#%% Create oemof Storages

# create storage object representing a battery
battery = solph.components.GenericStorage(label='battery_l',
                                          inputs={bus_electricity: solph.Flow(
                                                  investment=solph.Investment(
                                                             ep_costs=cc_battery,
                                                             maximum=capacities['battery']['storage_power_potential']),
                                                  variable_costs=mc_battery)},
                                          outputs={bus_electricity: solph.Flow()},
                                          loss_rate=tech['battery']['loss'],
                                          initial_storage_level=0,
                                          invest_relation_input_capacity=1/tech['battery']['max_hours'],
                                          invest_relation_output_capacity=1/tech['battery']['max_hours'],
                                          inflow_conversion_factor=1,
                                          outflow_conversion_factor=tech['battery']['efficiency'],
                                          investment=solph.Investment(
                                                     ep_costs=cc_battery_energy,
                                                     maximum=capacities['battery']['capacity_potential']),)

# create storage object representing a hydrogen
hydrogen = solph.components.GenericStorage(label='hydrogen_l',
                                          inputs={bus_electricity: solph.Flow(
                                                  investment=solph.Investment(
                                                             ep_costs=cc_hydrogen,
                                                             maximum=capacities['hydrogen']['storage_power_potential']),
                                                  variable_costs=mc_hydrogen)},
                                          outputs={bus_electricity: solph.Flow()},
                                          loss_rate=tech['hydrogen']['loss'],
                                          initial_storage_level=0,
                                          invest_relation_input_capacity=1/tech['hydrogen']['max_hours'],
                                          invest_relation_output_capacity=1/tech['hydrogen']['max_hours'],
                                          inflow_conversion_factor=1,
                                          outflow_conversion_factor=tech['hydrogen']['efficiency'],
                                          investment=solph.Investment(
                                                     ep_costs=cc_hydrogen_energy,
                                                     maximum=capacities['hydrogen']['capacity_potential']),)

# create storage object representing a adiabatic compressed air energy storage (ACAES)
acaes = solph.components.GenericStorage(label='acaes_l',
                                          inputs={bus_electricity: solph.Flow(
                                                  investment=solph.Investment(
                                                             ep_costs=cc_acaes,
                                                             maximum=capacities['acaes']['storage_power_potential']),
                                                  variable_costs=mc_acaes)},
                                          outputs={bus_electricity: solph.Flow()},
                                          loss_rate=tech['acaes']['loss'],
                                          initial_storage_level=0,
                                          invest_relation_input_capacity=1/tech['acaes']['max_hours'],
                                          invest_relation_output_capacity=1/tech['acaes']['max_hours'],
                                          inflow_conversion_factor=1,
                                          outflow_conversion_factor=tech['acaes']['efficiency'],
                                          investment=solph.Investment(
                                                     ep_costs=cc_acaes_energy,
                                                     maximum=capacities['acaes']['capacity_potential']),)

# create storage object representing a battery
tes = solph.components.GenericStorage(label='tes_l',
                                          inputs={bus_heat: solph.Flow(
                                                  investment=solph.Investment(
                                                             maximum=capacities['tes']['storage_power_potential']),
                                                  variable_costs=mc_tes)},
                                          outputs={bus_heat: solph.Flow()},
                                          loss_rate=tech['tes']['loss'],
                                          initial_storage_level=0,
                                          invest_relation_input_capacity=1/tech['tes']['max_hours'],
                                          invest_relation_output_capacity=1/tech['tes']['max_hours'],
                                          inflow_conversion_factor=1,
                                          outflow_conversion_factor=tech['tes']['efficiency'],
                                          investment=solph.Investment(
                                                     ep_costs=cc_tes_energy,
                                                     maximum=capacities['tes']['capacity_potential']),)

#%% Create oemof Transormers

# create transformer object representing heat pumps
hp = solph.Transformer(label='hp_l',
                              inputs={bus_electricity: solph.Flow()},
                              outputs={bus_heat: solph.Flow(
                                       investment=solph.Investment(
                                                  ep_costs=cc_hp),
                                       variable_costs=mc_hp)},
                              conversion_factors={bus_electricity: 1/tech['hp']['efficiency']},
                              )

# create transformer object representing CHP plants
chp = solph.Transformer(label='chp_l',
                        inputs={bus_biomass: solph.Flow(
                                    variable_costs=mc_chp)},
                        outputs={bus_electricity: solph.Flow(
                                    investment=solph.Investment(
                                               ep_costs=cc_chp,
                                               existing=capacities['biomass']['capacity_existing'])),
                                 bus_heat: solph.Flow()},
                        conversion_factors={bus_electricity: tech['biomass']['electric_efficiency'],
                                            bus_heat: tech['biomass']['thermal_efficiency']},
                        )

#%% Add all components to the energysystem
energysystem.add(electricity_excess, electricity_demand,
                 heat_excess, heat_space_demand, heat_dhw_demand,
                 wind_offshore, wind_onshore, pv, ror, biomass,
                 battery, hydrogen, acaes, tes, 
                 hp, chp)

#%% Optimise the energy system

logging.info('Optimise the energy system')

# initialise the operational model
om = solph.Model(energysystem)

# if tee_switch is true solver messages will be displayed
logging.info('Solve the optimization problem')
om.solve(solver='cbc', solve_kwargs={'tee': False})

#Extract main results save results to dump (optional)
energysystem.results['main'] = solph.processing.results(om)
energysystem.dump('../results/dumps',
                  filename='model.oemof')

#%% Restore EnergySystem results
# energysystem = solph.EnergySystem()

# # Restore optimization results
# energysystem.restore('../results/dumps', filename='model.oemof')

#%% Extract results 
results = energysystem.results['main']
# Extract results dict
#results = solph.processing.results(om)

# Extract component results
results_wind_offshore = solph.views.node(results, 'wind_offshore_l')
results_wind_onshore = solph.views.node(results, 'wind_onshore_l')
results_pv = solph.views.node(results, 'pv_l')
results_ror = solph.views.node(results, 'ror_l')

results_biomass = solph.views.node(results, 'bus_biomass_l')
results_chp = solph.views.node(results, 'chp_l')
results_hp = solph.views.node(results, 'hp_l')

results_battery = solph.views.node(results, 'battery_l')
results_hydrogen = solph.views.node(results, 'hydrogen_l')
results_acaes = solph.views.node(results, 'acaes_l')
results_tes = solph.views.node(results, 'tes_l')

# Extract bus results
results_electricity_bus = solph.views.node(results, 'bus_electricity_l')
results_heat_bus = solph.views.node(results, 'bus_heat_l')
results_biomass_bus = solph.views.node(results, 'bus_biomass_l')


#%% Installed capacities
# Define capacity results dict
results_capacity = OrderedDict()

# installed capacity of wind power, pv and ror plant in MW
results_capacity['wind_onshore_invest_MW'] = results[(wind_onshore, bus_electricity)]['scalars']['invest']
results_capacity['wind_offshore_invest_MW'] = results[(wind_offshore, bus_electricity)]['scalars']['invest']
results_capacity['pv_invest_MW'] = results[(pv, bus_electricity)]['scalars']['invest']
results_capacity['ror_invest_MW'] = results[(ror, bus_electricity)]['scalars']['invest']

# installed capacity of chp and hp plant in MW
results_capacity['chp_invest_MW_el'] = results[(chp, bus_electricity)]['scalars']['invest']
results_capacity['hp_invest_MW_th'] = results[(hp, bus_heat)]['scalars']['invest']

# installed capacity of battery, hydrogen, acaes and tes storage in MWh and power capacity in MW
results_capacity['battery_invest_MWh'] = results[(battery, None)]['scalars']['invest']
results_capacity['battery_invest_MW_ch'] = results[(bus_electricity, battery)]['scalars']['invest']
results_capacity['battery_invest_MW_dch'] = results[(battery, bus_electricity)]['scalars']['invest']

results_capacity['hydrogen_invest_MWh'] = results[(hydrogen, None)]['scalars']['invest']
results_capacity['hydrogen_invest_MW_ch'] = results[(bus_electricity, hydrogen)]['scalars']['invest']
results_capacity['hydrogen_invest_MW_dch'] = results[(hydrogen, bus_electricity,)]['scalars']['invest']

results_capacity['acaes_invest_MWh'] = results[(acaes, None)]['scalars']['invest']
results_capacity['acaes_invest_MW_ch'] = results[(bus_electricity, acaes)]['scalars']['invest']
results_capacity['acaes_invest_MW_dch'] = results[(acaes, bus_electricity)]['scalars']['invest']

results_capacity['thermal_storage_invest_MWh'] = results[(tes, None)]['scalars']['invest']
results_capacity['thermal_storage_invest_MW_ch'] = results[(bus_heat, tes)]['scalars']['invest']
results_capacity['thermal_storage_invest_MW_dch'] = results[(tes, bus_heat)]['scalars']['invest']

pp.pprint(results_capacity)

# Transfer dict to DataFRame and transpose for better readybility
results_capacity_df = pd.DataFrame(results_capacity, index=[0]).T

#%% Investment costs

## Investment costs
results_inv_costs = OrderedDict()
# Wind on/offshore, PV, RoR, CHP, HP
results_inv_costs['wind_onshore_mio'] = round(a_onshore * results_capacity['wind_onshore_invest_MW'] /1e6, 2)
results_inv_costs['wind_offshore_mio'] = round(a_offshore * results_capacity['wind_offshore_invest_MW'] /1e6, 2)
results_inv_costs['pv_mio'] = round(a_pv * results_capacity['pv_invest_MW'] /1e6, 2)
results_inv_costs['ror_mio'] = round(a_ror * results_capacity['ror_invest_MW'] /1e6, 2)
results_inv_costs['chp_mio'] = round(a_chp * results_capacity['chp_invest_MW_el'] /1e6, 2)
results_inv_costs['hp_mio'] = round(a_hp * results_capacity['hp_invest_MW_th'] /1e6, 2)

# Storages: battery, hydrogen, acaes, tes with storage energy costs and storage power costs
results_inv_costs['battery_mio'] = round(a_battery_energy * results_capacity['battery_invest_MWh'] /1e6, 2)
results_inv_costs['battery_power_ch_mio'] = round(a_battery * results_capacity['battery_invest_MW_ch'] /1e6, 2)
results_inv_costs['battery_power_dch_mio'] = round(a_battery * results_capacity['battery_invest_MW_dch'] /1e6, 2)
# hydrogen
results_inv_costs['hydrogen_mio'] = round(a_hydrogen_energy * results_capacity['hydrogen_invest_MWh'] /1e6, 2)
results_inv_costs['hydrogen_power_ch_mio'] = round(a_hydrogen * results_capacity['hydrogen_invest_MW_ch'] /1e6, 2)
results_inv_costs['hydrogen_power_dch_mio'] = round(a_hydrogen * results_capacity['hydrogen_invest_MW_dch'] /1e6, 2)
# acaes
results_inv_costs['acaes_mio'] = round(a_acaes_energy * results_capacity['acaes_invest_MWh'] /1e6, 2)
results_inv_costs['acaes_power_ch_mio'] = round(a_acaes * results_capacity['acaes_invest_MW_ch'] /1e6, 2)
results_inv_costs['acaes_power_dch_mio'] = round(a_acaes * results_capacity['acaes_invest_MW_dch'] /1e6, 2)
# tes
results_inv_costs['tes_mio'] = round(a_tes_energy * results_capacity['thermal_storage_invest_MWh'] /1e6, 2)

# Total
results_inv_costs['total'] = sum(results_inv_costs.values())

pp.pprint(results_inv_costs)

# Transfer dict to DataFRame and transpose for better readybility
results_inv_costs_df = pd.DataFrame(results_inv_costs, index=[0]).T


## Variable costs
#wind_offshore_variable = [b for a, b in om.flows.items() if a[0] == wind_offshore][0].variable_costs
#battery_variable = [b for a, b in om.flows.items() if a[0] == battery][0].variable_costs
#chp_variable = [b for a, b in om.flows.items() if a[0] == chp][0].variable_costs

#%% Biomass, Heat and electricty generation mix in TWh
results_energy_biomass = OrderedDict()
results_energy_biomass['total_TWh'] = results_chp['sequences'][(('bus_biomass_l', 'chp_l'), 'flow')].sum() / 1e6
pp.pprint(results_energy_biomass)

results_energy_heat = OrderedDict()
results_energy_heat['chp_TWh'] = results_chp['sequences'][('chp_l','bus_heat_l'),'flow'].sum() / 1e6
results_energy_heat['hp_TWh'] = results_hp['sequences'][('hp_l','bus_heat_l'),'flow'].sum() / 1e6
results_energy_heat['total_TWh'] = sum(results_energy_heat.values())
pp.pprint(results_energy_heat)

results_energy_electricity = OrderedDict()
results_energy_electricity['wind_onshore_TWh'] = results_wind_onshore['sequences'][('wind_onshore_l','bus_electricity_l'),'flow'].sum() / 1e6
results_energy_electricity['wind_offshore_TWh'] = results_wind_offshore['sequences'][('wind_offshore_l','bus_electricity_l'),'flow'].sum() / 1e6
results_energy_electricity['pv_TWh'] = results_pv['sequences'][('pv_l','bus_electricity_l'),'flow'].sum() / 1e6
results_energy_electricity['ror_TWh'] = results_ror['sequences'][('ror_l','bus_electricity_l'),'flow'].sum() / 1e6
results_energy_electricity['chp_TWh'] = results_chp['sequences'][('chp_l','bus_electricity_l'),'flow'].sum() / 1e6
results_energy_electricity['total_TWh'] = sum(results_energy_electricity.values())

pp.pprint(results_energy_electricity)

# Transfer dict to DataFrame and transpose for better readybility
results_energy_biomass_df = pd.DataFrame(results_energy_biomass, index=[0]).T
results_energy_heat_df = pd.DataFrame(results_energy_heat, index=[0]).T
results_energy_electricity_df = pd.DataFrame(results_energy_electricity, index=[0]).T


#%%  Calculation of storage hourly and mean annual SoCs

# Define result dicts
results_storages_soc_h = OrderedDict()
results_storages_soc_a = OrderedDict()

# Battery
results_storages_soc_h['battery'] = \
    (solph.views.node(results, 'battery_l')['sequences'][(('battery_l', 'None'), 'storage_content')]) \
    / (solph.views.node(results, 'battery_l')['scalars'][(('battery_l', 'None'), 'invest')])    
results_storages_soc_a['battery'] = results_storages_soc_h['battery'].mean()                 
     
# Hydrogen
results_storages_soc_h['hydrogen'] = \
    (solph.views.node(results, 'hydrogen_l')['sequences'][(('hydrogen_l', 'None'), 'storage_content')]) \
    / (solph.views.node(results, 'hydrogen_l')['scalars'][(('hydrogen_l', 'None'), 'invest')])
results_storages_soc_a['hydrogen'] = results_storages_soc_h['hydrogen'].mean()
              
# ACAES        
results_storages_soc_h['acaes'] = \
    (solph.views.node(results, 'acaes_l')['sequences'][(('acaes_l', 'None'), 'storage_content')]) \
    / (solph.views.node(results, 'acaes_l')['scalars'][(('acaes_l', 'None'), 'invest')])
results_storages_soc_a['acaes'] = results_storages_soc_h['acaes'].mean()

# TES
results_storages_soc_h['tes'] = \
    (solph.views.node(results, 'tes_l')['sequences'][(('tes_l', 'None'), 'storage_content')]) \
    / (solph.views.node(results, 'tes_l')['scalars'][(('tes_l', 'None'), 'invest')])
results_storages_soc_a['tes'] = results_storages_soc_h['tes'].mean()


pp.pprint(results_storages_soc_h)
pp.pprint(results_storages_soc_a)

# Transfer dict to DataFrame and transpose for better readybility
results_storages_soc_h_df = pd.DataFrame(results_storages_soc_h, index=[0]).T
results_storages_soc_a_df = pd.DataFrame(results_storages_soc_a, index=[0]).T

#%% PLotting hourly SoC
# fig = plt.figure(figsize=(15,5))
# plt.plot(results_storages_soc_h['acaes'], alpha=0.75, label='ACAES')
# plt.plot(results_storages_soc_h['battery'], alpha=0.75, label='Battery')
# plt.plot(results_storages_soc_h['hydrogen'], alpha=0.75, label='H2')
# plt.ylabel('SoC')
# plt.xlabel('Time')
# plt.legend()
# plt.show()

#%% Collection of all results and exporting to ecxel file

# Create a Pandas Excel writer using XlsxWriter as the engine.
with pd.ExcelWriter('../results/results_overview.xlsx', engine='xlsxwriter') as writer:  
    
    # Write each dataframe to a different worksheet.
    results_capacity_df.to_excel(writer, sheet_name='capacities')
    results_energy_biomass_df.to_excel(writer, sheet_name='energy_biomass')
    results_energy_heat_df.to_excel(writer, sheet_name='energy_heat')
    results_energy_electricity_df.to_excel(writer, sheet_name='energy_elec')
    results_inv_costs_df.to_excel(writer, sheet_name='inv_costs')
    results_storages_soc_a_df.to_excel(writer, sheet_name='soc_a')
    
#%% Results check --> Maybe assignment

## MUSS NOCH UMGESCHRIEBEN WREDEN FÜR LABELS!!

## Wind onshore check
capacity_factor_wind_onshore = results_wind_onshore['sequences'] \
                     / (results[(wind_onshore, bus_electricity)]['scalars']['invest']+capacities['onshore']['capacity_existing'])
print(capacity_factor_wind_onshore.resample('Y').mean(), '=', timeseries['onshore'].resample('Y').mean())

## Wind offshore check
capacity_factor_wind_offshore = results_wind_offshore['sequences'] \
                     / (results[(wind_offshore, bus_electricity)]['scalars']['invest']+capacities['offshore']['capacity_existing'])
print(capacity_factor_wind_offshore.resample('Y').mean(), '=', timeseries['offshore'].resample('Y').mean())

## PV check
capacity_factor_pv = results_pv['sequences'] \
                     / (results[(pv, bus_electricity)]['scalars']['invest']+capacities['pv']['capacity_existing'])
print(capacity_factor_pv.resample('Y').mean(), '=', timeseries['pv'].resample('Y').mean())

## ROR check
capacity_factor_ror = results_ror['sequences'] \
                     / (results[(ror, bus_electricity)]['scalars']['invest']+capacities['ror']['capacity_existing'])
print(capacity_factor_ror.resample('Y').mean(), '=', timeseries['ror'].resample('Y').mean())

## CHP check
biomass_power_max = results_chp['sequences'][('bus_biomass_l','chp_l'),'flow'].max()
print(biomass_power_max, '=', results_capacity['chp_invest_MW_el'])

chp_biomass_sum = results_chp['sequences'][('bus_biomass_l','chp_l'),'flow'].sum()
chp_electricity_sum = results_chp['sequences'][('chp_l','bus_electricity_l'),'flow'].sum()
chp_heat_sum = results_chp['sequences'][('chp_l','bus_heat_l'),'flow'].sum()

print(chp_biomass_sum*tech['biomass']['electric_efficiency'], '=', \
      chp_electricity_sum)
print(chp_biomass_sum*tech['biomass']['thermal_efficiency'], '=', \
      chp_heat_sum)    


## HP
hp_power_th_max = results_hp['sequences'][('hp_l', 'bus_heat_l'),'flow'].max()
print(hp_power_th_max, '=', results_capacity['hp_invest_MW_th'])

hp_electricity_sum = results_hp['sequences'][('bus_electricity_l', 'hp_l'),'flow'].sum()
hp_heat_sum = results_hp['sequences'][('hp_l','bus_heat_l'),'flow'].sum()
print(hp_electricity_sum*tech['hp']['efficiency'] , '=', hp_heat_sum )

## Battery
print('Battery')
bat_storage_content_max = results_battery['sequences'][('battery_l','None'),'storage_content'].max()
print(bat_storage_content_max, '=', results_capacity['battery_invest_MWh'])

bat_ch_sum = results_battery['sequences'][('bus_electricity_l','battery_l'),'flow'].sum()
bat_dch_sum = results_battery['sequences'][('battery_l', 'bus_electricity_l'),'flow'].sum()
# Only valid in case initial and final storage content is equal
print(bat_ch_sum, '=', bat_dch_sum/tech['battery']['efficiency']) # Only valid in case initial and final storage content is equal


## H2
hydrogen_storage_content_max = results_hydrogen['sequences'][('hydrogen_l','None'),'storage_content'].max()
print(hydrogen_storage_content_max, '=', results_capacity['hydrogen_invest_MWh'])

hydrogen_ch_sum = results_hydrogen['sequences'][('bus_electricity_l','hydrogen_l'),'flow'].sum()
hydrogen_dch_sum = results_hydrogen['sequences'][('hydrogen_l', 'bus_electricity_l'),'flow'].sum()
# Only valid in case initial and final storage content is equal
print(hydrogen_ch_sum, '=', hydrogen_dch_sum/tech['hydrogen']['efficiency']) # Only valid in case initial and final storage content is equal


## ACAES
acaes_storage_content_max = results_acaes['sequences'][('acaes_l','None'),'storage_content'].max()
print(acaes_storage_content_max, '=', results_capacity['acaes_invest_MWh'])

acaes_ch_sum = results_acaes['sequences'][('bus_electricity_l','acaes_l'),'flow'].sum()
acaes_dch_sum = results_acaes['sequences'][('acaes_l', 'bus_electricity_l'),'flow'].sum()
# Only valid in case initial and final storage content is equal
print(acaes_ch_sum, '=', acaes_dch_sum/tech['acaes']['efficiency']) # Only valid in case initial and final storage content is equal


## TES
tes_storage_content_max = results_tes['sequences'][('tes_l','None'),'storage_content'].max()
print(tes_storage_content_max, '=', results_capacity['thermal_storage_invest_MWh'])

tes_ch_sum = results_tes['sequences'][('bus_heat_l','tes_l'),'flow'].sum()
tes_dch_sum = results_tes['sequences'][('tes_l', 'bus_heat_l'),'flow'].sum()
# Only valid in case initial and final storage content is equal AND no storage loss specified!
print(tes_ch_sum, '=', tes_dch_sum/tech['tes']['efficiency']) # Only valid in case initial and final storage content is equal


#%% Storage capacities and power check

print('Battery')
bat_storage_content_max = results_battery['sequences'][('battery_l','None'),'storage_content'].max()
bat_ch_max = results_battery['sequences'][('bus_electricity_l','battery_l'),'flow'].max()
bat_dch_max = results_battery['sequences'][('battery_l', 'bus_electricity_l'),'flow'].max()
print('bat_storage_content_max' , bat_storage_content_max)
print('bat_ch_max ', bat_ch_max)
print('bat_dch_max ', bat_dch_max)

print('acaes')
acaes_storage_content_max = results_acaes['sequences'][('acaes_l','None'),'storage_content'].max()
acaes_ch_max = results_acaes['sequences'][('bus_electricity_l','acaes_l'),'flow'].max()
acaes_dch_max = results_acaes['sequences'][('acaes_l', 'bus_electricity_l'),'flow'].max()
print('acaes_storage_content_max' , acaes_storage_content_max)
print('acaes_ch_max ', acaes_ch_max)
print('acaes_dch_max ', acaes_dch_max)

print('Hydrogen')
hydrogen_storage_content_max = results_hydrogen['sequences'][('hydrogen_l','None'),'storage_content'].max()
hydrogen_ch_max = results_hydrogen['sequences'][('bus_electricity_l','hydrogen_l'),'flow'].max()
hydrogen_dch_max = results_hydrogen['sequences'][('hydrogen_l', 'bus_electricity_l'),'flow'].max()
print('hydrogen_storage_content_max' , hydrogen_storage_content_max)
print('hydrogen_ch_max ', hydrogen_ch_max)
print('hydrogen_dch_max ', hydrogen_dch_max)

print('thermal_storage')
thermal_storage_storage_content_max = results_tes['sequences'][('tes_l','None'),'storage_content'].max()
thermal_storage_ch_max = results_tes['sequences'][('bus_heat_l','tes_l'),'flow'].max()
thermal_storage_dch_max = results_tes['sequences'][('tes_l', 'bus_heat_l'),'flow'].max()
print('thermal_storage_storage_content_max' , thermal_storage_storage_content_max)
print('thermal_storage_ch_max ', thermal_storage_ch_max)
print('thermal_storage_dch_max ', thermal_storage_dch_max)

