
# Renewable Energy System of Schleswig Holstein

This project aims to analyze the renewable-based energy system model for the case study of Schleswig Holstein, the northernmost federal state of Germany.
An optimization model was developed
under the Open Energy Modeling Framework (Oemof). The study assumes three scenarios representing 25%, 50%, and 100% of the
total available biomass potentials. The enery model examines the sector-coupled network of electricity bus and heat bus, with expectation to increase the energy storage capacity and provide additional flexibility to the energy system.
Energy buses are Electricity bus and Heat bus.
PV, Wind onshore and offshore, Hydro RoR, Biomass are modelled as Sources.
Electricity grid, Electricity demand, Heat grid, Heat demand, Hot water demand are modelled as Sinks.
Heat Pump is modelled as Transformer.
Other components are Battery, Hydrogen, ACAES, CHP, Thermal storage.



## Lessons Learned

It was drawn that the less availability of biomass capacity was, the more investment in other technologies such as PV, Wind, Hydro, Battery and Storage were required. 
## Optimizations

The model optimized the energy system by minimizing the total system costs with the flexibility of different technologies. For further improvement, other renewable-based energy technologies at the place can be considered such as tidal and geothermal energy.

