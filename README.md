# Elevator-Sim

### See ```Elevator_Report.pdf``` for extensive details on functionality, simulation results, and the broader implications for elevator scheduling.

ElevatorSim.py is a discrete-event simulation of elevator behavior in an office building over the course of one workday. It is an agent-based simulation in which each agent is an occupant of the building. Agents' elevator utilization is realistic and stochastic, modelled off of data collected by Kuusinen et al. (2011). Simulated elevators may use one of two commercially standard *scheduling algorithms*, which are algorithms that determine the elevator that services a passenger's request when they push a button in the hallway.

To conduct a simulation, instantiate the ```Building``` class:
### ```Building(number_of_floors, number_of_elevators)```
#### ```number_of_floors:``` the number of floors in the building, including the lobby.
#### ```number_of_elevators:``` the number of elevators in the building, each of which serves all floors.
---
Call either ```Building.run_conventional_alg(mult)``` to perform a simulation in which elevators are scheduled using the conventional algorithm, or ```Building.run_destination_dispatch(mult)``` for elevators to be scheduled using destination dispatch. ```mult``` determines the intensity of passenger traffic. By default, ```mult=1```.
