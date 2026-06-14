
## SIMULATION

px_to_m = 1 / 30


baseline_args = {
    "initial_barge_position": (0.49, 0.00),  # 2-tuple (x, theta)
}

misaligned_50cm_args = {
    "initial_barge_position": (0.99, 0.00),  # 2-tuple (x, theta)
}

misaligned_100cm_args = {
    "initial_barge_position": (1.49, 0.00),
}

misaligned_150cm_args = {
    "initial_barge_position": (1.99, 0.00),
}

misaligned_500cm_args = {

}

# Less thruster range available
low_efficiency_args = {
    "main_thruster_range": 0.7, 
}

# Less side thruster range available
false_nozzle_sing_args = {
    "side_thruster_range": 0.7,  
}

mass_correction_args = {
    "mass_correction_factor": 1.3,  
}

less_fuel_args = {
    "mass_correction_factor": 0.72,  
}

args_collections = {}