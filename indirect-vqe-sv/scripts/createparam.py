import numpy as np

def create_random_params(
        per_layer_angle_params: int,
        n_layers: int,
        time: float,
        seed: int = None
) -> np.ndarray:
    """
    Creates random parameters for the VQE ansatz.

    Args:
        per_layer_angle_params (int): The number of angle parameters per layer.
        n_layers (int): The number of layers in the ansatz.
        time (float): The total time for the evolution.
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        np.ndarray: An array of random parameters.
    """
    # Use the seed if provided for reproducibility
    if seed is not None:
        np.random.seed(seed)
    else:
        raise ValueError("Seed must be provided for reproducibility.")
    
    # Time parameters (one per layer, sorted to be monotonically increasing)
    time_params = np.random.uniform(0, time, n_layers)
    time_params = np.sort(time_params)

    # Generate random angle parameters
    total_angle_params = per_layer_angle_params * n_layers
    params = np.random.uniform(0, 1, total_angle_params) * 1e-1
    
    # Combine time and angle parameters
    combined_params = np.concatenate((time_params, params))

    # param_dict = {
    #     "time_params": time_params,
    #     "angle_params": params,
    #     "combined_params": combined_params
    # }

    return combined_params
    
# def create_param(layer: int, gateset: int, ti: float, tf: float) -> np.ndarray:
#     """
#     Creates initial parameters for the citcuit. Parameters are time, and theta: angle for rotation gates.

#     time: 0 - max time
#     theta: 0 - 1

#     Args:
#         layer (int): number of layer.
#         gateset (int): Number of rotation gate set. Each set has 4 gates.
#         ti (float): initial time
#         tf (float): final time

#     Returns:
#         numpy.ndarray: An nd-array [
#         t1, t2, ... td, theta1, ..., theatd * 4
#     ]

#     """
#     param = np.array([])
#     # Time param
#     time = np.random.uniform(ti, tf, layer)
#     time = np.sort(time)  # Time must be in incresing order
#     for i in time:
#         param = np.append(param, i)

#     # Theta param
#     theta = np.random.random(layer * gateset * 4) * 1e-1  # Each layer has 4 rotation gates
#     for i in theta:
#         param = np.append(param, i)

#     return param