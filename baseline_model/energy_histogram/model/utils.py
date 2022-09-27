import numpy as np
import pickle


def get_griddata(griddata):
    with open(griddata, 'rb') as f:
        return pickle.load(f)


def get_grid(grid):
    parameter = {}

    with open(grid) as f:
        for line in f:
            data = line.strip().split()
            parameter[data[0]] = np.array(data[1:], dtype='float')

    return parameter


def calculate_volume(a, b, c, angle_a, angle_b, angle_c):
    a_ = np.cos(angle_a * np.pi / 180)
    b_ = np.cos(angle_b * np.pi / 180)
    c_ = np.cos(angle_c * np.pi / 180)

    v = a * b * c * np.sqrt(1 - a_ ** 2 - b_ ** 2 - c_ ** 2 + 2 * a_ * b_ * c_)

    return v.item() / (60 * 60 * 60)  # normalized volume


def get_energy_histogram(grid_data):
    """ 
    Get energy grid histogram from grid_data
    """
    R = np.array([8.3145])
    energy = grid_data * R * 1.5 / 1000  # convert unit K -> KJ/mol

    energy[energy > 0] = 0.5  # Positive energy
    energy[energy < -10] = -10.5  # Energy under -10 KJ/mol

    i, _ = np.histogram(energy, bins=12, range=(-11, 1))  # Get energy histogram
    i = i / np.sum(i)  # Normalize

    return i


def get_indice(n, m):
    """
    n : total_list
    m : subset_list
    
    return indices of m in n
    """

    return np.vstack([np.where(n == m_dex) for m_dex in m]).squeeze()
