# -*- coding: utf-8 -*-
"""

@author: Barisoo

This file allows for easy extraction of the following features cif files, 
particularly for metal-organic frameworks:
    
    number of hydrogen atoms per unit cell
    number of carbon atoms per unit cell
    number of nitrogen atoms per unit cell
    number of oxygen atoms per unit cell
    number of fluorine atoms per unit cell
    number of chlorine atoms per unit cell
    number of bromine atoms per unit cell
    number of vanadium atoms per unit cell
    number of copper atoms per unit cell
    number of zinc atoms per unit cell
    number of zirconium atoms per unit cell
    metal type
    total degree of unsaturation
    degree of unsaturation per carbon
    metallic percentage
    oxygen to metal ratio
    electronegative atoms to total atoms ratio
    weighted electronegativity per atom
    nitrogen to oxygen ratio
    volume of MOF's unit cell

"""

import os
import numpy as np
from os import listdir
import argparse


def get_atom_dict(file_address):
    """
    Takes the full adress of a cif file as a string
    Returns atom counts of elements that make up a MOF
    
    This will be used to get the counts of specific atoms, getting the metal-count etc.
    """

    atoms = {}
    file = open(file_address, "r")
    content = file.read()
    file.close()
    """loop_
      _atom_site_label
      _atom_site_occupancy
      _atom_site_fract_x
      _atom_site_fract_y
      _atom_site_fract_z
      _atom_site_thermal_displace_type
      _atom_site_B_iso_or_equiv
      _atom_site_type_symbol"""
    atom_type_column_index = None
    column_count = 0  # sets the atom_type_column_index; it continues to count even after setting the column index -this can be disabled to potentially gain a slight run-time improvement
    lines = content.split("\n")
    for line in lines:
        if line == "loop_":
            column_count = -1

        if line != "":
            words = line.split()
            if words[0] == "_atom_site_type_symbol":
                atom_type_column_index = column_count
            if len(words) >= 5 and words[0][
                0] != "_" and atom_type_column_index != None:  # assumes there are 5 'words' only in the lines that contain atom type information besides the "_audit_creation_method" line
                atom = words[atom_type_column_index]
                if atom not in atoms:
                    atoms[atom] = 0
                atoms[atom] += 1
            column_count += 1
    return atoms


def sin(angle):
    """
    Returns the cos of an angle in degrees
    
    Utilises the sin and deg2rad functions of Numpy
    """
    return np.sin(np.deg2rad(angle))


def get_volume(file_address):
    """
    Takes the full adress of a cif file as a string
    Returns the volume of the MOF as a float
    
    This function can be used if a per unit volume basis for other features is desired 
    and by dividing previously obtained values by the volume calculated here
    """
    dimensions = {"_cell_length_a": None, "_cell_length_b": None, "_cell_length_c": None, "_cell_angle_alpha": None,
                  "_cell_angle_beta": None, "_cell_angle_gamma": None}
    file = open(file_address, "r")
    content = file.read()
    file.close()

    lines = content.split("\n")
    for line in lines:
        if line != "":
            words = line.split()
            if words[0] in dimensions:
                dimensions[words[0]] = float(words[1])
        if all([dimensions[i] is not None for i in dimensions]):
            break

    sina = sin(dimensions["_cell_angle_alpha"])
    sinb = sin(dimensions["_cell_angle_beta"])
    sinc = sin(dimensions["_cell_angle_gamma"])
    lengtha = dimensions["_cell_length_a"]
    lengthb = dimensions["_cell_length_b"]
    lengthc = dimensions["_cell_length_c"]
    volume = lengtha * lengthb * lengthc * sina * sinb * sinc
    return volume


def get_metal_count(atom_dict):
    """
    Takes an dictionary of atom counts (output from get_atom_dict);
    Returns the number of metals in the MOF/dictionary
    """
    metals = ['Li', 'Be', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga',
              'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Cs', 'Ba', 'La', 'Ce',
              'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os',
              'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm',
              'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh',
              'Fl', 'Mc', 'Lv']
    metal_count = 0

    for atom_type in atom_dict:
        if atom_type in metals:
            metal_count += atom_dict[atom_type]

    return metal_count


def get_metallic_percentage(atom_dict):
    """
    Returns the metallic percantage feature as described at the top of the document
    """
    metal_count = get_metal_count(atom_dict)

    if "C" in atom_dict:
        ratio = metal_count / atom_dict["C"]
    else:
        ratio = 0

    percent = ratio * 100
    return percent


def get_total_degree_of_unsaturation(atom_dict):
    """
    Takes an dictionary of atom counts (output from get_atom_dict);
    Returns the number of metals in the MOF/dictionary
    
    calculated as (((Number of carbons * 2) +2 - number of hydrogens) /2) 
    Halides F,Cl,Br, and I are counted as a Hydrogen, N is counted as half a Carbob
    """
    carbon_equivalent = 0
    hydrogen_equivalent = 0

    if "C" in atom_dict:
        carbon_equivalent += atom_dict["C"]

    if "N" in atom_dict:
        hydrogen_equivalent += atom_dict["N"] / 2

    if "H" in atom_dict:
        hydrogen_equivalent += atom_dict["H"]

    halides = ["F", "Cl", "Br", "I"]
    for halide_type in halides:
        if halide_type in atom_dict:
            hydrogen_equivalent += atom_dict[halide_type]

    return ((carbon_equivalent * 2) + 2 - hydrogen_equivalent) / 2


def get_electronegative_atom_ratio(atom_dict):
    """
    Returns the ratio of the number of electronegtive atoms to the total number of atoms in an atom_dict
    """
    electronegatives = ["O", "N", "F", "Cl", "Br"]
    electro_negative_count = 0
    total_count = 0
    for atom_type in atom_dict:
        total_count += atom_dict[atom_type]
        if atom_type in electronegatives:
            electro_negative_count += atom_dict[atom_type]
    ratio = electro_negative_count / total_count
    return ratio


def get_weighted_electronegative_atom_ratio(atom_dict):
    """
    Returns the ratio of the number of electronegtive atoms to the total number of atoms in an atom_dict
    
    Weighs only the electronegtivities of O, N, F, Cl, and Br. Electronegativities gathered from Wikipedia
    are on the Pauling Scale. 
    """
    electronegatives = {"O": 3.44, "N": 3.04, "F": 3.98, "Cl": 3.16, "Br": 2.96}
    summed_weight = 0
    total_count = 0
    for atom_type in atom_dict:
        total_count += atom_dict[atom_type]
        if atom_type in electronegatives:
            summed_weight += atom_dict[atom_type] * electronegatives[atom_type]
    ratio = summed_weight / total_count
    return ratio


def gather_features_main(directory, outputname):
    """
    Returns features listed in output text from cif files in directory; prints & writes output
    
    
    Parameters
    ----------
    directory: string
    adress of directory containing cif files e.g. 'C:/Users/ExampleUser/Documents/CifFiles/'
    
    outputname: string
    adress and name of output file (should contain extension) e.g. 'C:/Users/ExampleUser/Documents/Output.csv'
    """

    output_text = "MOF, H,C,N,F,Cl,Br,V,Cu,Zn,Zr,metal type, total degree of unsaturation,metalic percentage, oxygetn-to-metal ratio,electronegtive-to-total ratio, weighted electronegativity per atom, nitrogen to oxygen \n"

    element_feats = ['H', 'C', 'N', 'F', 'Cl', 'Br', 'V', 'Cu', 'Zn', 'Zr']
    metal_types = ['V', 'Cu', 'Zn', 'Zr']

    cif_files = [i for i in listdir(directory) if i[-4:] == ".cif"]
    if directory[-1] != "/":
        directory += "/"
    for cif in cif_files:
        new_line = cif[:-4] + ", "
        file_dir = directory + cif
        atom_dict = get_atom_dict(file_dir)
        unsaturation = get_total_degree_of_unsaturation(atom_dict)
        metalic_perct = get_metallic_percentage(atom_dict)
        weighted_EN = get_weighted_electronegative_atom_ratio(atom_dict)
        EN_ratio = get_electronegative_atom_ratio(atom_dict)

        for element in element_feats:
            if element in atom_dict:
                new_line += str(atom_dict[element]) + ", "
            else:
                new_line += "0, "

        for element in metal_types:
            if element in atom_dict:
                new_line += element
        new_line += ", " + str(unsaturation) + ", " + str(metalic_perct) + ", "

        if "O" in atom_dict:
            try:
                oxygen_to_metal = atom_dict["O"] / get_metal_count(atom_dict)
            except ZeroDivisionError:
                oxygen_to_metal = atom_dict["O"]
            if "N" in atom_dict:
                nitrogen_to_oxygen = atom_dict["N"] / atom_dict["O"]
            else:
                nitrogen_to_oxygen = 0
        else:
            oxygen_to_metal = 0
            nitrogen_to_oxygen = 0

        new_line += str(oxygen_to_metal) + ", " + str(EN_ratio) + ", " + str(weighted_EN) + ", " + str(
            nitrogen_to_oxygen) + "\n"
        output_text += new_line
        print(new_line)

    out = open(outputname, "w+")
    out.write(output_text)
    out.close


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""
Based on descriptors used in:
Pardakhti, M., Moharreri, E., Wanik, D., Suib, S. L., & Srivastava, R. (2017). Machine Learning Using Combined Structural and Chemical Descriptors for Prediction of Methane Adsorption Performance of Metal Organic Frameworks (MOFs). ACS Combinatorial Science, 19(10), 640-645. doi:10.1021/acscombsci.7b00056
reference : 10.1021/acsami.1c18521""")

    parser.add_argument(
        '--cifpath', '-c', type=str, help='(str) path of cif files'
    )
    parser.add_argument(
        '--output', '-o', type=str, help='(str) csv output file name'
    )

    args = parser.parse_args()

    assert os.path.exists(args.cifpath), f"{args.cifpath} does not exists"

    gather_features_main(args.cifpath, args.output)
