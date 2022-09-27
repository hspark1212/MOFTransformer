import os
import argparse
import json
import pandas as pd


def geometric_descriptor_to_dataframe(di, df, gsa, vsa, vf, output='output.csv'):
    result = {}
    for path, label in zip([di, df, gsa, vsa, vf], ['di', 'df', 'gsa', 'vsa', 'vf']):
        assert os.path.exists(path), f"{path} does not exists"
        with open(path) as f:
            data = json.load(f)
            result[label] = data

    dataframe = pd.DataFrame(result)

    dataframe.to_csv(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""
    Based on descriptors used in:
    Pardakhti, M., Moharreri, E., Wanik, D., Suib, S. L., & Srivastava, R. (2017). Machine Learning Using Combined Structural and Chemical Descriptors for Prediction of Methane Adsorption Performance of Metal Organic Frameworks (MOFs). ACS Combinatorial Science, 19(10), 640-645. doi:10.1021/acscombsci.7b00056
    reference : 10.1021/acsami.1c18521""")

    parser.add_argument(
        '--largest-cavity-diameter', '-di', type=str,
        help='(str) path of json largest cavity diameter (di)'
    )
    parser.add_argument(
        '--pore-limiting-diameter', '-df', type=str,
        help='(str) path of json pore-limiting diamater (df)'
    )
    parser.add_argument(
        '--gravimetric-accessible-surface-area', '-gsa', type=str,
        help='(str) path of json gravimetric accessible surface area; m2/g (ASSA)'
    )
    parser.add_argument(
        '--volumetric-accessible-surface-area', '-vsa', type=str,
        help='(str) path of json volumetric accessible surface area; m2/g (ASSA)'
    )
    parser.add_argument(
        '--volume-fraction', '-vf', type=str,
        help='(str) path of json volume fraction (VF)'
    )
    parser.add_argument(
        '--output', '-o', type=str, help='(str) csv output file name'
    )

    args = parser.parse_args()

    df = args.pore_limiting_diameter
    di = args.largest_cavity_diameter
    gsa = args.gravimetric_accessible_surface_area
    vsa = args.volumetric_accessible_surface_area
    vf = args.volume_fraction
    output = args.output

    geometric_descriptor_to_dataframe(di, df, gsa, vsa, vf, output)
