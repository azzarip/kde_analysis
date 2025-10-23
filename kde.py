#!/usr/bin/env python3
import sys
import os
import pandas as pd

max_index = 500
h = 0.01


def K1(x):
    """Epanechnikov kernel."""
    return 0.75 * (1 - x * x) if abs(x) < 1 else 0


def K(x, x0):
    """Scaled kernel function."""
    return K1((x - x0) / h) / h


def getKDE(data):
    """Compute KDE and normalize so that cumulative ends at 1."""
    index_values = [i / max_index for i in range(max_index+1)]
    result = pd.DataFrame(index=index_values, columns=['value'])
    result['value'] = 0.0
    result.index.name = 'time_s'

    for x in data:
        for i in result.index:
            result.loc[i, 'value'] += K(i, x)

    # Average
    result['value'] /= len(data)

    # Compute step size (dx)
    dx = result.index[1] - result.index[0]

    # Normalize so that the integral (sum(value * dx)) = 1
    result['value'] /= (result['value'].sum() * dx)

    # Compute cumulative distribution correctly
    result['cumulative'] = result['value'].cumsum() * dx
    result['cumulative'] /= result['cumulative'].iloc[-1]

    return result


def main():
    if len(sys.argv) < 2:
        print("Usage: python file.py <file_to_examine.csv|.xlsx>")
        sys.exit(1)

    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"Error: file '{file_path}' not found.")
        sys.exit(1)

    # Read input data
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.csv':
        df = pd.read_csv(file_path, header=0)
    elif ext in ['.xls', '.xlsx']:
        df = pd.read_excel(file_path)
    else:
        print("Error: Only .csv or .xlsx files are supported.")
        sys.exit(1)

    data = df.values.flatten().astype(float).tolist()

    print('Start of analysis... it could take some minutes to finish...')
    print('\n')
    # Compute KDE
    result = getKDE(data)

    # Save output
    dir_name = os.path.dirname(file_path) or '.'
    base_name = os.path.basename(file_path)
    out_path = os.path.join(
        dir_name, f"analyzed_{os.path.splitext(base_name)[0]}.csv")

    result.to_csv(out_path)
    print(f"Analysis complete. Saved to: {out_path}")


if __name__ == "__main__":
    main()
