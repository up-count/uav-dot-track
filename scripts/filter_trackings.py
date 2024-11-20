import numpy as np
from glob import glob
from pathlib import Path
import click
import os

@click.command()
@click.option('--input', '-i', required=True, help='Path to the tracking directory')
@click.option('--min-len', '-m', default=45, help='Minimum length of a track to be kept')
def main(input, min_len):

    source = Path(input + '/data/')
    if not source.exists():
        print('Source directory does not exist, check the path')
        return
    
    files = sorted(glob(str(source) + '/*.txt'))

    name = input.split('/')[-2] if input[-1] == '/' else input.split('/')[-1]

    DEST = str(source).replace(name, 'filtered_'+name)

    Path(DEST).mkdir(parents=True, exist_ok=True)

    for file in files:
        data = np.loadtxt(file, delimiter=',').reshape(-1, 10)
        
        keep_ids = []

        un_id = np.unique(data[:, 1])

        for id in un_id:
            id_data = data[data[:, 1] == id]
            if len(id_data) >= min_len:
                keep_ids.append(id)

        keep_ids = np.array(keep_ids)

        keep_data = []

        for id in keep_ids:
            keep_data.append(data[data[:, 1] == id])

        if len(keep_data) == 0:
            keep_data = np.empty((0, 10))
        elif len(keep_data) == 1:
            keep_data = keep_data[0]
        else:
            keep_data = np.concatenate(keep_data)

        base = file.split('/')[-1]

        np.savetxt(DEST + '/' + base, keep_data, delimiter=',', fmt='%d,%d,%d,%d,%d,%d,%.2f,%.2f,%.2f,%.2f')
        

if __name__ == '__main__':
    main()
