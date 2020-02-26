import argparse
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os
import sys

import utils


parser = argparse.ArgumentParser()
parser.add_argument('--split', default=0.8, help='Fraction of train/val split')
parser.add_argument('--output-dir', required=True, help= 'Output directory')
opt = parser.parse_args()

directories = [
    '../MICCAI_BraTS_2018_Data_Training/HGG', #210 patients
    '../MICCAI_BraTS_2018_Data_Training/LGG' #75 patients
]
    
def normalize(input_data):
    non_zero_mask = np.where(input_data > 0)
    masked_input_data = input_data[non_zero_mask]
    masked_mean = masked_input_data.mean()
    masked_std = masked_input_data.std()
    normalized_data = np.zeros(input_data.shape)
    normalized_data[non_zero_mask] = (input_data[non_zero_mask] - masked_mean) / (masked_std + 1e-12)
    return normalized_data, (normalized_data[non_zero_mask].mean(), normalized_data[non_zero_mask].std())

def main():

    patients = []

    for dir in directories:
        for patient in os.listdir(dir):
            if not patient.startswith('.'):
                patients.append({ 'patient': patient, 'path': os.path.join(dir, patient)})

    print(len(patients))

    #shuffle and split into train and validation sets
    np.random.shuffle(patients)
    split_index = int(opt.split * len(patients))
    training_set = patients[:split_index]
    validation_set = patients[split_index:]

    print('Training set: {} patients'.format(len(training_set)))
    print('Validation set: {} patients'.format(len(validation_set)))

    for item in training_set:
        for root, dirs, files in os.walk(item['path']):
            for file in files:
                if file.endswith('.nii.gz'):

                    full_file_path = os.path.join(root, file)

                    output_path = os.path.join(
                        opt.output_dir,
                        'train',
                        item['patient'],
                        file.replace('.nii.gz', '.nii')
                    )

                    # Create output directory if not yet exists
                    directory = os.path.dirname(output_path)
                    if not os.path.exists(directory):
                        os.makedirs(directory)

                    # Read NifTI file
                    data, affine = utils.read_nii(full_file_path)

                    # Normalize
                    if not file.endswith('seg.nii.gz'):
                        data, stats = normalize(data)

                    # Transpose
                    data = data.transpose(2, 0, 1)

                    # Save to file
                    utils.save_nii(data, output_path)

if __name__ == "__main__":
    main()
    
