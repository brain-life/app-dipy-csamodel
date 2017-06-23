import time
import numpy as np
import nibabel as nib
import json
import os, sys
from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.reconst.shm import CsaOdfModel
from dipy.data import default_sphere
from dipy.direction import peaks_from_model
from dipy.io.peaks import save_peaks, load_peaks

def main():
    start = time.time()

    with open('config.json') as config_json:
        config = json.load(config_json)
    
    # Load the data
    dmri_image = nib.load(config['data_file'])
    dmri = dmri_image.get_data()
    aparc_im = nib.load(config['freesurfer'] + "/mri/aparc+aseg.mgz")
    aparc = aparc_im.get_data()
    end = time.time()
    print('Loaded Files: ' + str((end - start)))

    # Create the white matter mask
    start = time.time()
    wm_regions = [2, 41, 16, 17, 28, 60, 51, 53, 12, 52, 12, 52, 13, 18,
                  54, 50, 11, 251, 252, 253, 254, 255, 10, 49, 46, 7]

    wm_mask = np.zeros(aparc.shape)
    for l in wm_regions:
        wm_mask[aparc == l] = 1
    print('Created white matter mask: ' + str(time.time()-start))

    # Create the gradient table from the bvals and bvecs
    start = time.time()
    bvals, bvecs = read_bvals_bvecs(config['data_bval'], config['data_bvec'])

    gtab = gradient_table(bvals, bvecs, b0_threshold=100)
    end = time.time()
    print('Created Gradient Table: ' + str((end - start)))

    # Create the csa model and calculate peaks
    start = time.time()
    csa_model = CsaOdfModel(gtab, sh_order=6)
    print('Created CSA Model: ' + str(time.time() - start))
    csa_peaks = peaks_from_model(csa_model, dmri, default_sphere,
                                 relative_peak_threshold=.8,
                                 min_separation_angle=45,
                                 mask=wm_mask)
    print('Generated peaks: ' +  str(time.time() - start))
    save_peaks('csa_peaks.pam5', csa_peaks)

main()
