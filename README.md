# SBD-task-dependent
This code corresponds to the preprint **"Reducing the cost of posterior sampling in linear
inverse problems via task-dependent score learning"**.
## Setup

### Dependencies
The following command installs some of the most important python packages

`pip install -r requirements.txt`

### External Dependencies

This project includes code from the GitHub repository score_sde_pytorch, which is licensed under the Apache License 2.0.

#### LIDC-IRI Dataset

This project requires the LIDC-IRI Dataset and in order for the dataloader to work, need the downloaded images in .pt format in the folder specified in the code.
The dataset can be found at https://www.cancerimagingarchive.net/collection/lidc-idri/.

#### Modified Files
The configs in the folder `csgm` are slight adaptations of the folder csgm in the csgm repository. The original can be found at

- https://github.com/alisiahkoohi/csgm


The code from the csgm repository is licensed under the MIT License.

## Usage

### Training
The code that performs the training of a score function approximation is in the file `training.ipynb`.


### Posterior Sampling
For generating posterior samples as described in our paper, you can access the code in two separate notebooks: `sampling_deblurr.ipynb` for deblurring tasks and `sampling_ct.ipynb` for CT-imaging tasks. These notebooks contain the necessary code to generate posterior samples based on our approach.

## Pretrained models
Checkpoints of the pretrained unconditional model and the task-dependent models for deblurring and ct-imaging can be found in this
[Google drive](https://drive.google.com/drive/folders/1YIQzhNMF-5Mm24D_JcpMvuu-v7goz_w_). Download the checkpoints and place them in a folder named `checkpoints`.

## License

This project is licensed under the MIT License.


Please see the `LICENSE` file for details.

