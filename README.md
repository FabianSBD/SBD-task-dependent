# SBD-task-dependent
This code corresponds to the preprint **"Reducing the cost of posterior sampling in linear
inverse problems via task-dependent score learning"**.
## Setup

### Dependencies
The following command installs some of the most important python packages

`pip install -r requirements.txt`

### External Dependencies

This project includes code from the GitHub repository score_sde_pytorch, which is licensed under the Apache License 2.0.

#### Files Included

The following files are included from the score_sde_pytorch repository:

- https://github.com/yang-song/score_sde_pytorch/tree/main/models
- https://github.com/yang-song/score_sde_pytorch/blob/main/utils.py
- https://github.com/yang-song/score_sde_pytorch/blob/main/sde_lib.py

score_sde_pytorch repository: https://github.com/yang-song/score_sde_pytorch

To obtain a copy of the necessary files, run

`python setup.py`

#### Modified Files
The configs in the folder `custom_configs` are adaptations of the configs provided in the score_sde_pytorch repository. The original can be found at

- https://github.com/yang-song/score_sde/tree/main/configs


The code from the score_sde_pytorch repository is licensed under the Apache License, Version 2.0. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0. 

## Usage

### Training
The code that performs the training of a score function approximation is in the file `training.ipynb`.


### Posterior Sampling
For generating posterior samples as described in our paper, you can access the code in two separate notebooks: `sampling_deblurr.ipynb` for deblurring tasks and `sampling_ct.ipynb` for CT-imaging tasks. These notebooks contain the necessary code to generate posterior samples based on our approach.

## Pretrained models
Checkpoints of the pretrained unconditional model and the task-dependent models for deblurring and ct-imaging can be found in this
[Google drive](https://drive.google.com/drive/folders/1YIQzhNMF-5Mm24D_JcpMvuu-v7goz_w_). Download the checkpoints and place them in a folder named `checkpoints`.

## License

This project is dual-licensed under the MIT License and the Apache License 2.0.

- The code authored for this project is licensed under the MIT License.
- Portions of this codebase are derived from the score_sde_pytorch repository, which is licensed under the Apache License 2.0.

Please see the `LICENSE` and `LICENSE_APACHE` files for details.

