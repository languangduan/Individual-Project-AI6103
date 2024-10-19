# AI6103 Individual Project

This repository contains the code and resources for my individual project in the Deep Learning course.

## Project Structure

- **data/**: Contains the CIFAR-100 dataset.
  - `cifar-100-python`: Extracted dataset files.
  - `cifar-100-python.tar.gz`: Compressed dataset file.
- **exp_bash/**: Bash scripts to run different experiments.
  - `activation_function_test.sh`: Script to test various activation functions.
  - `lr_schedule_test.sh`: Script to test different learning rate schedules.
  - `lr_test.sh`: Script to experiment with learning rates.
  - `weight_decay_test.sh`: Script to test the effect of weight decay.
- **experiment/**: Directory for experiment results and logs.
- **graphs/**: Contains scripts and files for generating graphs.
- **mobilenet_sigmoid/**: Directory for MobileNet with Sigmoid activation function.
- **models/**: Contains model definitions and related files.
- **.gitignore**: Specifies files and directories to be ignored by git.
- **graph.py**: Script for generating and handling graphs.
- **main.py**: Main script for running the project.
- **requirements.txt**: Lists all the dependencies required for the project.

## How to Run Experiments

1. Ensure all dependencies are installed using:

   ```bash
   pip install -r requirements.txt
   ```

2. Run any of the bash scripts in the `exp_bash` directory to start an experiment. For example:

   ```bash
   bash exp_bash/activation_function_test.sh
   ```

## Dataset

The CIFAR-100 dataset is used in this project. Make sure the dataset is properly extracted in the `data/` directory.

## Contact

For any questions or issues, please contact Duan Yiyang at YIYANG007@e.ntu.edu.sg.