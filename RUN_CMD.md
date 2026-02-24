# ASR_Urdu

## Overview

ASR_Urdu is an Automatic Speech Recognition training project for Urdu language using Whisper-based models with LoRA fine-tuning.

## Environment Setup

Create and activate the Conda environment using one of the following commands:

```bash
conda env create -f environment.yml
```

OR (for full dependencies)

```bash
conda env create -f environment_all.yml
```

Then activate the environment:

```bash
conda activate asr_urdu
```

## Training

Run training using `main.py`:

```bash
python main.py train --num_train_epochs 10 --train_batch_size 4 --learning_rate 5e-5 --output_dir ./whisper-fr-LoRA --num_workers 1 --max_input_length 30 --auth_token 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
```

### Training Arguments

* `--num_train_epochs`: Number of training epochs
* `--train_batch_size`: Batch size for training
* `--learning_rate`: Learning rate
* `--output_dir`: Directory to save trained model
* `--num_workers`: Number of data loading workers
* `--max_input_length`: Maximum input audio length
* `--auth_token`: Authentication token (if required)

## Requirements

* Python >=3.10
* Conda environment
* CUDA-compatible GPU is recommended for training

## Project Structure

* `main.py`: Entry point for training and evaluation
* Dataset loaders and model configurations are included in the source code
