# tensorboard --logdir ./whisper-fr-LoRA
# Import necessary modules and libraries

import os
import sys
import argparse
import logging
import warnings
from pathlib import Path
import librosa
import torch
from torch.utils.data import DataLoader
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset, Audio
from peft import LoraConfig, get_peft_model
import evaluate
from tqdm.auto import tqdm
import numpy as np
from accelerate import Accelerator
from accelerate.utils import set_seed

from utils import count_parameters, compute_module_sizes
from data_collate import DataCollatorSpeechSeq2SeqWithPadding

from huggingface_hub import login

# Setting up logging to show information level messages
logging.basicConfig(format="%(message)s", level=logging.INFO)

# Common transcript keys across datasets
TRANSCRIPT_KEYS = ["text", "sentence", "normalized_text", "transcript", "transcription"]

def initialize_accelerator(args):
    accelerator = Accelerator(
        mixed_precision='fp16',
        log_with="tensorboard",
        project_dir=args.output_dir
    )
    return accelerator

def set_environment(accelerator, seed):
    """
    Set random seed and adjust backend settings for optimal performance.

    Args:
        accelerator (Accelerator): Accelerator instance.
        seed (int): Seed for reproducibility.
    """
    set_seed(seed)
    if accelerator.device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    logging.info(f"Training on: {accelerator.device}, using mixed precision: {accelerator.mixed_precision}")


def load_and_prepare_datasets(args, processor):
    """
    Load and preprocess training and validation datasets.

    Args:
        args (argparse.Namespace): Parsed arguments.
        processor (WhisperProcessor): Whisper processor for handling text normalization.

    Returns:
        dict: Preprocessed 'train' and 'validation' datasets.
    """
    logging.info("Loading dataset...")

    login(token=args.auth_token)

    datasets = {}
    datasets["train"] = load_dataset(
      "google/fleurs",
      args.language,
      split="train",
      trust_remote_code=True
      )

    datasets["validation"] = load_dataset(
      "google/fleurs",
      args.language,
      split="validation",
      trust_remote_code=True
    )

    # Keep only the audio and transcript columns
    columns_to_keep = ["audio"] + TRANSCRIPT_KEYS  
    remove_columns = [col for col in datasets["train"].features.keys() if col not in columns_to_keep]
    datasets["train"] = datasets["train"].remove_columns(remove_columns)
    datasets["validation"] = datasets["validation"].remove_columns(remove_columns)

    # Use a small subset if in debug mode
    if args.debug:
        logging.info(f"Debug mode: Using only {args.debug_subset_size} samples.")
        datasets["train"] = datasets["train"].select(range(min(args.debug_subset_size, len(datasets["train"]))))
        datasets["validation"] = datasets["validation"].select(range(min(args.debug_subset_size, len(datasets["validation"]))))
    
    # Adjust the audio to the required sampling rate (16kHz for Whisper)
    sampling_rate = processor.feature_extractor.sampling_rate
    datasets["train"] = datasets["train"].cast_column("audio", Audio(sampling_rate=sampling_rate))
    datasets["validation"] = datasets["validation"].cast_column("audio", Audio(sampling_rate=sampling_rate))
    
    # --- Preprocessing for Training (with 30% Degradation) ---
    def preprocess_train(batch, idx):
        audio = batch["audio"]
        array = audio["array"]
        orig_sr = audio["sampling_rate"]

        # Apply degradation: Downsample to 8k and upsample back to 16k
        if idx % 10 < 3:
            degraded = librosa.resample(array, orig_sr=orig_sr, target_sr=8000)
            array = librosa.resample(degraded, orig_sr=8000, target_sr=orig_sr)
            
        # We only check a few samples to avoid slowing down training
        if idx < 30: 
            rolloff = librosa.feature.spectral_rolloff(y=array, sr=orig_sr, roll_percent=0.85)[0]
            avg_rolloff = np.mean(rolloff)
            
            status = "SUCCESS" if avg_rolloff < 4100 else "WARNING"
            logging.info(f"[Sample {idx}] Degradation Check: {status} (Avg Rolloff: {avg_rolloff:.2f} Hz)")
        # ------------------------------------------
            
        processed = processor(
            audio=array, 
            sampling_rate=orig_sr,
            text=batch["transcription"],
        )
        processed["input_length"] = len(array) / orig_sr
        return processed

    # --- Preprocessing for Validation (Clean Data) ---
    def preprocess_val(batch):
        audio = batch["audio"]
        processed = processor(
            audio=audio["array"],
            sampling_rate=audio["sampling_rate"],
            text=batch["transcription"],
        )
        processed["input_length"] = len(audio["array"]) / audio["sampling_rate"]
        return processed

    logging.info("Preprocessing training set with 30% degradation...")
    datasets["train"] = datasets["train"].map(
        preprocess_train,
        with_indices=True,
        num_proc=args.num_workers,
        remove_columns=["audio", "transcription"],
    )

    logging.info("Preprocessing validation set (clean)...")
    datasets["validation"] = datasets["validation"].map(
        preprocess_val,
        num_proc=args.num_workers,
        remove_columns=["audio", "transcription"],
    )

    # Filter out long audio samples
    def filter_length(batch):
        return [length < args.max_input_length for length in batch["input_length"]]
    
    logging.info("Filtering long audio samples...")
    datasets["train"] = datasets["train"].filter(
        filter_length,
        batched=True,
        batch_size=1000,
        num_proc=args.num_workers,
        desc="Filtering training set"
    )
    datasets["validation"] = datasets["validation"].filter(
        filter_length,
        batched=True,
        batch_size=1000,
        num_proc=args.num_workers,
        desc="Filtering validation set"
    )

    return datasets
def setup_model(processor):
    logging.info("Loading model for GradScaler (FP16 base + FP32 LoRA)...")
    
    # Load base model in FP16
    whisper_model = WhisperForConditionalGeneration.from_pretrained(
        "kingabzpro/whisper-large-v3-urdu",
        torch_dtype=torch.float16, 
        use_cache=False 
    )

    lora_config = LoraConfig(
        r=32,           
        lora_alpha=64,  
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none"
    )

    model = get_peft_model(whisper_model, lora_config)
    
    # GradScaler REQUIREMENT: Trainable parameters must be Float32
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)
            
    return model


def prepare_dataloaders(args, datasets, data_collator):
    """
    Create DataLoaders for the training and validation datasets.

    Args:
        args (argparse.Namespace): Parsed arguments containing data settings.
        datasets (dict): The preprocessed training and validation datasets.
        data_collator (DataCollatorSpeechSeq2SeqWithPadding): Handles padding for batches.

    Returns:
        tuple: (train_dataloader, validation_dataloader)
    """
    logging.info("Creating DataLoaders...")

    train_dataloader = DataLoader(
        datasets["train"],
        batch_size=args.train_batch_size,
        collate_fn=data_collator,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=True  # Randomize order of training samples
    )

    validation_dataloader = DataLoader(
        datasets["validation"],
        batch_size=args.train_batch_size,
        collate_fn=data_collator,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False  # No need to shuffle validation data
    )

    return train_dataloader, validation_dataloader


def setup_optimizer_scheduler(model, learning_rate):
    """
    Set up the optimizer and learning rate scheduler for training.

    Args:
        model (PeftModel): The model to be trained.
        learning_rate (float): The learning rate for the optimizer.

    Returns:
        tuple: (optimizer, scheduler)
    """
    logging.info("Setting up optimizer and learning rate scheduler...")

    # Train only the LoRA adapters, so we grab the parameters that require gradients
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    # AdamW optimizer for weight decay and better generalization
    optimizer = torch.optim.AdamW(params=trainable_params, lr=learning_rate)

    # Set up a learning rate scheduler that reduces the LR when validation loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',  # Minimize the validation loss
        factor=0.5,  # Halve the learning rate when triggered
        patience=2,  # How many epochs to wait before reducing
        verbose=True,  # Show updates
        min_lr=1e-6  # Set a floor for the learning rate
    )

    return optimizer, scheduler

def train_epoch(model, dataloader, optimizer, accelerator):
    model.train()
    total_loss = 0.0
    
    for batch in tqdm(dataloader, desc="Training", disable=not accelerator.is_local_main_process):
        # Accelerator handles moving and casting to fp16/fp32 automatically
        outputs = model(**batch)
        loss = outputs.loss

        # This triggers the GradScaler.scale(loss).backward() internally
        accelerator.backward(loss)
        
        if accelerator.sync_gradients:
            # GradScaler unscales the gradients here before clipping
            accelerator.clip_grad_norm_(model.parameters(), 1.0)

        # GradScaler.step() and GradScaler.update() happen here
        optimizer.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)


def validate(model, dataloader, processor, wer_metric, accelerator):
    """
    Validate the model on the validation set and calculate WER.

    Args:
        model (PeftModel): The trained model.
        dataloader (DataLoader): Validation DataLoader.
        processor (WhisperProcessor): Processor to decode predictions.
        wer_metric (evaluate.Metric): Metric to calculate Word Error Rate (WER).
        accelerator (Accelerator): Accelerator for distributed validation.

    Returns:
        tuple: (average validation loss, average WER)
    """
    model.eval()
    total_eval_loss = 0.0
    total_wer = 0.0

    progress_bar = tqdm(dataloader, desc="Validating", disable=not accelerator.is_local_main_process, leave=False)
    
    for batch in progress_bar:
        # Apply the same casting here
        batch = {k: v.to(device=accelerator.device, dtype=torch.float16 if v.dtype == torch.float32 else v.dtype) 
                 for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss
            loss = loss / accelerator.num_processes  # Normalize loss for distributed training
            total_eval_loss += loss.item()

            # Decode predictions and calculate WER
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            decoded_preds = processor.batch_decode(predictions, skip_special_tokens=True)

            labels = batch["labels"]
            labels = torch.where(labels != -100, labels, processor.tokenizer.pad_token_id)
            decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)

            wer = wer_metric.compute(predictions=decoded_preds, references=decoded_labels)
            total_wer += wer
    
    avg_eval_loss = total_eval_loss / len(dataloader)
    avg_wer = total_wer / len(dataloader)
    return avg_eval_loss, avg_wer


def save_model(model, processor, output_dir, accelerator):
    """
    Save the trained model and processor.

    Args:
        model (PeftModel): The fine-tuned model.
        processor (WhisperProcessor): The processor used with the model.
        output_dir (str): Directory where the model and processor should be saved.
        accelerator (Accelerator): Accelerator instance to ensure saving happens only once.
    """
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)  # Save only the LoRA adapters
        processor.save_pretrained(output_dir)
        logging.info(f"Model and processor saved to {output_dir}")

def train(args):
    # Initialize Accelerator
    accelerator = initialize_accelerator(args)

    # Set environment
    set_environment(accelerator, args.seed)

    # Load the Whisper processor
    logging.info("Loading Whisper processor...")
    processor = WhisperProcessor.from_pretrained("kingabzpro/whisper-large-v3-urdu", language="urdu", task="transcribe")

    # Load and preprocess datasets
    datasets = load_and_prepare_datasets(args, processor=processor)

    # Set up data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # Load model
    model = setup_model(processor)

    # Prepare DataLoaders
    train_dataloader, validation_dataloader = prepare_dataloaders(args, datasets, data_collator)
    
    # Initialize TensorBoard trackers
    if accelerator.is_main_process:
        accelerator.init_trackers("urdu_fine_tuning")

    # Set up optimizer and scheduler
    optimizer, scheduler = setup_optimizer_scheduler(model, args.learning_rate)

    # Prepare for training
    model, optimizer, train_dataloader, validation_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, validation_dataloader, scheduler
    )

    # Metrics and early stopping variables
    wer_metric = evaluate.load("wer")
    best_wer = float('inf')
    epochs_no_improve = 0

    logging.info("Starting the training process...")

    for epoch in tqdm(range(args.num_train_epochs), desc="Epochs", disable=not accelerator.is_local_main_process, leave=False):
        logging.info(f"\nEpoch {epoch + 1}/{args.num_train_epochs}")

        # Training
        avg_train_loss = train_epoch(model, train_dataloader, optimizer, accelerator)
        logging.info(f"Average training loss: {avg_train_loss:.4f}")

        # Validation
        avg_eval_loss, avg_wer = validate(model, validation_dataloader, processor, wer_metric, accelerator)
        logging.info(f"Validation loss: {avg_eval_loss:.4f}, WER: {avg_wer:.4f}")
        
        # Log to TensorBoard
        accelerator.log({
            "train/loss": avg_train_loss,
            "val/loss": avg_eval_loss,
            "val/wer": avg_wer,
            "train/learning_rate": optimizer.param_groups[0]['lr']
        }, step=epoch)

        # Scheduler step
        scheduler.step(avg_eval_loss)

        # Save Best Model
        if avg_wer < best_wer - args.early_stopping_min_delta:
            best_wer = avg_wer
            epochs_no_improve = 0
            # Save into a dedicated 'best_model' subfolder
            best_model_path = os.path.join(args.output_dir, "best_model")
            save_model(model, processor, best_model_path, accelerator)
            logging.info(f"New best WER: {best_wer:.4f}. Best model saved.")
        else:
            epochs_no_improve += 1
            logging.info(f"No improvement in WER for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= args.early_stopping_patience:
                logging.info("Early stopping triggered.")
                break

    # End training to flush logs
    accelerator.end_training()
    logging.info("Training process completed.")
    if accelerator.is_main_process:
        logging.info(f"Model and LoRA adapters have been saved to {args.output_dir}. You can now evaluate the performance.")
