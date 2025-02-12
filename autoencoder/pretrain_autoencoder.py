import numpy as np
import torch.nn.functional as F
import torch
import os 
import json
import sys
from trainer_class import Trainer
import argparse

def main(args):
    
    trainer = Trainer(
        args=args,
        dataset_name=args.dataset_name,
        train_batch_size = args.train_batch_size,
        eval_batch_size = args.eval_batch_size,
        train_lr = args.learning_rate,
        train_num_steps = args.num_train_steps,
        lr_schedule = args.lr_schedule,
        num_warmup_steps = args.lr_warmup_steps,
        adam_betas = (args.adam_beta1, args.adam_beta2),
        adam_weight_decay = args.adam_weight_decay,
        eval_every = args.eval_every,
        results_folder = args.output_dir
    )
    trainer.train()
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="umls")
    parser.add_argument("--max_seq_len", type=int, default=64)
    parser.add_argument("--enc_dec_model", type=str, default="GanjinZero/biobart-base")
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--num_encoder_latents", type=int, default=32)
    parser.add_argument("--num_decoder_latents", type=int, default=32)
    parser.add_argument("--dim_ae", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--l2_normalize_latents", action="store_true", default=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="autoencoder_pretrained")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_train_steps", type=int, default=50000)
    parser.add_argument("--lr_schedule", type=str, default="linear")
    parser.add_argument("--lr_warmup_steps", type=int, default=1000)
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--wandb_name", type=str, default="autoencoder_training")
    parser.add_argument("--lm_mode", type=str, default="freeze")
    parser.add_argument("--eval", action="store_true", default=False)
    parser.add_argument("--resume_training", action="store_true", default=False)
    parser.add_argument("--resume_dir", type=str, default=None)
    args = parser.parse_args()
    
    main(args)