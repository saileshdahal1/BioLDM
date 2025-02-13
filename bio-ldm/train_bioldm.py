import argparse
from transformers import AutoConfig
import json
import os
import numpy as np
import torch
from diffusion_trainer import GaussianDiffusion, Trainer
from diffusion_transformer import DiffusionTransformer

def main(args):
    latent_dim = 64
    lm_dim = 768
    model = DiffusionTransformer(
        tx_dim = args.tx_dim,
        tx_depth = args.tx_depth,
        heads = args.tx_dim//64, #ATTN_HEAD_DIM
        latent_dim = latent_dim,
        max_seq_len = args.max_seq_len,
        self_condition = args.self_condition,
        scale_shift = args.scale_shift,
        dropout = 0 if args.disable_dropout else 0.1,
        class_conditional= args.class_conditional,
        num_classes= 0,
        class_unconditional_prob= args.class_unconditional_prob,
        seq2seq=True,
        seq2seq_context_dim=lm_dim, 
        num_dense_connections=args.num_dense_connections,
    ).cuda()

    args.trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    diffusion = GaussianDiffusion(
        model,
        max_seq_len = model.max_seq_len,
        sampling_timesteps = args.sampling_timesteps,     # number of sampling steps
        sampler = args.sampler,
        train_schedule= args.train_schedule, 
        sampling_schedule= args.sampling_schedule,
        loss_type = args.loss_type,            # L1 or L2
        objective = args.objective,
        train_prob_self_cond = args.train_prob_self_cond,
        seq2seq_unconditional_prob = args.seq2seq_unconditional_prob,
        scale = args.scale,
    ).cuda()

    trainer = Trainer(
        args=args,
        diffusion=diffusion,
        dataset_name=args.dataset_name,
        train_batch_size = args.train_batch_size,
        eval_batch_size = args.eval_batch_size,
        gradient_accumulate_every = args.gradient_accumulation_steps,
        train_lr = args.learning_rate,
        train_num_steps = args.num_train_steps,
        lr_schedule = args.lr_schedule,
        num_warmup_steps = args.lr_warmup_steps,
        ema_update_every = args.ema_update_every,
        ema_decay = args.ema_decay,
        adam_betas = (args.adam_beta1, args.adam_beta2),
        adam_weight_decay = args.adam_weight_decay,
        save_and_sample_every = args.save_and_sample_every,
        num_samples = args.num_samples,
        seq2seq_candidates = args.seq2seq_candidates,
        results_folder = args.output_dir,
        amp = args.amp,
        mixed_precision = args.mixed_precision,
    )

    if args.eval:
        trainer.load(args.resume_dir, best=trainer.diffusion.diffusion_model.seq2seq)
        if trainer.diffusion.diffusion_model.seq2seq:
            trainer.sample_seq2seq(cls_free_guidance=2.0)
        else:
            trainer.sample()
        if args.class_conditional:
            for class_id in range(model.num_classes):
                trainer.sample(class_id=class_id)
        return
    if args.eval_test:
        trainer.load(args.resume_dir, best=trainer.diffusion.diffusion_model.seq2seq)
        if trainer.diffusion.diffusion_model.seq2seq:
            trainer.sample_seq2seq(split='test', cls_free_guidance=2.0, incremental=False)
        else:
            for seed in [42, 43, 44, 45, 46]:
                trainer.dataset = trainer.dataset.shuffle(seed)
                trainer.sample(seed=seed, test=True)
                if args.class_conditional:
                    for class_id in range(model.num_classes):
                        trainer.sample(class_id=class_id, seed=seed, test=True)
        return

    if args.resume_training:
        trainer.load(args.resume_dir)
    if args.init_path:
        trainer.load(args.init_path, init_only=True)
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training arguments")
    parser.add_argument("--dataset_name", type=str, default="umls")
    parser.add_argument("--save_dir", type=str, default="trained_bioldm")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default="bio-ldm-training")
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--num_train_steps", type=int, default=145000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--lr_schedule", type=str, default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=1000)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-6)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--ema_update_every", type=int, default=1)
    parser.add_argument("--objective", type=str, default="pred_v")
    parser.add_argument("--loss_type", type=str, default="l2")
    parser.add_argument("--train_schedule", type=str, default="cosine")
    parser.add_argument("--sampling_schedule", type=str, default=None)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--sampling_timesteps", type=int, default=250)
    parser.add_argument("--normalize_latent", action="store_true", default=False)
    parser.add_argument("--save_and_sample_every", type=int, default=100)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--seq2seq_candidates", type=int, default=5)
    parser.add_argument("--max_seq_len", type=int, default=64)
    parser.add_argument("--self_condition", action="store_true", default=True)
    parser.add_argument("--train_prob_self_cond", type=float, default=0.5)
    parser.add_argument("--sampler", type=str, default='ddpm')
    parser.add_argument("--enc_dec_model", type=str, default="GanjinZero/biobart-base")
    parser.add_argument("--tx_dim", type=int, default=768)
    parser.add_argument("--tx_depth", type=int, default=12)
    parser.add_argument("--scale_shift", action="store_true", default=True)
    parser.add_argument("--num_dense_connections", type=int, default=3)
    parser.add_argument("--disable_dropout", action="store_true", default=False)
    parser.add_argument("--class_conditional", action="store_true", default=False)
    parser.add_argument("--class_unconditional_prob", type=float, default=.1)
    parser.add_argument("--seq2seq_unconditional_prob", type=float, default=0.1)
    # Accelerate arguments
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--mixed_precision", type=str, default="no")
    parser.add_argument("--eval", action="store_true", default=False)
    parser.add_argument("--eval_test", action="store_true", default=False)
    parser.add_argument("--resume_training", action="store_true", default=False)
    parser.add_argument("--resume_dir", type=str, default=None)
    parser.add_argument("--latent_model_path", type=str, default="../autoencoder/autoencoder_pretrained/umls/2025-02-01_19-00-00")
    parser.add_argument("--init_path", type=str, default=None)
    
    args = parser.parse_args()
    assert not (args.eval and args.resume_training)
    if args.eval or args.resume_training:
        assert args.resume_dir is not None

    if args.eval or args.resume_training or args.eval_test:
        with open(os.path.join(args.resume_dir, 'args.json'), 'rt') as f:
            saved_args = json.load(f)
        args_dict = vars(args)
        # Hold out sampling/evaluation parameters
        heldout_params = {'wandb_name', 'output_dir', 'resume_dir', 'eval', 'eval_test', 'num_samples', 'sampling_timesteps', 'sampling_schedule', 'seq2seq_candidates', 'scale', 'sampler', 'resume_training'}
        # Overwrite args with saved args
        for k,v in saved_args.items():
            if k in heldout_params:
                continue
            args_dict[k] = v
    main(args)
