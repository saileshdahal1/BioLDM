import math
import copy
from pathlib import Path
import random 
from collections import namedtuple, Counter
import os
import numpy as np
import json
import torch
from torch import nn, einsum
import torch.nn.functional as F
import timeit
from einops import rearrange, reduce, repeat
from tqdm.auto import tqdm
from datetime import datetime
from torch.optim import AdamW
from transformers import get_scheduler, AutoTokenizer, PreTrainedTokenizerBase 
from transformers.models.bart.modeling_bart import BartForConditionalGeneration
import wandb
import data_handler as data_handler
import re
import torch.nn as nn
from perceiver_latent import PerceiverAutoEncoder
from evaluate import load

class BARTForConditionalGenerationLatent(BartForConditionalGeneration):
    def __init__(self, config, num_encoder_latents, num_decoder_latents, dim_ae, num_layers=2, l2_normalize_latents=False):
        super().__init__(config)
        self.num_encoder_latents = num_encoder_latents
        self.dim_ae = dim_ae
        self.l2_normalize_latents = l2_normalize_latents
        self.perceiver_ae = PerceiverAutoEncoder(dim_lm=config.d_model, num_encoder_latents=num_encoder_latents, num_decoder_latents=num_decoder_latents, dim_ae=dim_ae, depth=num_layers, transformer_decoder=True, l2_normalize_latents=l2_normalize_latents)

    def get_diffusion_latent(self, encoder_outputs, attention_mask):
        hidden_state = encoder_outputs[0]
        latent = self.perceiver_ae.encode(hidden_state, attention_mask.bool())
        return latent
        
    def get_decoder_input(self, diffusion_latent):
        return self.perceiver_ae.decode(diffusion_latent)
    
    # Map encoder outputs to decoder inputs
    def encoder_output_to_decoder_input(self, encoder_outputs, attention_mask):
        diffusion_latent = self.get_diffusion_latent(encoder_outputs, attention_mask)  
        encoder_outputs['last_hidden_state'] = self.get_decoder_input(diffusion_latent)
        return encoder_outputs
    
generate_kwargs = {'beam': {'max_length':64, 'min_length':5, 'do_sample':False, 'num_beams':4, 'no_repeat_ngram_size':3, 'repetition_penalty':1.2}}

def compute_perplexity(all_texts_list, model_id='gpt2-large'):
    torch.cuda.empty_cache() 
    perplexity = load("perplexity", module_type="metric")
    results = perplexity.compute(predictions=all_texts_list, model_id=model_id, device='cuda')
    return results['mean_perplexity']

def compute_bleu(all_texts_list, human_references):
    bleu = load("bleu")

    human_references = [[ref] for ref in human_references]
    results = bleu.compute(predictions=all_texts_list, references=human_references)
    
    return results['bleu']
def compute_rouge(all_texts_list, human_references, use_aggregator=True, use_stemmer=False):
    rouge = load("rouge")

    human_references = [[ref] for ref in human_references]
    results = rouge.compute(predictions=all_texts_list, references=human_references, use_aggregator=use_aggregator, use_stemmer=use_stemmer)
    
    return results

def get_output_dir(args):
    model_dir = f'{Path(args.dataset_name).stem}/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    output_dir = os.path.join(args.save_dir, model_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f'Created {output_dir}')
    return output_dir

def compute_grad_norm(parameters):
    parameters = [p for p in parameters if p.grad is not None]
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), p=2) for p in parameters]), p=2).item()
    return total_norm

def separate_weight_decayable_params(params):
    no_wd_params = [param for param in params if param.ndim < 2]
    wd_params = [param for param in params if param not in set(no_wd_params)]
    return wd_params, no_wd_params

def get_adamw_optimizer(params, lr, betas, weight_decay, eps=1e-8):
    params = list(params)
    wd_params, no_wd_params = separate_weight_decayable_params(params)
    param_groups = [
        {'params': wd_params},
        {'params': no_wd_params, 'weight_decay': 0},
    ]
    return AdamW(param_groups, lr = lr, weight_decay = weight_decay, betas=betas, eps=eps)

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def set_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_latent_model(args):
    config = BartForConditionalGeneration.from_pretrained(
        args.enc_dec_model).config
    lm = BARTForConditionalGenerationLatent.from_pretrained(
        args.enc_dec_model, config=config, num_encoder_latents=args.num_encoder_latents, num_decoder_latents=args.num_decoder_latents, dim_ae=args.dim_ae, num_layers=args.num_layers, l2_normalize_latents=args.l2_normalize_latents, _fast_init=False)
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        args.enc_dec_model)
    for (param_name, param) in lm.named_parameters():
        if re.fullmatch(".*perceiver.*", param_name):
            param.requires_grad = True
            print(f"Trainable: {param_name}")
        else:
            param.requires_grad = False
    return lm, tokenizer, config

class Trainer(object):
    def __init__(
        self,
        args,
        dataset_name,
        *,
        train_batch_size = 16,
        eval_batch_size = 64,
        train_lr = 1e-4,
        train_num_steps = 100000,
        lr_schedule = 'cosine',
        num_warmup_steps = 500,
        adam_betas = (0.9, 0.99),
        adam_weight_decay = 0.01,
        num_samples = None,
        eval_every = 1000,
        results_folder = './results',
        seed=43,
    ):
        super().__init__()
        set_seeds(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_num_steps = train_num_steps
        self.eval_every = eval_every
        self.args = args
        self.best_val_metric = 0
        self.num_samples = num_samples
        if args.output_dir is None:
            args.output_dir = get_output_dir(args)
        results_folder = args.output_dir
        with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        run = os.path.split(__file__)[-1].split(".")[0]
        wandb.init(project="your_project_name", name=args.wandb_name if args.wandb_name else run, dir=results_folder, config=vars(args))
        self.enc_dec_model = args.enc_dec_model
        self.lm, self.tokenizer, config = get_latent_model(args)
        self.lm.to(self.device)
        num_trainable_params = sum(p.numel() for p in self.lm.parameters() if p.requires_grad)
        print(f'Num trainable params: {num_trainable_params}')
        self.eval_every = eval_every
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.train_num_steps = train_num_steps
        self.dataset = data_handler.get_dataset(
            dataset_name,
        )
        if args.eval:
            self.dataset['train'] = self.dataset['train'].select(range(1000))
        self.dataloader = data_handler.get_dataloader(args, self.dataset['train'], config, self.tokenizer, args.max_seq_len, context_tokenizer=self.tokenizer)
        self.val_dataloader = data_handler.get_dataloader(args, self.dataset['valid'], config, self.tokenizer, args.max_seq_len, shuffle=False, context_tokenizer=self.tokenizer)
        self.max_seq_len = args.max_seq_len
        self.opt = get_adamw_optimizer(self.lm.parameters(), lr = train_lr, betas = adam_betas, weight_decay=adam_weight_decay)
        self.lr_scheduler = get_scheduler(
            lr_schedule, optimizer=self.opt, num_warmup_steps=num_warmup_steps, num_training_steps=train_num_steps
        )
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)
        self.step = 0
        self.data_iter = cycle(self.dataloader)
        self.val_iter = cycle(self.val_dataloader)
    
    def save(self):
        data = {
            'step': self.step,
            'model': self.lm.state_dict(),
            'opt': self.opt.state_dict(),
        }
        torch.save(data, str(self.results_folder / f'model.pt'))

    def load(self, file_path=None, resume_training=False):
        file_path = Path(file_path) if file_path else self.results_folder
        data = torch.load(str(file_path / 'model.pt'), map_location=self.device)

        self.lm.load_state_dict(data['model'])
        self.step = data['step']
        self.opt.load_state_dict(data['opt'])

        # Resume training scheduler steps
        if resume_training:
            for _ in range(self.step):
                self.lr_scheduler.step()
    
    def validation(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lm.eval()
        pred_text = {k:[] for k,_ in generate_kwargs.items()}    
        bart_text = {k:[] for k,_ in generate_kwargs.items()}    
        ref_text = []
        for batch in tqdm(self.val_dataloader):
            gen_kwargs = generate_kwargs['beam']
            gen_kwargs['max_length'] = self.max_seq_len
            data = {k:v.to(device) for k,v in batch.items()}
            encoder_outputs = self.lm.get_encoder()(input_ids = data['input_ids'], attention_mask = data['attention_mask'])
            encoder_outputs = self.lm.encoder_output_to_decoder_input(encoder_outputs, data['attention_mask'])
            sample_ids = self.lm.generate(encoder_outputs=encoder_outputs, **gen_kwargs)
            sample_ids = F.pad(sample_ids, (0, self.max_seq_len - sample_ids.shape[-1]), value=self.tokenizer.pad_token_id)
            gathered_sample_ids = sample_ids.to('cpu')
            texts_list = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip() for g in gathered_sample_ids]
            pred_text['beam'].extend(texts_list)
            sample_ids2 = self.lm.generate(input_ids = data['input_ids'], attention_mask = data['attention_mask'], **gen_kwargs)
            sample_ids2 = F.pad(sample_ids2, (0, self.max_seq_len - sample_ids2.shape[-1]), value=self.tokenizer.pad_token_id)
            gathered_sample_ids2 = sample_ids2.to('cpu')
            texts_list = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip() for g in gathered_sample_ids2]
            bart_text['beam'].extend(texts_list)
            gathered_input_ids = data['input_ids'].to('cpu')
            texts_list = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip() for g in gathered_input_ids]
            ref_text.extend(texts_list)
            if len(ref_text) > 1000:
                break
        metrics = {}
        metrics[f'autoencoder/bleu'] = compute_bleu(pred_text['beam'], ref_text)
        metrics[f'reference/bleu'] = compute_bleu(bart_text['beam'], ref_text)
        if all(pred_text['beam']):
            metrics[f'autoencoder/perplexity'] = compute_perplexity(pred_text['beam'])
        if all(bart_text['beam']):
            metrics[f'reference/perplexity'] = compute_perplexity(bart_text['beam'])
        rouge_metrics = compute_rouge(pred_text['beam'], ref_text)
        for k,v in rouge_metrics.items():
            metrics[f'autoencoder/{k}'] = v
        rouge_metrics = compute_rouge(bart_text['beam'], ref_text)
        for k,v in rouge_metrics.items():
            metrics[f'reference/{k}'] = v
        metrics['input/perplexity'] = compute_perplexity(ref_text)
        wandb.log(metrics, step=self.step)
        columns = ['reference'] + [f'autoencoder'] + [f'bart']
        data = []
        for i in range(len(ref_text)):
            row = [ref_text[i]]
            row.append(pred_text['beam'][i])
            row.append(bart_text['beam'][i])
            data.append(row)
        table = wandb.Table(columns=columns, data=data)
        wandb.log({f"Samples": table}, self.step)

    def train(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lm.train()
        with tqdm(initial=self.step, total=self.train_num_steps) as pbar:
            while self.step < self.train_num_steps:
                total_loss = 0.
                try:
                    batch = next(self.data_iter)
                except StopIteration:
                    self.data_iter = iter(self.dataloader)
                    batch = next(self.data_iter)
                data = {k: v.to(device) for k, v in batch.items()}
                with torch.amp.autocast("cuda"):  # Use "cuda" explicitly
                    encoder_outputs = self.lm.get_encoder()(input_ids=data['input_ids'], attention_mask=data['attention_mask'])
                    encoder_outputs = self.lm.encoder_output_to_decoder_input(encoder_outputs, data['attention_mask'])
                    loss = self.lm(labels=data['labels'], encoder_outputs=encoder_outputs).loss

                total_loss += loss.item()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.lm.parameters(), 1.0)

                # Optimizer step
                self.opt.step()
                self.lr_scheduler.step()
                self.opt.zero_grad()

                self.step += 1
                pbar.update(1)
                pbar.set_postfix(loss=total_loss)

                # Logging and validation
                if self.step % 50 == 0:
                    self.lm.eval()
                    with torch.no_grad():
                        total_val_loss = 0.
                        batch = next(self.val_iter)
                        data = {k: v.to(device) for k, v in batch.items()}
                        encoder_outputs = self.lm.get_encoder()(input_ids=data['input_ids'], attention_mask=data['attention_mask'])
                        encoder_outputs = self.lm.encoder_output_to_decoder_input(encoder_outputs, data['attention_mask'])
                        loss = self.lm(labels=data['labels'], encoder_outputs=encoder_outputs).loss
                        total_val_loss += loss.item()

                        logs = {
                            "train/loss": total_loss, 
                            "val/loss": total_val_loss,
                            "lr": self.lr_scheduler.get_last_lr()[0],
                            "step": self.step
                        }
                        wandb.log(logs, step=self.step)

                    self.lm.train()

                if self.step % self.eval_every == 0:
                    self.validation()
                    self.save()
                    self.lm.train()

        self.validation()
        self.save()
        print('Training complete')
