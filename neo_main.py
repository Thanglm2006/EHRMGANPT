# neo_main.py
import sys
import os

# Add the neo_m3gan directory to the START of sys.path to ensure
# local imports inside it (like 'import trainer') resolve to the neo versions.
sys.path.insert(0, os.path.abspath("neo_m3gan"))

from neo_m3gan.main import main
import argparse

if __name__ == '__main__':
    # Reuse the same parser structure from original main.py for compatibility
    parser = argparse.ArgumentParser(description="Neo M3GAN: Experimental Variant")
    parser.add_argument('--dataset', type=str, default="mimic", choices=['Mimic3', 'eicu', 'hirid'])
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--pre_batch_size', type=int, default=1024)
    parser.add_argument('--num_pre_epochs', type=int, default=500)
    parser.add_argument('--num_epochs', type=int, default=800)
    parser.add_argument('--epoch_ckpt_freq', type=int, default=100)
    parser.add_argument('--d_rounds', type=int, default=2)
    parser.add_argument('--g_rounds', type=int, default=1)
    parser.add_argument('--v_rounds', type=int, default=1)
    parser.add_argument('--v_lr_pre', type=float, default=0.0005)
    parser.add_argument('--v_lr', type=float, default=0.0001)
    parser.add_argument('--g_lr', type=float, default=0.0001)
    parser.add_argument('--d_lr', type=float, default=0.0001)
    parser.add_argument('--alpha_re', type=float, default=1)
    parser.add_argument('--alpha_kl', type=float, default=0.5)
    parser.add_argument('--alpha_mt', type=float, default=0.1)
    parser.add_argument('--alpha_ct', type=float, default=0.1)
    parser.add_argument('--c_beta_adv', type=float, default=1)
    parser.add_argument('--c_beta_fm', type=float, default=10.0)
    parser.add_argument('--d_beta_adv', type=float, default=1.0)
    parser.add_argument('--d_beta_fm', type=float, default=10.0)
    parser.add_argument('--enc_layers', type=int, default=3)
    parser.add_argument('--dec_layers', type=int, default=3)
    parser.add_argument('--gen_num_units', type=int, default=512)
    parser.add_argument('--gen_num_layers', type=int, default=3)
    parser.add_argument('--dis_num_layers', type=int, default=3)
    parser.add_argument('--resume_checkpoint', type=str, default=None)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--skip_pretrain', action='store_true')
    parser.add_argument('--use_amp', action='store_true')

    args = parser.parse_args()
    
    # Run the main function from the neo_m3gan package
    # NOTE: Output paths, etc., will be relative to the root directory
    main(args)
