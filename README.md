Train:
#python main.py --dataset Mimic3 --num_pre_epochs 500 --num_epochs 800 --epoch_ckpt_freq 100 --batch_size 256
#python main.py --dataset Mimic3 --num_epochs 800 --epoch_ckpt_freq 100 --batch_size 256 --resume_checkpoint Output/checkpoint/pretrain_vae_epoch_200.pth --num_pre_epochs 0
#python main.py --dataset Mimic3 --num_epochs 800 --epoch_ckpt_freq 100 --batch_size 256 --resume_checkpoint Output/checkpoint/pretrain_vae_epoch_200.pth --num_pre_epochs 500

python main.py --skip_pretrain --resume_checkpoint Output/checkpoint/best_pretrained_vae.pth --dataset Mimic3 --use_amp --batch_size 128

Test:
#python test.py --checkpoint Output/checkpoint/best_m3gan.pth --num_samples 5000



3080: python main.py --dataset Mimic3 --use_amp --batch_size 128
