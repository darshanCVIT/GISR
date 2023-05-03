#!/bin/bash
#SBATCH -A research
#SBATCH -n 36
#SBATCH --gres=gpu:4
#SBATCH -w gnode050
#SBATCH --mem-per-cpu=2048
#SBATCH --mail-user darshana.s@research.iiit.ac.in
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=ALL
module load u18/python/3.10.2

source /home2/darshana.s/miniconda3/etc/profile.d/conda.sh
echo "activating env"
conda activate CoFormer

mkdir -p /ssd_scratch/cvit/darshana.s/tdl
cd /ssd_scratch/cvit/darshana.s/tdl
mkdir CoFormer

#scp -r darshana.s@ada:/share3/darshana.s/clip_features_ViT_B32 .
scp -r darshana.s@ada:/share3/darshana.s/clip_features_ViT_L14_336px .
scp -r ~/CoFormer/SWiG .
scp darshana.s@ada:/share3/darshana.s/images_512.zip SWiG/
cd SWiG
unzip images_512.zip
cd ..


echo "starting script"
python -m torch.distributed.launch --nproc_per_node=4 --use_env ~/CoFormer/main.py \
           --backbone resnet50 --batch_size 4 --dataset_file swig --epochs 40 \
           --num_workers 4 --num_glance_enc_layers 3 --num_gaze_s1_dec_layers 3 \
           --num_gaze_s1_enc_layers 3 --num_gaze_s2_dec_layers 3 --dropout 0.15 --hidden_dim 512 \
           --output_dir CoFormer
