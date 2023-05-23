#!/bin/bash
#SBATCH -A ADSAC
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=5G
#SBATCH --time=5-00:00:00
#SBATCH --mail-type=END

module load u18/cuda/10.2
module load u18/cudnn/7.6.5-cuda-10.2

rm -rf /scratch/amit/ 
mkdir /scratch/amit/



echo "Transfering train_data from ADA to node."
scp ada:/share3/dnaveenr/fillin_data.zip /scratch/amit/
scp ada:/share3/dnaveenr/i3d_200.zip /scratch/amit/
scp ada:/share3/dnaveenr/preprocessed_data.tar.gz /scratch/amit/
scp ada:/share3/dnaveenr/tokenizer.tar.gz  /scratch/amit/

echo "Extracting data..."
unzip -qo /scratch/amit/i3d_200.zip -d /scratch/amit/data
mv /scratch/amit/data/i3d2 /scratch/amit/data/i3d	

unzip -o /scratch/amit/fillin_data.zip -d /scratch/amit/data

tar xvzf /scratch/amit/preprocessed_data.tar.gz -C /scratch/amit/data/fillin_data
tar xvzf /scratch/amit/tokenizer.tar.gz -C /scratch/amit/




eval "$(conda shell.bash hook)"
conda activate python38


BASE_DIR="/scratch/amit"
mkdir -p $BASE_DIR/experiments/

echo "Training started..."


echo "Training started..."

#python -m ipdb train.py --input_json $BASE_DIR/data/fillin_data/LSMDC16_info_fillin_alert_index_gt.json             --input_fc_dir $BASE_DIR/data/i3d/                 --input_face_dir $BASE_DIR/data/fillin_data/face_features_rgb_mtcnn_cluster/                 --input_label_h5 $BASE_DIR/data/fillin_data/LSMDC16_labels_fillin_new_augmented.h5                 --clip_gender_json $BASE_DIR/data/fillin_data/LSMDC16_annos_gender.json --learning_rate 5e-5  --encoding_size 768 --gender_loss 0.2 --batch_size 16  --losses_print_every 1   --losses_log_every 1   --pre_nepoch 30 --save_checkpoint_every 5                --checkpoint_path $BASE_DIR/experiments/exp3

#python -m ipdb train.py --input_json $BASE_DIR/data/fillin_data/LSMDC16_info_fillin_alert_index_gt.json                  --input_fc_dir $BASE_DIR/data/i3d/                 --input_face_dir $BASE_DIR/data/fillin_data/face_features_rgb_mtcnn_cluster/                 --input_label_h5 $BASE_DIR/data/fillin_data/LSMDC16_labels_fillin_new_augmented.h5                 --clip_gender_json $BASE_DIR/data/fillin_data/LSMDC16_annos_gender.json --encoding_size 512 --learning_rate 5e-5  --classify_gender --gender_loss 0.2 --batch_size 16  --losses_print_every 1   --losses_log_every 1   --pre_nepoch 100 --save_checkpoint_every 5                --checkpoint_path $BASE_DIR/experiments/exp3 --overfit

#python -m ipdb train.py --input_json $BASE_DIR/data/fillin_data/LSMDC16_info_fillin_sos_eos_blank_mask.json                 --input_fc_dir $BASE_DIR/data/i3d/                 --input_face_dir $BASE_DIR/data/fillin_data/face_features_rgb_mtcnn_cluster/                 --input_label_h5 $BASE_DIR/data/fillin_data/LSMDC16_labels_fillin_new_augmented.h5                 --clip_gender_json $BASE_DIR/data/fillin_data/LSMDC16_annos_gender.json --encoding_size 512 --learning_rate 5e-5  --gender_loss 0 --batch_size 16  --losses_print_every 1   --losses_log_every 1   --pre_nepoch 100 --save_checkpoint_every 5                --checkpoint_path $BASE_DIR/experiments/exp3 #--overfit


# For experiments containing [CLS] after each sentence.
#python train.py --input_json $BASE_DIR/data/fillin_data/LSMDC16_info_fillin_indexed.json                 --input_fc_dir $BASE_DIR/data/i3d/                 --input_face_dir $BASE_DIR/data/fillin_data/face_features_rgb_mtcnn_cluster/                 --input_label_h5 $BASE_DIR/data/fillin_data/LSMDC16_labels_fillin_new_augmented.h5                 --clip_gender_json $BASE_DIR/data/fillin_data/LSMDC16_annos_gender.json --learning_rate 5e-5  --gender_loss 0.2 --batch_size 16  --losses_print_every 1   --losses_log_every 1   --pre_nepoch 30 --save_checkpoint_every 5                --checkpoint_path $BASE_DIR/experiments/exp3 --track

# Added <blank_token> in front of the captions.
#python train.py --input_json $BASE_DIR/data/fillin_data/LSMDC16_info_fillin_alert_index.json              --input_fc_dir $BASE_DIR/data/i3d/                 --input_face_dir $BASE_DIR/data/fillin_data/face_features_rgb_mtcnn_cluster/                 --input_label_h5 $BASE_DIR/data/fillin_data/LSMDC16_labels_fillin_new_augmented.h5                 --clip_gender_json $BASE_DIR/data/fillin_data/LSMDC16_annos_gender.json --learning_rate 5e-5  --gender_loss 0.2 --batch_size 16  --losses_print_every 1   --losses_log_every 1   --pre_nepoch 30 --save_checkpoint_every 5                --checkpoint_path $BASE_DIR/experiments/exp3 --track

#  Added <blank_token> in front of the captions and GT captions generated.
#python train.py --input_json $BASE_DIR/data/fillin_data/LSMDC16_info_fillin_alert_index_gt.json                  --input_fc_dir $BASE_DIR/data/i3d/                 --input_face_dir $BASE_DIR/data/fillin_data/face_features_rgb_mtcnn_cluster/                 --input_label_h5 $BASE_DIR/data/fillin_data/LSMDC16_labels_fillin_new_augmented.h5                 --clip_gender_json $BASE_DIR/data/fillin_data/LSMDC16_annos_gender.json --learning_rate 5e-5  --gender_loss 0 --batch_size 16  --losses_print_every 1   --losses_log_every 1   --pre_nepoch 100 --save_checkpoint_every 5                --checkpoint_path $BASE_DIR/experiments/exp3 --track

# With encoding_size parameter
#python train.py --input_json $BASE_DIR/data/fillin_data/LSMDC16_info_fillin_alert_index_gt.json                  --input_fc_dir $BASE_DIR/data/i3d/                 --input_face_dir $BASE_DIR/data/fillin_data/face_features_rgb_mtcnn_cluster/                 --input_label_h5 $BASE_DIR/data/fillin_data/LSMDC16_labels_fillin_new_augmented.h5                 --clip_gender_json $BASE_DIR/data/fillin_data/LSMDC16_annos_gender.json --encoding_size 768 --learning_rate 5e-5  --gender_loss 0 --batch_size 16  --losses_print_every 1   --losses_log_every 1   --pre_nepoch 1 --save_checkpoint_every 5                --checkpoint_path $BASE_DIR/experiments/exp3 --track

# Overfit Run
#python train.py --input_json $BASE_DIR/data/fillin_data/LSMDC16_info_fillin_alert_index_gt.json                  --input_fc_dir $BASE_DIR/data/i3d/                 --input_face_dir $BASE_DIR/data/fillin_data/face_features_rgb_mtcnn_cluster/                 --input_label_h5 $BASE_DIR/data/fillin_data/LSMDC16_labels_fillin_new_augmented.h5                 --clip_gender_json $BASE_DIR/data/fillin_data/LSMDC16_annos_gender.json --encoding_size 768 --learning_rate 5e-5  --gender_loss 0 --batch_size 16  --losses_print_every 1   --losses_log_every 1   --pre_nepoch 100 --save_checkpoint_every 5                --checkpoint_path $BASE_DIR/experiments/exp3 --track --overfit


# OverFit Run For Testing with latest features
# To Check if everything is working as expected.
#python train.py --input_json $BASE_DIR/data/fillin_data/LSMDC16_info_fillin_sos_eos_blank_mask.json                   --input_fc_dir $BASE_DIR/data/i3d/                 --input_face_dir $BASE_DIR/data/fillin_data/face_features_rgb_mtcnn_cluster/                 --input_label_h5 $BASE_DIR/data/fillin_data/LSMDC16_labels_fillin_new_augmented.h5                 --clip_gender_json $BASE_DIR/data/fillin_data/LSMDC16_annos_gender.json --use_bert_embedding --bert_embedding_dir $BASE_DIR/data/fillin_data/bert_text_gender_embedding/  --encoding_size 512 --learning_rate 5e-5  --gender_loss 0 --batch_size 16  --losses_print_every 1   --losses_log_every 1   --pre_nepoch 100 --save_checkpoint_every 5                --checkpoint_path $BASE_DIR/experiments/exp3 --overfit --track 


# Run using (Use BERT embeddings pretrained on Gender Classification task) - Best Performance 
#python train.py --input_json $BASE_DIR/data/fillin_data/LSMDC16_info_fillin_alert_index_gt_precomputed.json                    --input_fc_dir $BASE_DIR/data/i3d/                 --input_face_dir $BASE_DIR/data/fillin_data/face_features_rgb_mtcnn_cluster/                 --input_label_h5 $BASE_DIR/data/fillin_data/LSMDC16_labels_fillin_new_augmented.h5                 --clip_gender_json $BASE_DIR/data/fillin_data/LSMDC16_annos_gender.json --use_bert_embedding --bert_embedding_dir $BASE_DIR/data/fillin_data/bert_text_gender_embedding/  --encoding_size 512 --learning_rate 5e-5  --gender_loss 0 --batch_size 16  --losses_print_every 1   --losses_log_every 1   --pre_nepoch 30 --save_checkpoint_every 5                --checkpoint_path $BASE_DIR/experiments/exp3  --track 


# Load from JSON file with precomputed position, segment IDs, blank masks etc
#CUDA_LAUNCH_BLOCKING=1 python train.py --input_json $BASE_DIR/data/fillin_data/LSMDC16_info_fillin_sos_eos_blank_mask.json                   --input_fc_dir $BASE_DIR/data/i3d/                 --input_face_dir $BASE_DIR/data/fillin_data/face_features_rgb_mtcnn_cluster/                 --input_label_h5 $BASE_DIR/data/fillin_data/LSMDC16_labels_fillin_new_augmented.h5                 --clip_gender_json $BASE_DIR/data/fillin_data/LSMDC16_annos_gender.json --use_bert_embedding --bert_embedding_dir $BASE_DIR/data/fillin_data/bert_text_gender_embedding/  --encoding_size 512 --learning_rate 5e-5  --gender_loss 0 --batch_size 16  --losses_print_every 1   --losses_log_every 1   --pre_nepoch 30 --save_checkpoint_every 5                --checkpoint_path $BASE_DIR/experiments/exp3 

python train.py --input_json $BASE_DIR/data/fillin_data/LSMDC16_info_fillin_sos_eos_blank_mask.json                   --input_fc_dir $BASE_DIR/data/i3d/                 --input_face_dir $BASE_DIR/data/fillin_data/face_features_rgb_mtcnn_cluster/                 --input_label_h5 $BASE_DIR/data/fillin_data/LSMDC16_labels_fillin_new_augmented.h5                 --clip_gender_json $BASE_DIR/data/fillin_data/LSMDC16_annos_gender.json --use_bert_embedding --bert_embedding_dir $BASE_DIR/data/fillin_data/bert_text_gender_embedding/  --encoding_size 512 --learning_rate 5e-5  --gender_loss 0 --batch_size 2  --losses_print_every 1   --losses_log_every 1   --pre_nepoch 3 --save_checkpoint_every 5                --checkpoint_path $BASE_DIR/experiments/exp3  --overfit


# Run with Custom Gender Classification On 
#python train.py --input_json $BASE_DIR/data/fillin_data/LSMDC16_info_fillin_alert_index_gt.json                  --input_fc_dir $BASE_DIR/data/i3d/                 --input_face_dir $BASE_DIR/data/fillin_data/face_features_rgb_mtcnn_cluster/                 --input_label_h5 $BASE_DIR/data/fillin_data/LSMDC16_labels_fillin_new_augmented.h5                 --clip_gender_json $BASE_DIR/data/fillin_data/LSMDC16_annos_gender.json --encoding_size 512 --learning_rate 5e-5  --classify_gender --gender_loss 0.2 --batch_size 16  --losses_print_every 1   --losses_log_every 1   --pre_nepoch 30 --save_checkpoint_every 5                --checkpoint_path $BASE_DIR/experiments/exp3 --overfit --track

# Original LSMDC Fill-In Paper Run
#python train.py --input_json $BASE_DIR/data/fillin_data/LSMDC16_info_fillin_new_augmented.json                 --input_fc_dir $BASE_DIR/data/i3d/                 --input_face_dir $BASE_DIR/data/fillin_data/face_features_rgb_mtcnn_cluster/                 --input_label_h5 $BASE_DIR/data/fillin_data/LSMDC16_labels_fillin_new_augmented.h5                 --clip_gender_json $BASE_DIR/data/fillin_data/LSMDC16_annos_gender.json                 --use_bert_embedding --bert_embedding_dir $BASE_DIR/data/fillin_data/bert_text_gender_embedding/                 --learning_rate 5e-5  --gender_loss 0.2 --batch_size 64  --losses_print_every 1   --losses_log_every 1   --pre_nepoch 30 --save_checkpoint_every 5                --checkpoint_path $BASE_DIR/experiments/exp3
echo "Training finished..."
