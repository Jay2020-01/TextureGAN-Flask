#/bin/sh

python main.py --display_port 7779 --gpu 0 --model texturegan --feature_weight 10 --pixel_weight_ab 1e5 --global_pixel_weight_l 100 --style_weight 0 --discriminator_weight 10 --learning_rate 1e-3 --learning_rate_D 1e-4 --save_dir ./save_dir --data_path ../../TextureGAN_data/dataset/training_shoes_pretrain --batch_size 1 --save_every 500 --num_epoch 100000 --input_texture_patch original_image --loss_texture original_image --local_texture_size 50 --discriminator_local_weight 100 --num_input_texture_patch 1
