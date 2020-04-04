CUDA_VISIBLE_DEVICES="" python main.py \
                 --model_name=model_debug \
                 --phase=train \
                 --image_size=762 \
                 --ptcd=$(pwd)/data/places \
                 --ptad=$(pwd)/data/style
