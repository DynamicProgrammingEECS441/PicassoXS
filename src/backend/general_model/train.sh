CUDA_VISIBLE_DEVICES="" python main.py \
                 --model_name=model_debug \
                 --phase=train \
                 --ptcd=$(pwd)/data/places \
                 --ptad=$(pwd)/data/style