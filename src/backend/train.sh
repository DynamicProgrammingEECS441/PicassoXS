CUDA_VISIBLE_DEVICES="" python main.py \
                 --model_name=model_van-gogh_new \
                 --batch_size=1 \
                 --phase=train \
                 --image_size=768 \
                 --lr=0.0002 \
                 --dsr=0.8 \
                 --ptcd=/Users/zhangenhao/Pictures/20170402复旦同济 \
                 --ptad=./data/vincent-van-gogh_road-with-cypresses-1890