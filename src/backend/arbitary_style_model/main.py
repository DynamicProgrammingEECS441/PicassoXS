#
#   Copyright © 2020. All rights reserved.
#   python == 3.6 
#   tensorflow == 1.14
#   scipy==1.1.0
#


from train import train
from infer import inference
import argparse

def get_options():
    parser = argparse.ArgumentParser(description='')
    # Basic
    parser.add_argument("-mode", choices=['train', 'inference'], default='train')

    # Input
    parser.add_argument("-style_img_dir", help="./path/file to style image", default='WikiArt')
    parser.add_argument("-content_img_dir", help="./path/file to simple stich image", default='MS_COCO')

    # Model weight
    parser.add_argument("-checkpoint_encoder", help="encoder checkpoint", default='models/vgg19_normalised.npz')
    parser.add_argument("-checkpoint_model", help="model checkpoitn", default="models/style_weight_2e0.ckpt")
    parser.add_argument("-style_weight", help="weight for style", default=2)
    parser.add_argument("-content_weight", help="weight for style", default=1)

    # Save
    parser.add_argument("-checkpoint_save_dir", help="directory to save model", default='models/style_weight_2e0.ckpt')
    parser.add_argument("-output_dir", help="directory to save model", default='outputs')

    # Periodid
    parser.add_argument("-period_log", help="save log every few iteration", default=10)

    # Train
    parser.add_argument('-epoch', default=4, type=int)
    parser.add_argument('-epsilon', default=1e-5, type=float)
    parser.add_argument('-batch_size', default=8, type=int)
    parser.add_argument('-lr', default=1e-4, type=float)
    parser.add_argument('-lr_decay', default=5e-5, type=float)
    parser.add_argument('-lr_decay_step', default=1.0, type=float)
    parser.add_argument('-img_size', default=256, type=int)

    opt = parser.parse_args()
    print('opt', opt)

    return opt

def main():
    opt = get_options()

    if opt.mode == 'train':
        print('Train model with style weight {}'.format(opt.style_weight))
        train(opt)
        
    else:
        print('Inference model with style weight {} at {}'.format(opt.style_weight, opt.checkpoint_model))
        inference(opt)

    print('Finish')


if __name__ == '__main__':
    main()
