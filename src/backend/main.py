import tensorflow as tf
tf.random.set_seed(228)
import argparse

from train import Model

def parse_list(str_value):
    if ',' in str_value:
        str_value = str_value.split(',')
    else:
        str_value = [str_value]
    return str_value
 
parser = argparse.ArgumentParser(description='')

# ========================== GENERAL PARAMETERS ========================= #
parser.add_argument('--model_name',
                    dest='model_name',
                    default='model1',
                    help='Name of the model')
parser.add_argument('--phase',
                    dest='phase',
                    default='train',
                    help='Specify current phase: train or inference.')
parser.add_argument('--image_size',
                    dest='image_size',
                    type=int,
                    default=256*3,
                    help='For training phase: will crop out images of this particular size.'
                         'For inference phase: each input image will have the smallest side of this size. '
                         'For inference recommended size is 1280.')


# ========================= TRAINING PARAMETERS ========================= #
parser.add_argument('--ptad',
                    dest='path_to_art_dataset',
                    type=str,
                    #default='./data/vincent-van-gogh_paintings/',
                    default='./data/vincent-van-gogh_road-with-cypresses-1890',
                    help='Directory with paintings representing style we want to learn.')
parser.add_argument('--ptcd',
                    dest='path_to_content_dataset',
                    type=str,
                    default=None,
                    help='Path to Places365 training dataset.')


parser.add_argument('--total_steps',
                    dest='total_steps',
                    type=int,
                    default=int(3e5),
                    help='Total number of steps')

parser.add_argument('--batch_size',
                    dest='batch_size',
                    type=int,
                    default=1,
                    help='# images in batch')
parser.add_argument('--lr',
                    dest='lr',
                    type=float,
                    default=0.0002,
                    help='initial learning rate for adam')
parser.add_argument('--save_freq',
                    dest='save_freq',
                    type=int,
                    default=1000,
                    help='Save model every save_freq steps')
parser.add_argument('--ngf',
                    dest='ngf',
                    type=int,
                    default=32,
                    help='Number of filters in first conv layer of generator(encoder-decoder).')
parser.add_argument('--ndf',
                    dest='ndf',
                    type=int,
                    default=64,
                    help='Number of filters in first conv layer of discriminator.')

# Weights of different losses.
parser.add_argument('--dlw',
                    dest='discr_loss_weight',
                    type=float,
                    default=1.,
                    help='Weight of discriminator loss.')
parser.add_argument('--tlw',
                    dest='transformer_loss_weight',
                    type=float,
                    default=100.,
                    help='Weight of transformer loss.')
parser.add_argument('--flw',
                    dest='feature_loss_weight',
                    type=float,
                    default=100.,
                    help='Weight of feature loss.')
parser.add_argument('--dsr',
                    dest='discr_success_rate',
                    type=float,
                    default=0.8,
                    help='Rate of trials that discriminator will win on average.')


# ========================= INFERENCE PARAMETERS ========================= #
parser.add_argument('--ii_dir',
                    dest='inference_images_dir',
                    type=parse_list,
                    default=['./data/sample_photographs/'],
                    help='Directory with images we want to process.')
parser.add_argument('--save_dir',
                    type=str,
                    default=None,
                    help='Directory to save inference output images.'
                         'If not specified will save in the model directory.')

args = parser.parse_args()


def main(_):

    tfconfig = tf.compat.v1.ConfigProto(allow_soft_placement=False)
    tfconfig.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=tfconfig) as sess:
        model = Model(sess, args)

        if args.phase == 'train':
            print('Train')
            model.train(args)

        if args.phase == 'inference' or args.phase == 'test':
            print("Inference.")
            model.inference(args.inference_images_dir,
                            to_save_dir=args.save_dir)
        sess.close()
    print('FINISHED')

if __name__ == '__main__':
    tf.compat.v1.app.run()
