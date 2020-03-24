# ***PicassoXS Backend***
EECS 441 Backend @ UMich Project 



## *Description*

This repository contains all files needed to develop the backend. 


## *Repo Structure*

```shell
.
├── README.md
├── gcloud      # codes for using google cloud to deploy 
│   ├── Instruction.md
│   └── * others 
├── img_augm.py
├── inference.sh
├── layers.py
├── main.py
├── module.py
├── prepare_dataset.py
├── tfserver        # codes for using TF Server 
│   ├── Instruction.md
│   └── * others 
├── train.py
└── train.sh
```

## *Training*
To train a general-purpose style model, two training datasets are needed: content images and style images. 

The content images we used for training is a subset of Places365, which can be downloaded [here](http://data.csail.mit.edu/places/places365/train_large_places365standard.tar).

The style images should include a bunch of paintings from a particular artistic style, e.g. paintings of a particular artist.

### Usage Example: 
```
CUDA_VISIBLE_DEVICES=0 python main.py \
                 --model_name=model_van-gogh \
                 --phase=train \
                 --ptcd=[path to content image folder] \
                 --ptad=[path to style image folder]
```

`--model_name=NAME` specifies the name of the model to save as after training is finished.

We also include a shell script `train.sh` for users to run the above command. To use the script, 

1. Run `chmod +x train.sh`
2. Edit the contents if needed
3. Run `./train.sh`

## *Inference*
To use inference mode, you must already have your trained model ready in the `./models/` directory. You will have to specify:
1. the name of your model
2. input directories of the images that you want to do style transfer on
3. The output directory to save the generated image
### Usage Example:
```
CUDA_VISIBLE_DEVICES=0 python main.py \
                 --model_name=[YOUR_MODEL_NAME] \
                 --phase=inference \
                 --image_size=1280 \
                 --ii_dir ../[INPUT_DIR_1]/,../[INPUT_DIR_2]/ \
                 --save_dir=../[OUTPUT_DIR]/
```
If you wish to use CPU instead of GPU, set `CUDA_VISIBLE_DEVICES=''`

### Command Line Options:
- `--model_name NAME` - the name of the model (all model should as subfolders in `./models/`)
- `--phase=inference` - phase has to be inference in this mode (NO NEED TO CHANGE)
- `--image_size SIZE` - resolution of the images to generate (each input image will be scaled so that the smaller side will have this size)
- `--ii_dir INPUT_DIR` - path to the folder containing target content images. You can specify multiple folder separated with commas (DO NOT USE SPACE INSTEAD)
- `--save_dir SAVE_DIR` - path to save the generated images

We also include a shell script `inference.sh` for users to run the above command. To use the script, 

1. Run `chmod +x inference.sh`
2. Edit the contents if needed
3. Run `./inference.sh`

Note that the default inference uses `model_van-gogh` and CPU setting

## *Reference*
Sanakoyeu, Artsiom et al. “A Style-Aware Content Loss for Real-Time HD Style Transfer.” Lecture Notes in Computer Science (2018): 715–731. Crossref. Web.
