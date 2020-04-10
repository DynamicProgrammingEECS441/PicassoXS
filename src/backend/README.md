# ***PicassoXS Backend***

EECS 441 Backend @ UMich Project 



## *Description*

This repository contains all files needed to develop the backend. 



## *Repo Structure*

```shell
.
├── arbitrary_style_model       # model for user upload own style 
│   ├── servable.py
│   ├── infer.py
│   ├── main.py
│   ├── model.py
│   ├── servable.py
│   ├── train.py
│   ├── utils.py
│   └── * others
├── flask_app                  # flask web app 
│   ├── __init__.py
│   ├── config.py
│   ├── * others
│   └── views
│       ├── __init__.py
│       └── index.py 
├── gcloud                     # instructions to start cloud server 
│   ├── Instruction.md
│   ├── st_k8s.yaml
│   └── * others
├── train                     # instructions on how to train model on server  
│   ├── Instruction.md
│   ├── tf1-gpu.yml
│   ├── tf2-gpu.yml
│   └── * others
├── general_model              # model for portrait mode model & general model 
│   ├── README.md
│   ├── StyleTransferrer.proto
│   ├── img_augm.py
│   ├── inference.sh
│   ├── layers.py
│   ├── main.py
│   ├── module.py
│   ├── prepare_dataset.py
│   ├── setup.py
│   ├── train.py
│   └── train.sh
└── tfserver                  # TensorFlow server : web app for serving model 
│   ├── Instruction.md
│   ├── PackDocker.md
│   ├── QuickStart_ArbitaryStyle.md
│   ├── QuickStart_ArbitaryStyleModel.md
│   ├── QuickStart_GeneralModel.md
│   ├── QuickPackServable.ipynb
│   ├── servable_demo.ipynb
│   ├── SendRequestArbitaryStyleModel_gRPC.py
│   ├── SendRequestGeneralModel_REST.py
│   ├── SendRequestGeneralModel_gRPC.py
│   ├── models.config
│   ├── *others 
│   └──servable             # SERVABLE : packed model that can be used by TensorFlow Server
│       └── * others 
└── README.md
└── app.yaml
└── flask_run.sh
└── requirements.txt
└── setup.py
```



# *General Model*



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
2. Edit the contents if needed (for GPU avalibility)
3. Run `./train.sh`



## *Inference*

To use inference mode, you must already have your trained model ready in the `$(pwd)/models/` directory. You will have to specify:
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



# *Arbitary Model*

> model that support users to upload their own style 



## *Training*

Put all content image inside one directory 

Put all style image inside one directory 



### Usage Example:

```shell
python main.py 
		-style_img_dir <path to style image> 
		-content_img_dir <path to contet image>
```



## *Inference*

Put all the image you want to test inside one directory 



### Usage Example:



```shell
python main.py 
		-style_img_dir <path to style image> 
		-content_img_dir <path to contet image>
		-checkpoint_encoder <path to encoder weight>
		-checkpoitn_model <path to model weight>
		-mode inference 
		-output_dir <output directory>
```



## *Reference*

Xun Huang, Serge Belonggie, Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization, ICCV 2017, https://arxiv.org/abs/1703.06868