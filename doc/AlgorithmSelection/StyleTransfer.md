# Style Transfer Algorithm Selection



### General 

We follow the algorithm category for style transfer by Yongcheng J et al [1](https://arxiv.org/pdf/1705.04058.pdf)



For each paper, we explore from 0-5

| data avalibility             | algorithm speed     | visual quality     | Portrate / general | complexity of algorithm structrue | Choice Score                     | Paper URL | Year |
| ---------------------------- | ------------------- | ------------------ | ------------------ | --------------------------------- | -------------------------------- | --------- | ---- |
| 5 : avalible  0 : unavilible | 5 : quick  0 : slow | 5 : good  0 : poor |                    | 5 : complex  0 : simple           | 5 : would choose  0 : not choose |           |      |
|                              |                     |                    |                    |                                   |                                  |           |      |

\<demo photo\>





### Overall Structure

* IOB-NST Image-optimization based online neural method 
  * Info : update the photo using model structure, model do not hold parameter represent style, ostly through markov randon field. 
* MPB-NST Model optimization based offline neural method 
  * Info : let the model learn parameter to represent statistical distcibution of style image 





### IOB-NST

> No need to train model 
>
> Choice of content and style layer is important 
>
> Time consuming during inference 



* Demystifying Neural Style Transfer

| data avalibility             | algorithm speed     | visual quality     | Portrate / general | complexity of algorithm structrue | Choice Score                     | Paper URL                            | Year |
| ---------------------------- | ------------------- | ------------------ | ------------------ | --------------------------------- | -------------------------------- | ------------------------------------ | ---- |
| 5 : avalible  0 : unavilible | 5 : quick  0 : slow | 5 : good  0 : poor |                    | 5 : complex  0 : simple           | 5 : would choose  0 : not choose | https://arxiv.org/pdf/1701.01036.pdf | 2017 |
| No need data                 | 2                   | 4                  | General            | 3                                 | 2                                |                                      |      |



Note:

1. global scale style transfer 



* Stable and Controllable Neural Texture Synthesis and Style Transfer Using Histogram Losses

| data avalibility             | algorithm speed     | visual quality     | Portrate / general   | complexity of algorithm structrue | Choice Score                     | Paper URL                            | Year |
| ---------------------------- | ------------------- | ------------------ | -------------------- | --------------------------------- | -------------------------------- | ------------------------------------ | ---- |
| 5 : avalible  0 : unavilible | 5 : quick  0 : slow | 5 : good  0 : poor |                      | 5 : complex  0 : simple           | 5 : would choose  0 : not choose | https://arxiv.org/pdf/1701.08893.pdf | 2017 |
| No need data                 | 2                   | 4                  | Portrate & Landscape | 4 (due to histogran loss)         | 2                                |                                      |      |



Note:

1. clear brush trock information 
2. **Histogram Loss is a good idea** 
3. Global Scale style transfer 



* Combining Markov Random Fields and Convolutional Neural Networks for Image Synthesis

| data avalibility             | algorithm speed                            | visual quality                           | Portrate / general | complexity of algorithm structrue  | Choice Score                     | Paper URL                            | Year |
| ---------------------------- | ------------------------------------------ | ---------------------------------------- | ------------------ | ---------------------------------- | -------------------------------- | ------------------------------------ | ---- |
| 5 : avalible  0 : unavilible | 5 : quick  0 : slow                        | 5 : good  0 : poor                       |                    | 5 : complex  0 : simple            | 5 : would choose  0 : not choose | https://arxiv.org/pdf/1601.04589.pdf | 2016 |
| No need data                 | 1 (nearest neighbor search on feature map) | 4 for small texture; 0 for large texture | Portrate & General | 4 (due to nearest neighbor search) | 1                                |                                      |      |



Note:

1. local scale (patch) style transfer 
2. Good when have small brush stroke (monet / van goh)
3. Poor when have large painting structure (Japanese painting) - it generally fails when the content and style images have strong differences in perspective and structure since the image patches could not be correctly matched. 



* Controlling Perceptual Factors in Neural Style Transfer

| data avalibility             | algorithm speed     | visual quality     | Portrate / general | complexity of algorithm structrue   | Choice Score                     | Paper URL                            | Year |
| ---------------------------- | ------------------- | ------------------ | ------------------ | ----------------------------------- | -------------------------------- | ------------------------------------ | ---- |
| 5 : avalible  0 : unavilible | 5 : quick  0 : slow | 5 : good  0 : poor |                    | 5 : complex  0 : simple             | 5 : would choose  0 : not choose | https://arxiv.org/pdf/1611.07865.pdf | 2016 |
| No need data                 | 2                   | 3                  | general            | 4 (due to the color, scale control) | 2                                |                                      |      |



Note:

1. **control over spatial location, colour inoformation** is a good idea 
2. dealing stylized sky, backgroudn, target differnetly 
3. **Large size, high quality stylized** 
4. Early stage of work, not good idea 



### MPB-NST







| data avalibility             | algorithm speed     | visual quality     | Portrate / general | complexity of algorithm structrue | Choice Score                     | Paper URL |
| ---------------------------- | ------------------- | ------------------ | ------------------ | --------------------------------- | -------------------------------- | --------- |
| 5 : avalible  0 : unavilible | 5 : quick  0 : slow | 5 : good  0 : poor |                    | 5 : complex  0 : simple           | 5 : would choose  0 : not choose |           |
|                              |                     |                    |                    |                                   |                                  |           |







| data avalibility             | algorithm speed     | visual quality     | Portrate / general | complexity of algorithm structrue | Choice Score                     | Paper URL |
| ---------------------------- | ------------------- | ------------------ | ------------------ | --------------------------------- | -------------------------------- | --------- |
| 5 : avalible  0 : unavilible | 5 : quick  0 : slow | 5 : good  0 : poor |                    | 5 : complex  0 : simple           | 5 : would choose  0 : not choose |           |
|                              |                     |                    |                    |                                   |                                  |           |











MOB-IR t: Model-Optimisation-Based Offline Image Reconstruction: learn & train network to represent High dimensional information, feed forward to get result 

1. Per-Style-Per-Model Neural Methods

2. 1. Parametric PSPM with Summary Statistics

   2. 1. 47/48 : 效果一般 Johson 
      2. 50/51: instance normalization 

   3. Non-parametric PSPM with MRFs

   4. 1. 52 (inspire by 46) : Markovian feed-forward network using adversarial training 效果更好，但是脸部的效果一般

      2. 1. Precomputed real-time texture synthesis with markovian generative adversarial networks

3. Multi-Style-Per-Model Neural Network 

4. 1. tying only a small number of parameters in a network to each style

   2. 1. 53 : A learned representation for artistic style,
      2. 54 : Stylebank: An explicit representation for neural image style transfer 

   3. l exploiting only a single network like PSPM but combining both style and content as inputs

   4. 1. 55: “Diversified texture synthesis with feed-forward networks,
      2. 56: Multi-style generative network for realtime transfer,” 

5. Arbitrary-Style-Per-Model Neural Methods

6. 1. Non-parametric ASPM with MRF

   2. 1. 57: Fast patch-based style transfer of arbitrary style

   3. Parametric ASPM with Summary Statistics.

   4. 1. 58 : “Exploring the structure of a real-time, arbitrary neural artistic stylization network,”
      2. 51: “Arbitrary style transfer in real-time with adaptive instance normalization,”
      3. 57 : “Fast patch-based style transfer of arbitrary style
      4. 59: “Universal style transfer via feature transforms,”

7. High Resolution 

8. 1. 62: coarse to fine stylization : “Multimodal transfer: A hierarchical deep convolutional neural network for fast artistic style transfer 
   2. 63 : depth preserved : Depth-aware neural style transfer 
   3. 





多个效果对比

https://compvis.github.io/adaptive-style-transfer/#references



对于不同类型的style，不同模型不一样





