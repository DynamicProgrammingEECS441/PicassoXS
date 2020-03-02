# Demo 

* Reference 

[download docker instruction](https://docs.docker.com/install/linux/docker-ce/ubuntu/)

[docker tutorial](https://yeasy.gitbooks.io/docker_practice/image/pull.html)

[docker command document CH_zh](https://www.runoob.com/docker/docker-command-manual.html)

[tf server tutorial 1 - delpoy with tf1.x](https://zhuanlan.zhihu.com/p/52096200)

[tf server tutorial 2 - deploy with tf2.x keras](https://zhuanlan.zhihu.com/p/96917543)

[tf server tutorial 3 - multi model deploy](https://zhuanlan.zhihu.com/p/64413178)

[official document CH_zh](https://bookdown.org/leovan/TensorFlow-Learning-Notes/4-5-deploy-tensorflow-serving.html#serving-a-tensorflow-model--tensorflow-)

[tf official document - Serving a Tensorflow Model](https://www.tensorflow.org/tfx/serving/serving_basic)

[tf official document - tf1.x saved model builder ](https://www.tensorflow.org/api_docs/python/tf/compat/v1/saved_model/Builder)

[tf server official document - RESTful API 有空需要看](https://www.tensorflow.org/tfx/serving/api_rest)





## TODO 

查看是否能将 tf1.x 与 tf2.keras 结合在一起，把data preprocessing的部分加入到 tf2.keras中



## TF Note

* Difference between tf1.x tf2.x 

```python 
# 1. Eager mode in TF2

a = [1, 2, 3, 4]
b = [0.3, 0.1, 0,0]

# TF1 
with tf.Session() as sess:
    print(sess.run(tf.nn.softmax(a)))
    print(sess.run(tf.nn.softmax(b)))

# TF2 
sa = tf.nn.softmax(a)
sb = tf.nn.softmax(b)

# 2. Use TF1 funciton with TF2 
tf.compact.v1.<function>

# 3. Some tf1 funciton nolonger supported by tf2 
tf.placeholder() 
tf.variablescope()
tf.get_variable() 
  
```



* TF 2.x Basic information 

```python
# 4. TF2.Keras Build Model 

ValueError: This model has not yet been built. Build the model first by calling `build()` or calling `fit()` with some data, or specify an `input_shape` argument in the first layer(s) for automatic build.

```



* TF 1.x Basic Information 

```python
# 5. Session in TF 1.x

session.run() 进行了某种操作（E.g. 加法乘法）

matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],[2]])
product = tf.matmul(matrix1,matrix2)   # will not compute, becauuse not activate yet 

# method 1
sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()

# method 2
with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)
    
# 6. Variable 
在 Tensorflow 中，定义了某字符串是变量，它才是变量，这一点是与 Python 所不同的。

x = tf.Variable()


# 7. Placeholder 
Tensorflow 如果想要从外部传入data, 那就需要用到 tf.placeholder(), 然后以这种形式传输数据 sess.run(***, feed_dict={input: **})

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
ouput = tf.multiply(input1, input2)

with tf.Session() as sess:
    sess.run(ouput, feed_dict={input1: [7.], input2: [2.]})

```



* Simple TF1.x NN example 

```python
def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

x_data = np.linspace(-1,1,300, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
prediction = add_layer(l1, 10, 1, activation_function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                     reduction_indices=[1]))

optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
init = tf.global_variables_initializer()  # Initialize Global Variable 


sess = tf.Session()
sess.run(init)

for i in range(1000):
    # training
    sess.run(optimizer, feed_dict={xs: x_data, ys: y_data})
		
    # Compute loss 
    if i % 50 == 0:
      print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
```





## Docker

* Basic Ideas 

1. Repository : package of multiple images
2. Images : class 
3. Container : Instance of class 



* Install Docker  

```shell
# Remove old version 
sudo apt-get remove docker docker-engine docker.io containerd runc
 
sudo apt-get update
sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common
    
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
```



* Common use command 

```shell
# Check all images
docker images 

# Check all running container 
docker ps 

# Check all container (including previous container)
docker ps -a

# stop task 
docker kill <container_id>    # at terminal 

exit                          # inside docker 

# Prune all dangling container / images / volume 
docker system prune

# Remove image 
docker rmi <IMAGE_ID>

# Remove container 
docker rm CONTAINER_ID_or_NAME 

# Log into running docker 
docker ps                                                    # Find Container ID 
docker exec -it CONTAINER_ID_or_CONTAINER_NAME  /bin/bash    # Log into running container 

# Bind container's <docker_port> and map that to localhost's <machine_port>
docker run -p machine_port:docker_port 

# Bind local directory to container directory 
docker run -v local/path/:/container/path

# Docker set enviroment variable 
docker run -e MODEL_NAME=my_model

# Run docker in interactive mode & dummy access
docker run -it 
docker run -it tensorflow/serving:latest-devel /bin/bash

# Stop container 
docker stop

# Stop all container 
docker kill $(docker ps -q)

# Check all runnning task in unix 
ps -ef
ps -ef | grep <keyword_for_search>

# Kill running task in unix 
kill -9 <PID>

```



## Start TF Server

* Prepare servable 

see `servable_demo.ipynb` for prepare servable 



* use `tensorflow/serving:latest` to build docker that store and serve ONE servable 

```shell
(docker system prune)

# Start simple container (default using tf_serving inside simple container)
docker run -d --name serving_base tensorflow/serving:latest

# Copy servable into container
docker cp $(pwd)/tmp/generator/ serving_base:/models/generator

# Commmit change 
docker commit --change "ENV MODEL_NAME generator" serving_base only_generator_servable

# Run Docker
docker run -t -p 8501:8501 -p 8500:8500 only_generator_servable

```



* run tf server by mount servable on `tensorflow/serving:latest-devel`

```shell
# Run tfserver by mount 
docker run -t -p 8500:8500 -p 8501:8501 \
    -v $(pwd)/tmp/generator/:/models/generator \   # use -v as mapping 
    -e MODEL_NAME=generator \
    tensorflow/serving &
    
docker run -t --rm -p 8501:8501 \
    --mount type=bind,source=$(pwd)/model/,target=/models/model \  # use -mount as mount 
    -e MODEL_NAME=model \
    tensorflow/serving &
```



* Start TF Server inside container after copy servile into Images

```shell
# Start container 
docker run -it tensorflow/serving:latest-devel /bin/bash

# Create directory (inside docker)
cd ../../../../ && pwd && mkdir models

# Copy servable into container 
docker cp /your/local/file CONTAINER_ID_or_CONTAINER_NAME:/models/file
docker cp $(pwd)/tmp/generator/ 5703b69bc32e:/models/generator

# Commit Change (if not commit, change will loss)
docker commit CONTAINER_ID_or_CONTAINER_NAME REPOSITORY:NEW_TAG
docker commit 33313a26359d tensorflow/serving:xiaosong_test_1
docker commit 5703b69bc32e xiao_test_1

# Start container 
docker run -p localhost_port:container_port (--name NAME_FOR_CONTAINER) -it tensorflow/serving:latest-devel  bash
docker run -p 8501:8501 -p 8500:8500 -it tensorflow/serving:xiaosong_test_1 bash

# Run TF Server 
tensorflow_model_server --port=8500 --rest_api_port=8501 \
  --model_name=${MODEL_NAME} --model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME}
tensorflow_model_server --port=8500 --rest_api_port=8501 \
  --model_name=mnist --model_base_path=/models/mnist

# Check inside Docker container if tf server is runnig  
curl http://localhost:${PORT}/v1/models/${MODEL_NAME}
curl http://localhost:8501/v1/models/mnist
```



* Send request to TF Server 

see `servable_demo.ipynb` for information 



