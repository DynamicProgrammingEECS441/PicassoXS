# Google Cloud 



## Basic Info

1. Project Basic Info 
   1. gcloud project page [link](https://console.cloud.google.com/home/dashboard?project=tensorflow-serving-9905)
   2. project name : tensorflow-serving
   3. project id : tensorflow-serving-9905  (this is unique)
   4. project number : 57220460165  
2. Kubernete Engine : where the k8s cluster is hold 
3. Container Registry : hold out docker file 



## Kubernetes

* Basic 

1. 提供了在物理机或虚拟机集群上调度和运行容器的平台。可以跨多个机器进行调用
2. **Master（主节点）：** 控制 Kubernetes 节点的机器，也是创建作业任务的地方。
3. **Node（节点）：** 这些机器在 Kubernetes 主节点的控制下执行被分配的任务。
4. **Pod：** 由一个或多个容器构成的集合，作为一个整体被部署到一个单一节点。同一个 pod 中的容器共享 IP 地址、进程间通讯（IPC）、主机名以及其它资源。Pod 将底层容器的网络和存储抽象出来，使得集群内的容器迁移更为便捷。
5. **Replication controller（复制控制器）：** 控制一个 pod 在集群上运行的实例数量。
6. **Service（服务）：** 将服务内容与具体的 pod 分离。Kubernetes 服务代理负责自动将服务请求分发到正确的 pod 处，不管 pod 移动到集群中的什么位置，甚至可以被替换掉。
7. **Kubelet：** 这个守护进程运行在各个工作节点上，负责获取容器列表，保证被声明的容器已经启动并且正常运行。
8. **kubectl：** 这是 Kubernetes 的命令行配置工具。



## Start a k8s cluster 

1. pack the model into a docker image with `docker commit`

   ```shell
   docker run -d --name serving_base tensorflow/serving 
   docker cp $(pwd)/servable/${MODEL_NAME} serving_base:/models/${MODEL_NAME} 
   docker cp $(pwd)/servable/monet serving_base:/models/monet
   docker commit serving_base xiaosong99/servable:${MODEL_NAME}  
   docker kill serving_base
   docker rm serving_base 
   ```

2. test if the docker image you just packed work 

   ```shell
   docker run -p 8055:8500 -p 8501:8501 -t xiaosong99/servable:${MODEL_NAME}  
   ```

3. Log in to GCloud 

   ```shell
   gcloud auth login --project tensorflow-serving-9905 # use the project id instead of project name
   ```

4. start k8s containter cluster 

   ```shell
   #1. by cml 
   gcloud container clusters create ${CLUSTER_NAME} --num-nodes ${NUM_NODE}
   gcloud container clusters create test-k8s-cluster --num-nodes 4
   
   #2. by gcloud portal (recommend if you need customized configuration)
   ```

5. Link gcloud to k8s container 

   ```shell
   gcloud config set container/cluster ${CLUSTER_NAME}
   gcloud container clusters get-credentials  ${CLUSTER_NAME}
   ```

6. upload docker image into `container register`

   ```shell
   docker tag xiaosong99/servable:${MODEL_NAME} gcr.io/tensorflow-serving-9905/servable:${MODEL_NAME} 
   gcloud auth configure-docker
   docker push gcr.io/tensorflow-serving-9905/servable:${MODEL_NAME} 
   ```

7. configure k8s cluster 

   ```shell
   # NEED TO MODIFY the `st_k8s.yaml` file 
   kubectl create -f tensorflow_serving/example/st_k8s.yaml
   ```

8. check k8s deployment status

   ```shell
   kubectl get deployments
   kubectl get pods
   kubectl get services
   kubectl describe service tf-service
   ```

   

* other common use gcloud command 

```shell
>> gcloud config list # print out default configuration 
[accessibility]
screen_reader = true
[compute]
zone = us-central1-c
[container]
cluster = cluster-1
[core]
account = songxiao527@gmail.com
disable_usage_reporting = False
project = tensorflow-serving-9905
Your active configuration is: [default]

```





## Reference

9. [GCloud - Container Register Quick Start](https://cloud.google.com/container-registry/docs/quickstart)
10. [TF Server - k8s deploy tutorial](https://www.tensorflow.org/tfx/serving/serving_kubernetes)
3. [Tutorial - TF Server, K8S, MA Azure](https://towardsdatascience.com/deploy-your-machine-learning-models-with-tensorflow-serving-and-kubernetes-9d9e78e569db)







