# Deploy

> This file explains how to deploy the servable model on the AWS server 

2. 





## Kubernetes



* Basic 

1. 提供了在物理机或虚拟机集群上调度和运行容器的平台
2. **Master（主节点）：** 控制 Kubernetes 节点的机器，也是创建作业任务的地方。
3. **Node（节点）：** 这些机器在 Kubernetes 主节点的控制下执行被分配的任务。
4. **Pod：** 由一个或多个容器构成的集合，作为一个整体被部署到一个单一节点。同一个 pod 中的容器共享 IP 地址、进程间通讯（IPC）、主机名以及其它资源。Pod 将底层容器的网络和存储抽象出来，使得集群内的容器迁移更为便捷。
5. **Replication controller（复制控制器）：** 控制一个 pod 在集群上运行的实例数量。
6. **Service（服务）：** 将服务内容与具体的 pod 分离。Kubernetes 服务代理负责自动将服务请求分发到正确的 pod 处，不管 pod 移动到集群中的什么位置，甚至可以被替换掉。
7. **Kubelet：** 这个守护进程运行在各个工作节点上，负责获取容器列表，保证被声明的容器已经启动并且正常运行。
8. **kubectl：** 这是 Kubernetes 的命令行配置工具。









## Quick Start 

#### Start AWS AMI Instance

1. In general : follow the guid in this link to start an instance [official document](https://docs.aws.amazon.com/zh_cn/dlami/latest/devguide/launch-from-console.html)
2. For instance version : use  `AWS Deep Learning Base AMI (Ubuntu 18.04)` instance 



#### Connect to AWS Inatance







https://boto3.amazonaws.com/v1/documentation/api/latest/reference/core/resources.html



https://docs.aws.amazon.com/zh_cn/polly/latest/dg/example-Python-server-code.html



https://docs.aws.amazon.com/general/latest/gr/sigv4-signed-request-examples.html



https://docs.aws.amazon.com/zh_cn/IAM/latest/UserGuide/programming.html





## Reference 

1. [TF Server, REST](https://becominghuman.ai/creating-restful-api-to-tensorflow-models-c5c57b692c10)
2. [TF Server, K8n, Microsoft Azure](https://towardsdatascience.com/how-to-deploy-machine-learning-models-with-tensorflow-part-3-into-the-cloud-7115ff774bb6)
3. [blog k8s deploy tutorial](https://towardsdatascience.com/deploy-your-machine-learning-models-with-tensorflow-serving-and-kubernetes-9d9e78e569db)
4. [corresponding code to (3)](https://github.com/fpaupier/tensorflow-serving_sidecar)
5. [blog k8s deply tutorial](https://towardsdatascience.com/how-to-deploy-machine-learning-models-with-tensorflow-part-3-into-the-cloud-7115ff774bb6)
6. [tf server k8s deploy tutorial](https://www.tensorflow.org/tfx/serving/serving_kubernetes)
7. [ms azure k8s deploy tutorial](https://docs.microsoft.com/en-us/azure/aks/kubernetes-walkthrough)
8. [k8s official document](https://kubernetes.io/zh/docs/tasks/run-application/)
9. 







