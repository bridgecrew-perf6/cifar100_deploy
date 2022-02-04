# cifar100_deploy
Cifar100 Pytorch-Lightning SageMaker


#### YouTube Screen Recording



[![Screen Recording](https://i.postimg.cc/Wb0kJcT1/image.png)](https://youtu.be/VXOKb-g0y0I)






### Download Cifar100 data
> pip install cifar2png
> cifar2png cifar100 data



### Upload Raw files to S3
```
inputs = sagemaker_session.upload_data(path='./data', bucket=bucket, key_prefix=prefix)
print(f"s3 path is @ {inputs}")
```

### 4 GPU Managed Spot Training

```
estimator = PyTorch(
    entry_point='src/cifar.py', 
    dependencies=['src/requirements.txt'],
    role=role, 
    framework_version='1.8.0', 
    instance_count=1, 
    py_version='py3', 
    instance_type="ml.g4dn.12xlarge", 
    hyperparameters={"epochs": 4, "gpus": 4}, 
    use_spot_instances = True,
    max_run = 600,
    max_wait = 1200,
    checkpoints_s3_uri = "s3://sagemaker-af-south-1-509765771925/sagemaker/checkpoint"
)
```

```
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0,1,2,3]
Missing logger folder: /opt/ml/output/data/lightning_logs
  | Name     | Type   | Params
------------------------------------
0 | model_ft | ResNet | 21.3 M
------------------------------------
21.3 M    Trainable params
0         Non-trainable params
21.3 M    Total params
85.344    Total estimated model params size (MB)
```


### Inference Endpoint

```
predictor = estimator.deploy(
    initial_instance_count = 1, 
    instance_type="ml.m5.xlarge"
)
```

[![lobster-107132279-250.jpg](https://i.postimg.cc/dtKFYC7K/lobster-107132279-250.jpg)](https://postimg.cc/3W9VjdCL)


```
lobster = infer_transform(Image.open('lobster.jpg'))
response = predictor.predict(np.expand_dims(lobster, axis=0))
print(f"Prediction for image is -- {str_labels[np.argmax(response)]}")
```



> Prediction -- lobster







