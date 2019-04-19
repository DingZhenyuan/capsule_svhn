# Capsules

##### 这里对Capsule模型进行改动,来测试其在SVHN数据集上的效果.

Code for Capsule model used in the following paper:
* "Dynamic Routing between Capsules" by
Sara Sabour, Nickolas Frosst, Geoffrey E. Hinton.

Requirements:
* TensorFlow (see http://www.tensorflow.org for how to install/upgrade)
* NumPy (see http://www.numpy.org/)
* GPU

Verify if the setup is correct by running the tests, such as:
```
python layers_test.py
```

My SVHN training command:

```
CUDA_VISIBLE_DEVICES=0 nohup python3 experiment.py 
--data_dir=$DIR/capsules/testdata/svhn/ 
--dataset=svhn --max_steps=300000 
--summary_dir=$DIR/capsules/attempt/ 
>> $DIR/capsules/output.log &
```

My SVHN testing command:
```
CUDA_VISIBLE_DEVICES=0 python3 experiment.py 
--data_dir=$DIR/capsules/testdata/svhn/ 
--train=false --dataset=svhn  
--summary_dir=$DIR/capsules/attempt01/ 
--checkpoint=$DIR/capsules/attempt01/train/model.ckpt-1500 
--test=true
```

---
