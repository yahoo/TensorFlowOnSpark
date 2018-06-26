# wide&deep model with TensorflowOnSpark

wide deep model [wide and deep model ](https://www.tensorflow.org/tutorials/wide_and_deep) is considered as one of the state-of-art model in recommendation system. the tutorial above links only could be run locally.

this example demonstrates how to implement distribution using TensorflowOnSpark(tfos).



## How to run

this example in running under cdh-hadoop, however, it does not need extra dependencies. it is confirmed that it could be run under native hadoop environment well, the native hadoop installed in test cluster is hadoop-2.7.2[hadoop2.7.2 tutorial](https://hadoop.apache.org/docs/r2.7.2/),



change a bit to configure the export variables in the run.sh.

```shell
export HADOOP_HOME="/usr/local/hadoop-2.7.2/"
export LIB_HDFS="/usr/local/hadoop-2.7.2/lib/native"
export LIB_HADOOP="/usr/local/hadoop-2.7.2/lib/native"
```



the default field delimiter is "\t" in the train and evaluation log.

```shell
	nohup sh x.sh &
```



while finish the spark job, try to obtain logs `yarn logs -applicationId myappid`

there will be some similar output as following

![tfos job](log.jpg)



model metric like auc could by found in the log as below

`2018-06-25 21:24:07,724 INFO (MainThread-13640) Saving dict for global step 2007583: accuracy = 0.9842578, accuracy_baseline = 0.9842578, auc = 0.80605215, auc_prec
ision_recall = 0.055994354, average_loss = 0.070659064, global_step = 2007583, label/mean = 0.015742188, loss = 72.35488, precision = 0.0, prediction/mean = 0.01661
2884, recall = 0.0`
