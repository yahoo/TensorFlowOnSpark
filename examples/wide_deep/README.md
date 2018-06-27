# wide&deep model with TensorflowOnSpark

wide deep model [wide and deep model](https://www.tensorflow.org/tutorials/wide_and_deep) is considered as one of the state-of-art model in recommendation system. the tutorial above links only could be run locally.

this example demonstrates how to implement distribution using TensorflowOnSpark(tfos).



## How to run

this example in running under cdh-hadoop, variable  $HADOOP_HOME is configured in the run script

```shell
export HADOOP_HOME=/opt/cloudera/parcels/CDH-5.11.0-1.cdh5.11.0.p0.34
```



however, it is confirmed that it could be run under native hadoop environment well, the native hadoop installed in test cluster is hadoop-2.7.2 [hadoop2.7.2 tutorial](https://hadoop.apache.org/docs/r2.7.2/),



change a bit to configure the export variables in the run.sh.

```shell
export HADOOP_HOME="/usr/local/hadoop-2.7.2/"
export LIB_HDFS="/usr/local/hadoop-2.7.2/lib/native"
export LIB_HADOOP="/usr/local/hadoop-2.7.2/lib/native"
```



the default field delimiter is "\t" in the train and evaluation log, then run with this

```shell
	nohup sh x.sh &
```



while finish the spark job, try to obtain logs `yarn logs -applicationId myappid`

there will be some similar output as following

```shell
2018-06-25 21:08:27,516 INFO (MainThread-20224) loss = 65.8497, step = 1641531 (12.828 sec)
2018-06-25 21:08:39,976 INFO (MainThread-20224) loss = 67.78998, step = 1652706 (12.460 sec)
2018-06-25 21:08:52,890 INFO (MainThread-20224) loss = 90.358444, step = 1663924 (12.914 sec)
2018-06-25 21:09:06,029 INFO (MainThread-20224) loss = 107.57178, step = 1675366 (13.139 sec)
2018-06-25 21:09:19,541 INFO (MainThread-20224) loss = 100.06123, step = 1686721 (13.512 sec)
2018-06-25 21:09:31,997 INFO (MainThread-20224) loss = 70.46988, step = 1697007 (12.456 sec)
2018-06-25 21:09:44,647 INFO (MainThread-20224) loss = 87.51865, step = 1707784 (12.650 sec)
2018-06-25 21:09:57,039 INFO (MainThread-20224) loss = 79.045105, step = 1718133 (12.391 sec)
2018-06-25 21:10:09,348 INFO (MainThread-20224) loss = 77.59738, step = 1727763 (12.309 sec)
2018-06-25 21:10:20,848 INFO (MainThread-20224) loss = 71.914154, step = 1736495 (11.500 sec)
2018-06-25 21:10:32,106 INFO (MainThread-20224) loss = 99.22389, step = 1744589 (11.258 sec)
2018-06-25 21:10:43,603 INFO (MainThread-20224) loss = 57.03349, step = 1753055 (11.497 sec)
2018-06-25 21:10:54,831 INFO (MainThread-20224) loss = 103.64122, step = 1761227 (11.228 sec)
2018-06-25 21:11:06,378 INFO (MainThread-20224) Loss for final step: 37.241173.
2018-06-25 21:11:06,379 INFO (MainThread-20224) Finished TensorFlow worker:17 on cluster node 82
18/06/25 21:11:06 INFO python.PythonRunner: Times: total = 2096696, boot = 374, init = 781, finish = 2095541
18/06/25 21:11:06 INFO executor.Executor: Finished task 82.0 in stage 0.0 (TID 82). 1268 bytes result sent to driver
```



model metric like auc could by found in the log as below

```shell
2018-06-25 21:24:07,724 INFO (MainThread-13640) Saving dict for global step 2007583: accuracy = 0.9842578, accuracy_baseline = 0.9842578, auc = 0.80605215, auc_prec

ision_recall = 0.055994354, average_loss = 0.070659064, global_step = 2007583, label/mean = 0.015742188, loss = 72.35488, precision = 0.0, prediction/mean = 0.01661

2884, recall = 0.0

```



