#wide&deep with TensorflowOnSpark#

wide deep model [wide and deep model ](https://www.tensorflow.org/tutorials/wide_and_deep) is considered as one of the state-of-art model in recommendation system. the tutorial above links only could be run locally.

this example demonstrates how to implement distribution using TensorflowOnSpark(tfos).



## How to run

the default field delimiter is "\t" in the train and evaluation log.

>  nohup sh run.sh &

while finish the spark job, try to obtain logs `yarn logs -applicationId myappid`

there will be some similar output as following

![tfos job](https://github.com/crafet/TensorFlowOnSpark/blob/master/examples/wide_deep/success.jpg)