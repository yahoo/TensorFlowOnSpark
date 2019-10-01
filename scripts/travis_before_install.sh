#!/bin/bash

# TensorFlow 2.0.0 is tested/supported on Ubuntu 16 (xenial) or later
# But Travis' xenial build env uses JDK11, while Spark requires JDK8

# Install JDK8
sudo add-apt-repository -y ppa:openjdk-r/ppa
sudo apt-get update
sudo apt-get install -y openjdk-8-jdk --no-install-recommends
sudo update-java-alternatives -s java-1.8.0-openjdk-amd64

# Download and install Spark
curl -LO http://www-us.apache.org/dist/spark/spark-2.4.4/spark-2.4.4-bin-hadoop2.7.tgz
export SPARK_HOME=./spark
mkdir $SPARK_HOME
tar -xf spark-2.4.4-bin-hadoop2.7.tgz -C $SPARK_HOME --strip-components=1
export SPARK_LOCAL_IP=127.0.0.1
export SPARK_CLASSPATH=./lib/tensorflow-hadoop-1.0-SNAPSHOT.jar
export PATH=$SPARK_HOME/bin:$PATH
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64

# Update Python
# Note: TensorFlow 2.0.0 requires pip>19.0
pip install --upgrade pip
export PYTHONPATH=$(pwd)
