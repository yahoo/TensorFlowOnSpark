#!/bin/bash -x

# Install JDK8
yum install -y java-1.8.0-openjdk
export JAVA_HOME=/usr/lib/jvm/jre-1.8.0

# Install Spark
export SPARK_VERSION=2.4.7
curl -LO http://www-us.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop2.7.tgz
mkdir $SPARK_HOME
tar -xf spark-${SPARK_VERSION}-bin-hadoop2.7.tgz -C $SPARK_HOME --strip-components=1
