#!/usr/bin/env bash
# Copyright 2017 Yahoo Inc.
# Licensed under the terms of the Apache 2.0 license.
# Please see LICENSE file in the project root for terms.
#
# This script install Spark locally

wget http://archive.apache.org/dist/spark/spark-1.6.0/spark-1.6.0-bin-hadoop2.6.tgz
gunzip spark-1.6.0-bin-hadoop2.6.tgz 
tar -xvf spark-1.6.0-bin-hadoop2.6.tar 
