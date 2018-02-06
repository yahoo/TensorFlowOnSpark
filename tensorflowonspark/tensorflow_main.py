# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def map_fun(argv, ctx):
	import argparse
	import copy
	import os
	import runpy
	import sys
	from tensorflowonspark import gpu_info
	from tensorflowonspark import process_code
	from functools import reduce

	user_argv = copy.deepcopy(sys.argv)
	sys.argv = argv[1:]
	user_parser = argparse.ArgumentParser()
	user_parser.add_argument("--udf_tf_pys", "--udf-tf-pys", help="user defined tensorflow py scripts", default=None)
	user_parser.add_argument("--udf_tf_main_py", "--udf-tf-main-py", help="user defined tensorflow main py",
													 default=None)
	user_parser.add_argument("--num_gpus", "--num-gpus", help="num of gpu", default=0)

	(user_args, user_rem) = user_parser.parse_known_args(sys.argv)
	udef_tf_pys = user_args.udf_tf_pys
	tf_main_py = user_args.udf_tf_main_py
	if not tf_main_py and udef_tf_pys and udef_tf_pys.endswith(".py"):
		tf_main_py = tf_pys
	origin_main_py = tf_main_py

	assert udef_tf_pys, "User code is None."
	if not os.path.exists(udef_tf_pys) and os.path.exists("__pyfiles__/{0}".format(udef_tf_pys)):
		udef_tf_pys = "__pyfiles__/{0}".format(udef_tf_pys)
	assert os.path.exists(udef_tf_pys), "{0} is not exists.".format(udef_tf_pys)
	assert tf_main_py, "Main py file is not set."


	# set code dir in PYTHONPATH
	def set_path(_dir):
		if not os.path.exists(_dir) or os.path.isfile(_dir):
			return
		_path = os.path.abspath(_dir)
		for name in os.listdir(_path):
			set_path(_path + "/" + name)
			sys.path.insert(0, os.path.dirname(_path))
	# begin process code package, set code dir in PYTHONPATH and get main python script
	if os.path.isdir(udef_tf_pys):
		udef_tf_pys = os.path.abspath(udef_tf_pys)
		set_path(udef_tf_pys)
	else:
		target = process_code.un_file(_file=udef_tf_pys)
		target = os.path.abspath(target)
		if os.path.isfile(target):
			target = os.path.dirname(target)
		set_path(target)

		if not os.path.exists(tf_main_py):
			pyfiles = "__pyfiles__"
			if os.path.exists(target + os.sep + tf_main_py):
				tf_main_py = target + os.sep + tf_main_py
			elif os.path.exists("{0}/{1}".format(pyfiles, tf_main_py)):
				tf_main_py = "{0}/{1}".format(pyfiles, tf_main_py)
			else:
				print("{0} not found in {1}.".format(tf_main_py, os.path.split(os.path.abspath(target))[-1]))
			if os.path.abspath(target) != os.path.dirname(tf_main_py):
				sys.path.insert(0, os.path.dirname(tf_main_py))
	if os.path.dirname(os.path.abspath(tf_main_py)) in sys.path:
		sys.path.remove(os.path.dirname(os.path.abspath(tf_main_py)))
	if "." in sys.path:
		sys.path.remove(".")
	if "" in sys.path:
		sys.path.remove("")
	sys.path.insert(0, os.path.dirname(os.path.abspath(tf_main_py)))
	sys.path.insert(0, os.getcwd())

	# add args to sys.argv
	user_argv.append("--default_fs")
	user_argv.append(ctx.defaultFS)
	# ADD Worker Num
	user_argv.append("--worker_num, {0}".format(ctx.worker_num))

	# Add Gpus num
	user_argv.append("--num_gpus {0}".format(user_args.num_gpus))

	cluster_spec = ctx.cluster_spec
	# ADD PS Hosts
	ps_hosts = cluster_spec.get("ps", [])
	if ps_hosts:
		hosts = reduce(lambda x, y: "{0},{1}".format(x, y), ps_hosts)
		user_argv.append("--ps_hosts {0}".format(hosts))

		# Add Worker Hosts
		worker_hosts = cluster_spec.get('worker', [])
		assert worker_hosts, "worker hosts is None."
		if worker_hosts:
			hosts = reduce(lambda x, y: "{0},{1}".format(x, y), worker_hosts)
			user_argv.append("--worker_hosts {0}".format(hosts))

		# ADD Job Name
		job_name = ctx.job_name
		user_argv.append("--job_name {0}".format(job_name))

		# ADD Task Index
		task_index = ctx.task_index
		user_argv.append("--task_index {0}".format(task_index))

	# set CUDA_VISIBLE_DEVICES
	if user_args.num_gpus:
		gpus_to_use = gpu_info.get_gpus(user_args.num_gpus)
		os.environ['CUDA_VISIBLE_DEVICES'] = gpus_to_use

	sys.argv = user_argv
	for arg in user_rem:
		sys.argv.append(arg)
		sys.argv[0] = origin_main_py
		assert os.path.exists(tf_main_py), "{0} not found in {1}.".format(tf_main_py, tf_pys)
	# lanuch code package
	runpy.run_path(path_name=tf_main_py, run_name='__main__')

if __name__ == '__main__':
	import sys
	import argparse
	import os
	from tensorflowonspark import TFCluster
	from pyspark.context import SparkContext

	parser = argparse.ArgumentParser()
	parser.add_argument("--num_ps", "--num-ps", help="number ps", type=int, default=0)
	parser.add_argument("--num_worker", "--num-worker", help="number worker", type=int, default=1)
	parser.add_argument("--udf_tf_pys", "--udf-tf-pys", help="user defined tensorflow py scripts", default=None)
	parser.add_argument("--udf_tf_main_py", "--udf-tf-main-py", help="user defined tensorflow main py", default=None)
	parser.add_argument("--tb", "--tensorboard", help="launch tensorboard", default=False)
	(args, rem) = parser.parse_known_args()

	input_mode = TFCluster.InputMode.TENSORFLOW
 	sc = SparkContext()
	num_executors = int(sc._conf.get("spark.executor.instances"))
	default_fs = sc._jsc.hadoopConfiguration().get("fs.defaultFS")
	if "hdfs://" not in default_fs:
		default_fs = "hdfs://%s" % default_fs
	num_ps = args.num_ps
	num_worker = num_executors - num_ps
	tf_pys = args.udf_tf_pys
	main_py = args.udf_tf_main_py
	if not main_py and tf_pys and tf_pys.endswith(".py"):
		main_py = tf_pys

	assert main_py, "user defined tensorflow main py is None."
	assert num_ps + num_worker == num_executors, "number of ps add number of worker not equal num_executors."
	assert num_ps >= 0, "number of ps not less than 0."
	assert num_worker >= 1, "number of worker not less than 1."
	tensorboard_dir = args.tb_dir

	rem.insert(0, sys.argv[0])
	rem.append('--udf_tf_main_py')
	rem.append(str(main_py))
	rem.append('--udf_tf_pys')
	rem.append(str(tf_pys))
	rem.append("--num_ps")
	rem.append(str(args.num_ps))
	rem.append("--num_worker")
	rem.append(str(args.num_worker))
	try:
		cluster = TFCluster.run(sc, map_fun, rem, num_executors, num_ps, args.tensorboard)
		cluster.shutdown()
	except:
		sys.exit(-1)
