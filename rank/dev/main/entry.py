#!/usr/bin/env python
# encoding: utf-8
import importlib
import json
import os
import sys
import time
from datetime import datetime

import tensorflow as tf
from tensorflow.python.framework import graph_util

sys.path.append(os.getcwd())
from utils.util import get_evaluation_all, Timer
from conf.config import cfg
import pandas as pd

network_type = cfg.network_type

pkg = importlib.import_module('model.%s' % network_type)
Model = pkg.__getattribute__(network_type)

model_type_set = ['wide', 'wide_rw', 'wide_deep', 'wide_deep_dcn', 'wide_fm', 'wide_deep_self', 'wide_deep_trans']


class ModelMangerBase(object):

    def __init__(self, mode, data_files_train, data_files_val, data_files_test, stat):

        """The location of models may need to be adjusted"""
        if cfg.mode == 'train' or cfg.mode == 'train_goon':
            self.model = Model('train', data_files_train, data_files_val, data_files_test, stat)
        elif cfg.mode == 'predict':
            self.model = Model('predict', data_files_train, data_files_val, data_files_test, stat)
        else:
            raise Exception("mode type is error,must in train/predict")

        self.steps = 0
        self.start_epoch = 0

    def initialize_session(self, ckpt_path):
        if cfg.gpu_mem < 1.0:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=cfg.gpu_mem,
                                        allow_growth=True)
        else:
            gpu_options = tf.GPUOptions()
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False,
                                device_count={"CPU": cfg.cpu_num},
                                inter_op_parallelism_threads=4,
                                intra_op_parallelism_threads=4
                                )

        with self.model.graph.as_default():
            # profie hook
            if not cfg.local_mode:
                # hooks = [
                #    tf.train.ProfilerHook(save_steps=200000, output_dir=ckpt_path, show_memory=True, show_dataflow=True),
                #    tf.train.SummarySaverHook(save_steps=5000, output_dir=self.tensorboard_dir,
                #                              scaffold=tf.train.Scaffold(summary_op=tf.summary.merge_all()))]
                hooks = [tf.train.ProfilerHook(save_steps=400000, output_dir=self.tensorboard_dir, show_memory=True,
                                               show_dataflow=True)]
                if self.model.sync_train:
                    sync_replicas_hook = self.model.optimizer.make_session_run_hook(is_chief=(self.task_index == 0),
                                                                                    num_tokens=0)
                    hooks.append(sync_replicas_hook)
                self.session = tf.train.MonitoredTrainingSession(master=self.server.target,
                                                                 is_chief=(self.task_index == 0),
                                                                 checkpoint_dir=None,
                                                                 save_checkpoint_steps=None,
                                                                 # forbbdien the checkpoint savers
                                                                 save_checkpoint_secs=None,
                                                                 save_summaries_steps=None,
                                                                 save_summaries_secs=None,
                                                                 hooks=hooks)
                ckpt_finetune = tf.train.get_checkpoint_state(cfg.restore_localpath)
                ckpt = tf.train.get_checkpoint_state(ckpt_path)
                if ckpt_finetune and ckpt_finetune.model_checkpoint_path:
                    # fine tune process and load pre-trained model
                    self.model.saver.restore(self.session, ckpt_finetune.model_checkpoint_path)
                    self.start_epoch = 0
                elif ckpt and ckpt.model_checkpoint_path:
                    self.model.saver.restore(self.session, ckpt.model_checkpoint_path)
                    self.start_epoch = int(
                        ckpt.model_checkpoint_path.split('/')[-1].split('.')[0].split('_')[-1].split('-')[0]) + 1
                else:
                    self.session.run(self.model.init_all_vars)
            else:
                print("use local session")
                # logger.info("use local session")
                self.session = tf.Session(config=config)
                ckpt_finetune = tf.train.get_checkpoint_state(cfg.restore_localpath)
                ckpt = tf.train.get_checkpoint_state(ckpt_path)
                if ckpt and ckpt.model_checkpoint_path:
                    # 先初始化所有参数，避免出现未初始化变量的error
                    self.session.run(self.model.init_all_vars)
                    self.model.saver.restore(self.session, ckpt.model_checkpoint_path)
                    self.start_epoch = int(
                        ckpt.model_checkpoint_path.split('/')[-1].split('.')[0].split('_')[-1].split('-')[0]) + 1
                    # self.session.run(self.model.init_all_vars)
                elif ckpt_finetune and ckpt_finetune.model_checkpoint_path:
                    # fine tune process and load pre-trained model
                    self.session.run(self.model.init_all_vars)
                    self.model.saver.restore(self.session, ckpt_finetune.model_checkpoint_path)
                    self.start_epoch = 0
                else:
                    self.session.run(self.model.init_all_vars)

        print ("===runing graph:", self.session.graph)
        # logger.info("===runing graph:", self.session.graph)
        if self.model.train_mode != 'predict':
            self.test_handle = self.session.run(self.model.test_iterator.string_handle())
            if self.model.train_mode == 'train':
                self.training_handle = self.session.run(self.model.training_iterator.string_handle())
                self.validation_handle = self.session.run(self.model.validation_iterator.string_handle())

    def train(self, n_epochs, ckpt_path, tensorboard_dir):
        self.model.train_mode = "train"
        self.model.build_graph()
        self.train_base(n_epochs, ckpt_path, tensorboard_dir)

    def ckpt_to_pb(self,ckpt_path):
        '''
        :param input_checkpoint:
        :param output_graph: PB模型保存路径
        :return:
        '''
        # checkpoint = tf.train.get_checkpoint_state(model_folder) #检查目录下ckpt文件状态是否可用
        # input_checkpoint = checkpoint.model_checkpoint_path #得ckpt文件路径

        # 指定输出的节点名称,该节点名称必须是原模型中存在的节点
        if type(self.model).__name__ == "wide_ctr_cvr":
            output_node_names = "output_layer/predict_value,output_layer/predict_value_cvr"
        else:
            output_node_names = "output_layer/predict_value"
        output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=self.session,
            input_graph_def=self.session.graph_def,# 等于:sess.graph_def
            output_node_names=output_node_names.split(","))# 如果有多个输出节点，以逗号隔开

        builder = tf.saved_model.builder.SavedModelBuilder(ckpt_path)
        builder.add_meta_graph_and_variables(self.session,['serve'])
        builder.save()

        print("%d ops in the final graph." % len(output_graph_def.node)) #得到当前图有几个操作节点

    def train_base(self, n_epochs, ckpt_path, tensorboard_dir):
        "Training initialization"
        self.initialize_session(ckpt_path)

        self.session.run(tf.tables_initializer())

        batch_num = int(self.model.train_size / cfg.batch_size)

        timer = Timer()

        # Tensorboard Logs
        self.summary_writer = tf.summary.FileWriter(tensorboard_dir, tf.get_default_graph())

        ops_to_run = [self.model.trainer, self.model.loss, self.model.assert_op, self.model.nnz]
        ops_to_single = [self.model.trainer, self.model.assert_op]
        ops_with_summary = [self.model.trainer, self.model.loss, self.model.assert_op, self.model.nnz,
                            self.model.merged]
        g_step = 0
        for i_epoch in range(self.start_epoch, n_epochs):
            self.steps = 0
            self.session.run(self.model.training_iterator.initializer)
            while True:
                try:
                    # if self.steps > 8000:
                    #     break

                    if self.steps % (cfg.print_period) == 0 or self.steps == 1:
                        print("--------------------Logging Tensorboard Info----------------------")
                        _, loss, assertion, nnz, summaries = self.session.run(ops_with_summary, feed_dict={
                            self.model.handle: self.training_handle, self.model.is_train: True})
                        self.summary_writer.add_summary(summaries, g_step)
                    elif self.steps % (cfg.print_period) == 0 or self.steps == 1:
                        _, loss, assertion, nnz = self.session.run(ops_to_run,
                                                                   feed_dict={self.model.handle: self.training_handle,
                                                                              self.model.is_train: True})
                    else:
                        # _, loss, assertion, nnz = self.session.run(ops_to_run,
                        #                                            feed_dict={self.model.handle: self.training_handle,
                        #                                                       self.model.is_train: True})
                        _, assertion = self.session.run(ops_to_single,
                                                        feed_dict={self.model.handle: self.training_handle,
                                                                   self.model.is_train: True})

                    if self.steps % cfg.print_period == 0 or self.steps == 1:
                        print ('%s|sample epoch: %d: %d/%d, global step:%d -- loss: %.4f, nnz:%.4f, duration:%.4f'
                               % (
                                   datetime.now().strftime('%H:%M:%S'), i_epoch, self.steps, batch_num, g_step, loss,
                                   nnz,
                                   timer.elapsed()))

                    # eval validation set
                    if self.steps > 0 and self.steps % (cfg.eval_period) == 0:
                        val_auc = get_evaluation_all(self.session, self.model, self.validation_handle, g_step,
                                                     'validation')
                        auc_summary = tf.Summary(value=[
                            tf.Summary.Value(tag='glo_auc', simple_value=val_auc)
                        ])
                        self.summary_writer.add_summary(auc_summary, g_step)

                    self.steps += 1
                    g_step += 1
                except tf.errors.OutOfRangeError:
                    print('Done training for %d epochs, %d steps.' % (i_epoch, self.steps))
                    # logger.info('Done training for %d epochs, %d steps.' % (i_epoch, self.steps))
                    # global_step_tensor = training_util._get_or_create_global_step_read(self.session.graph)
                    # global_step_all = self.session.run(global_step_tensor)
                    save_path = self.model.saver.save(self.session, ckpt_path + '/epoch_' + str(i_epoch),
                                                      global_step=g_step)
                    print("checkpoint saved in path: %s" % save_path)
                    # test data eval,consume much time
                    _ = get_evaluation_all(self.session, self.model, self.test_handle, g_step, 'test', istop=True)
                    break
            self.steps = 0

        # global_step_tensor = training_util._get_or_create_global_step_read(self.session.graph)
        # global_step_all = self.session.run(global_step_tensor)
        # the last validation step
        val_auc = get_evaluation_all(self.session, self.model, self.validation_handle, g_step, 'validation')
        auc_summary = tf.Summary(value=[
            tf.Summary.Value(tag='glo_auc', simple_value=val_auc)
        ])
        self.summary_writer.add_summary(auc_summary, g_step)
        assert val_auc > 0.58
        save_path = self.model.saver.save(self.session,ckpt_path + '/epoch_' + str(n_epochs),
                                          global_step=g_step)
        print("last time checkpoint saved in path: %s" % save_path)

        self.ckpt_to_pb(ckpt_path + '/pb')
        # destroy session
        self.session.close()

    def freezy_rawwide(self, checkpoint_dir, wide_dict_path, wide_outpath):
        # restore last checkpoint and get final bias value
        self.model.build_graph()
        checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            print('restore last checkpoint', checkpoint.model_checkpoint_path)
            # logger.info('restore last checkpoint',checkpoint.model_checkpoint_path)
            saver.restore(sess, checkpoint.model_checkpoint_path)
            bias = sess.run('wide_layer_variable/bias:0')
            wide_weight = sess.run('wide_layer_variable/w:0')
            # wide_dict = load_dict(wide_dict_path)
            # print('wide dict size:',len(wide_dict))
            with open(wide_outpath, "w+") as file:
                # file.writelines('bias'+","+str(bias)+'\n')
                for idx, weight in enumerate(wide_weight):
                    if abs(weight[0]) > 1e-8:
                        file.writelines(str(idx) + "," + str(weight[0]) + '\n')
            print("%s|Freeze Model Done!" % datetime.now().strftime('%H:%M:%S'))

    def freezy_wide(self, checkpoint_dir, wide_dict_path, wide_outpath, stat_path, raw_widepath):
        '''

        :param checkpoint_dir: model train ckpt dir for get the last ckpt state
        :param wide_dict_path: wide dict path,such as name,idx,hash
        :param wide_outpath: final wide model output path,such as hash,weight
        :param stat_path: final stat file output path
        :param raw_widepath: raw wide model output path,such as name,weight
        :return: none
        '''
        # restore last checkpoint and get final bias value

        self.model.build_graph()
        checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
        tmp_path = raw_widepath + "_tmp_" + str(int(time.time()))
        print ('start saver')
        saver = tf.train.Saver()
        print ('saver done')
        cnt = 0
        print('start  export weight')
        dict_df = pd.read_csv(wide_dict_path, header=None, names=['feature', 'hash', 'idx'])
        wide_weight_len = 0
        current = 0
        with tf.Session() as sess:
            print('restore last checkpoint', checkpoint.model_checkpoint_path)
            saver.restore(sess, checkpoint.model_checkpoint_path)

            bias = sess.run('wide_layer_variable/bias:0')
            wide_weight = sess.run('wide_layer_variable/w:0')

            print('bias:', str(bias))
            # freeze wide weight for online inference
            wide_weight_len = len(wide_weight)
            print("model size:" + str(wide_weight_len))
            print("freeze wide weight")
            with open(tmp_path, "w+") as file:
                lines = []
                for idx, weight in enumerate(wide_weight):
                    current += 1
                    if abs(weight[0]) < 1e-8:
                        continue
                    cnt += 1
                    lines.append(str(idx) + "," + str(weight[0]) + '\n')

                    if cnt % 10000000 == 0 or current == wide_weight_len:
                        print('current:' + str(cnt))

                    if cnt % 1000000 == 0 or current == wide_weight_len:
                        file.writelines(lines)
                        lines = []
                if lines:
                    file.writelines(lines)
                    lines = []

            # write bias to stat file
            print("write bias to stat file")
            # self.model.meta['modelParam'] = []
            # self.model.meta['modelParam'].append({})
            self.model.meta['modelParam'][0]['bias'] = float(bias)
            self.model.meta['modelParam'][0]['version'] = long(cfg.timestamp)
            self.model.meta['modelParam'][0]['modelSize'] = int(cnt)
            print(self.model.meta)
            with open(stat_path, 'wb') as f:
                json.dump(self.model.meta, f)

        print('end export weight')

        dict_size = dict_df.count()['feature']
        print('wide dict size:', dict_size)
        model_df = pd.read_csv(tmp_path, header=None, names=['idx', "weight"])
        full = pd.merge(dict_df, model_df, how='inner', on='idx')

        # freeze wide weight for online inference
        print("freeze wide weight to ", wide_outpath)
        full.to_csv(wide_outpath, index=False, columns=["hash", "weight"], header=False)

        print("freeze raw wide weight to ", raw_widepath)
        full.to_csv(raw_widepath, index=False, columns=["feature", "weight"], header=False)

        print('final nnz:%f = %d/%d' % (1.0 * cnt / wide_weight_len, cnt, wide_weight_len))

        # save raw wide weight to raw_widepath
        print("%s|Freeze Model Done!" % datetime.now().strftime('%H:%M:%S'))
        os.remove(tmp_path)


class ModelManger(object):
    def __init__(self):
        self.train_path_pattern = None
        self.eval_path_pattern = None
        self.test_path_pattern = None
        self.ckpt_path = None
        self.stat = None
        self.model_path = None
        self.learning_rate = cfg.learn_rate
        self.total_epochs = cfg.max_epoch

    def task_manger(self):
        if cfg.mode == 'train' or cfg.mode == 'train_goon':
            model = ModelMangerBase('train', self.train_path_pattern, self.eval_path_pattern, self.test_path_pattern,
                                    self.stat)
            model.train(n_epochs=self.total_epochs, ckpt_path=self.ckpt_path, tensorboard_dir=cfg.tensorboard_dir)
        elif cfg.mode == 'predict':
            model = ModelMangerBase('predict', self.train_path_pattern, self.eval_path_pattern, self.test_path_pattern,
                                    self.stat)
            model.freezy_wide(self.ckpt_path, cfg.wide_dict_name, cfg.model_dir, cfg.stat_file, cfg.raw_wide_weight)
            # model.freezy_rawwide(self.ckpt_path,cfg.wide_dict_name,cfg.model_dir)
        else:
            raise Exception("mode type is error,must in train/predict")


def main():
    print('%s|Start Train!' % datetime.now().strftime('%H:%M:%S'))
    print('tf version:{}'.format(tf.__version__))
    model_path = cfg.model_dir
    data_path = cfg.train_path
    data_path_val = cfg.valid_path
    data_path_test = cfg.test_path
    ckpt_path = cfg.ckpt_dir
    stat_path = cfg.stat_file
    if cfg.mode == 'predict':
        model = ModelManger()
        model.train_path_pattern = data_path
        model.test_path_pattern = data_path_test
        model.eval_path_pattern = data_path_val
        model.stat = stat_path
        model.ckpt_path = ckpt_path
        model.model_path = model_path
        model.task_manger()
    else:
        model = ModelManger()
        model.train_path_pattern = data_path
        model.test_path_pattern = data_path_test
        model.eval_path_pattern = data_path_val
        model.stat = stat_path
        model.ckpt_path = ckpt_path
        print('ckpt path:', model.ckpt_path)
        model.task_manger()
    os._exit(0)


if __name__ == '__main__':
    main()

