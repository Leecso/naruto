#!/usr/bin/env python
# encoding: utf-8
import json
import os
import tensorflow as tf
from conf.config import cfg
from model.model_base import ModelBase


class wide_ctr(ModelBase):

    def __init__(self, mode, train_data=None, val_data=None, test_data=None, stat=None):
        super(wide_ctr, self).__init__(mode, train_data, val_data, test_data, stat)

    def build_learnable_params(self):
        with open(self.stat, "rb") as f:
            self.meta = json.load(f)

        self.train_size = self.meta["statInfo"]['trainSetSize']
        self.val_size = self.meta['statInfo']['validationSetSize']
        self.test_size = self.meta['statInfo']['testSetSize']
        self.wide_feature_size = self.meta['dictInfo'][0]['dictSize']
        print ('wide_feature_size:', self.wide_feature_size)
        with tf.variable_scope("wide_layer_variable"):
            self.w = tf.get_variable(name='w', shape=[self.wide_feature_size, 1],
                                     initializer=tf.truncated_normal_initializer(stddev=self.init_std))
            self.b = tf.Variable(self.init_std, trainable=True, name='bias')

        schema_path = self.stat.replace('param', 'train.schema')
        print ('stat,{0},scheam,{1}'.format(self.stat, schema_path))
        with open(schema_path, 'rb') as f:
            self.schema = json.load(f)

    def build_loss(self):
        labels = tf.cast(self.label, tf.float32, name='true_label')
        labels = tf.slice(labels, [0], tf.shape(self.logits))
        losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=self.logits)
        loss = tf.reduce_mean(losses, name='loss')
        self.auc = tf.metrics.auc(labels, self.prob)
        return loss

    def build_network(self):
        with tf.variable_scope("wide_layer"):
            u_emb = self.features
            if cfg.train_type.lower() == 'dense':
                indicies = tf.where(tf.not_equal(u_emb, -1))
                w_sp = tf.SparseTensor(values=tf.gather_nd(u_emb, indicies), indices=indicies,
                                       dense_shape=[self.batch_size, self.wide_feature_size])
                add_sum = tf.nn.embedding_lookup_sparse(self.w, w_sp, None, combiner='sum', name="SparseSum")
            else:
                indicies = tf.where(tf.not_equal(u_emb, 0))
                w_sp = tf.SparseTensor(values=tf.gather_nd(u_emb, indicies), indices=indicies,
                                       dense_shape=[self.batch_size, self.wide_feature_size])
                add_sum = tf.nn.safe_embedding_lookup_sparse(self.w, w_sp, sparse_weights=None, combiner='sum',
                                                             default_id=0, name='SparseSum')
            self.add_sum = tf.reshape(add_sum, [-1], name='wide_sum')
            self.logits_wide = self.add_sum + self.b
            self.ctr = tf.sigmoid(self.logits_wide)

        with tf.variable_scope("output_layer"):
            self.logits = self.logits_wide
            print ('logits_shape', self.logits.get_shape())
            self.y = tf.identity(self.logits, name='predict_value')
            self.prob = tf.identity(tf.nn.sigmoid(self.logits), name='prob')
        return self.logits

    def build_summary(self):
        with tf.name_scope("summaries"):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('auc', self.auc[1])
            tf.summary.scalar('logits', tf.reduce_mean(self.logits))
            tf.summary.histogram('logits_hist', self.logits)

            with tf.name_scope('wide_summaries'):
                tf.summary.scalar('wide_bias', self.b)
                tf.summary.scalar('wide_logits', tf.reduce_mean(self.logits_wide))
                tf.summary.histogram('wide_histogram', self.w)
                tf.summary.histogram('wide_logits_histogram', self.logits_wide)
            merged = tf.summary.merge_all()
        return merged

    def build_graph(self):
        self.graph = tf.get_default_graph()
        self.graph.seed = self.seed
        self.global_step = tf.train.get_or_create_global_step()
        self.build_learnable_params()
        self.build_input()
        self.logits = self.build_network()
        if self.train_mode != 'predict':
            self.loss = self.build_loss()
            self.build_debugger()
            self.build_count()
            self.merged = self.build_summary()
            self.optimizer = self.bulid_optimize(self.lr)
            self.trainer = self.optimizer.minimize(self.loss, global_step=self.global_step)
            self.init_all_vars = tf.group(tf.global_variables_initializer(), tf.initialize_local_variables())
            self.saver = tf.train.Saver(max_to_keep=2)
