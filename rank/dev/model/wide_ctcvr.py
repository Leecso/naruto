#!/usr/bin/env python
# encoding: utf-8
import json
import os
import tensorflow as tf
from conf.config import cfg
from model.model_base import ModelBase


class wide_ctr_cvr(ModelBase):

    def __init__(self, mode, train_data=None, val_data=None, test_data=None, stat=None):
        super(wide_ctr_cvr, self).__init__(mode, train_data, val_data, test_data, stat)

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
        with tf.variable_scope("wide_layer_cvr_variable"):
            self.w_cvr = tf.get_variable(name='w_cvr', shape=[self.wide_feature_size, 1],
                                     initializer=tf.truncated_normal_initializer(stddev=self.init_std))
            self.b_cvr = tf.Variable(self.init_std, trainable=True, name='cvr_bias')
        with tf.variable_scope("ctr_weight_variable"):
            self.ctr_weight = tf.Variable(0.1, trainable=True, name='ctr_weight')
        """
        with tf.variable_scope("shared_variable"):
            self.w_shared = tf.get_variable(name='w_shared', shape=[self.wide_feature_size, 1],
                                             initializer=tf.truncated_normal_initializer(stddev=self.init_std))
            self.ctr_weight = tf.get_variable(name='w_shared_ctr_weight', shape=[self.wide_feature_size, 1],
                                              initializer=tf.truncated_normal_initializer(stddev=self.init_std))
            self.cvr_weight = tf.get_variable(name='w_shared_cvr_weight', shape=[self.wide_feature_size, 1],
                                                          initializer=tf.truncated_normal_initializer(stddev=self.init_std))
            self.b= tf.Variable(self.init_std, trainable=True, name='ctr_bias')
            self.b_cvr = tf.Variable(self.init_std, trainable=True, name='cvr_bias')
        """
        """
        with tf.variable_scope("cvr_weight_variable"):
            self.cvr_weight = tf.Variable(0.1, trainable=True, name='cvr_weight')
        """
        schema_path = self.stat.replace('param', 'train.schema')
        print ('stat,{0},scheam,{1}'.format(self.stat, schema_path))
        with open(schema_path, 'rb') as f:
            self.schema = json.load(f)

    def build_loss(self):
        labels = tf.cast(self.label, tf.float32, name='true_label')
        labels = tf.slice(labels, [0], tf.shape(self.logits))
        order_labels = tf.cast(self.order_label, tf.float32, name='true_label')
        order_labels = tf.slice(order_labels, [0], tf.shape(self.logits))
        losses1 = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=self.logits) #ctr loss
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        losses2 = bce(order_labels, self.ctr*self.cvr) #ctcvrloss
        losses3 = tf.nn.sigmoid_cross_entropy_with_logits(labels=order_labels, logits=self.logits_cvr) #cvr loss
        #losses2 = tf.nn.sigmoid_cross_entropy(labels=self.order_label, logits=self.ctr*self.cvr)
        losses = losses1 + losses2
        self.losses1 = tf.reduce_mean(losses1, name='losses1')
        self.losses2 = tf.reduce_mean(losses2, name='losses2')
        loss = tf.reduce_mean(losses, name='loss') + tf.reduce_sum(losses3*labels,name='cvr_loss')/tf.cast(tf.count_nonzero(labels),tf.float32)
        #self.auc = tf.metrics.auc(labels, self.prob)
        return loss
    """
    def build_network(self):
        with tf.variable_scope("shared_layer"):
            u_emb = self.features
            indicies = tf.where(tf.not_equal(u_emb, -1))
            w_sp = tf.SparseTensor(values=tf.gather_nd(u_emb, indicies), indices=indicies,
                                   dense_shape=[self.batch_size, self.wide_feature_size])
            w_shared_param = tf.nn.embedding_lookup_sparse(self.w_shared, w_sp, None)
            ctr_weight_param = tf.nn.embedding_lookup_sparse(self.ctr_weight, w_sp, None)
            cvr_weight_param = tf.nn.embedding_lookup_sparse(self.cvr_weight, w_sp, None)
            self.w = w_shared_param * ctr_weight_param
            self.w_cvr = w_shared_param * cvr_weight_param
            print ("w参数形式")
            print (self.w)
            print ("w_cvr参数形式")
            print (self.w_cvr)
            print ("参数")
            print (w_shared_param)
            print (ctr_weight_param)
            print (cvr_weight_param)
        with tf.variable_scope("wide_layer"):
            add_sum = tf.reduce_sum(self.w,1)
            self.add_sum = tf.reshape(add_sum, [-1], name='wide_sum_ctr')
            self.logits_wide = self.add_sum + self.b
        with tf.variable_scope("wide_layer"):
            add_sum = tf.reduce_sum(self.w_cvr,1)
            self.add_sum_cvr = tf.reshape(add_sum, [-1], name='wide_sum_cvr')
            self.logits_wide_cvr = self.add_sum_cvr + self.b_cvr
        with tf.variable_scope("output_layer"):
            self.logits = self.logits_wide
            self.logits_cvr = self.logits_wide_cvr

            self.ctr = tf.sigmoid(self.logits)
            self.cvr = tf.sigmoid(self.logits_cvr)

            self.y = tf.identity(self.logits, name='predict_value')
            self.y_cvr = tf.identity(self.logits_cvr, name='predict_value_cvr')
            self.probs = []
        return self.logits
    """
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
        with tf.variable_scope("wide_layer_cvr"):
            u_emb = self.features
            if cfg.train_type.lower() == 'dense':
                indicies = tf.where(tf.not_equal(u_emb, -1))
                w_sp = tf.SparseTensor(values=tf.gather_nd(u_emb, indicies), indices=indicies,
                                      dense_shape=[self.batch_size, self.wide_feature_size])
                add_sum = tf.nn.embedding_lookup_sparse(self.w_cvr, w_sp, None, combiner='sum', name="SparseSumCvr")
            else:
                indicies = tf.where(tf.not_equal(u_emb, 0))
                w_sp = tf.SparseTensor(values=tf.gather_nd(u_emb, indicies), indices=indicies,
                                       dense_shape=[self.batch_size, self.wide_feature_size])
                add_sum = tf.nn.safe_embedding_lookup_sparse(self.w_cvr, w_sp, sparse_weights=None, combiner='sum',
                                                             default_id=0, name='SparseSumCvr')
            self.add_sum_cvr = tf.reshape(add_sum, [-1], name='wide_sum_cvr')
            self.logits_wide_cvr = self.add_sum_cvr + self.b_cvr


        with tf.variable_scope("output_layer"):
            self.logits = self.logits_wide
            self.logits_cvr = self.logits_wide_cvr + self.ctr_weight*self.logits_wide

            self.ctr = tf.sigmoid(self.logits)
            self.cvr = tf.sigmoid(self.logits_cvr)

            self.y = tf.identity(self.logits, name='predict_value')
            self.y_cvr = tf.identity(self.logits_cvr, name='predict_value_cvr')
            self.probs = []

            for i in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
                for j in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
                    self.probs.append ([tf.constant(i),tf.constant(j),tf.identity(tf.pow(self.ctr,i)*tf.pow(self.cvr,j), name='prob')])

        return self.logits

    def build_summary(self):
        with tf.name_scope("summaries"):
            tf.summary.scalar('loss', self.loss)
            #tf.summary.scalar('auc', self.auc[1])
            tf.summary.scalar('logits', tf.reduce_mean(self.logits))
            tf.summary.scalar('logits_cvr', tf.reduce_mean(self.logits_cvr))
            tf.summary.scalar('ctr_weight', self.ctr_weight)
            #tf.summary.scalar('cvr_weight', self.cvr_weight)

            with tf.name_scope('wide_summaries'):
                tf.summary.scalar('wide_bias', self.b)
                tf.summary.scalar('wide_bias_cvr', self.b_cvr)
                tf.summary.scalar('wide_logits', tf.reduce_mean(self.logits_wide))
                tf.summary.scalar('wide_logits_cvr', tf.reduce_mean(self.logits_wide_cvr))
                tf.summary.histogram('wide_histogram', self.w)
                tf.summary.histogram('wide_histogram_cvr', self.w_cvr)
                tf.summary.histogram('wide_logits_histogram', self.logits_wide)
                tf.summary.histogram('wide_logits_histogram_cvr', self.logits_wide_cvr)
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

