import tensorflow as tf
import numpy as np
import os


class BaseNN():

    def _initialize(self):
        pass

    def _BuildGraph(self, inputs):
        pass

    def _loss(self, graphInputs, graphOutputs):
        pass

    def _makeTrainSteps(self, tf_learning_rate, losses):
        """
        generate a list of training step. Each item in the list corresponding to one kind of loss in loss list.
        This allows to update different losses at the same time.
        tf_learning_rate: 
        losses: list of losses
        return: list of training steps
        """
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_steps = []
        with tf.control_dependencies(update_ops):
            for il, loss in enumerate(losses):
                one_step = tf.train.AdamOptimizer(tf_learning_rate).minimize(loss)
                train_steps.append(one_step)
        return train_steps

    def _buildConvBnRelu(self, inputs, filters, isTraining, strides=1, kernel_size=[3, 3], axis=3):
        """
        build a conv+bn+relu block.
        inputs: input 2d tensor to apply conv to.
        filters: number of filters to apply.
        isTraining: flag for 'is training'
        """
        conv1 = tf.layers.conv2d(
            inputs=inputs,
            filters=filters,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            name='conv'
        )
        bn1 = tf.layers.batch_normalization(conv1, axis=axis, training=isTraining, name='bn')
        relu1 = tf.nn.relu(bn1, name='relu')
        return relu1

    def _buildDeconvBnReluCont(self, inputs, contlayer, filters, isTraining, axis=3):
        """
        :param inputs: input tensor
        :param contlayer: layer to concatenate at the end of this block, if applicable
        :param filters:
        :param isTraining:
        :param axis:
        :return:
        """
        upsample = tf.layers.conv2d_transpose(
            inputs=inputs,
            filters=filters,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            kernel_size=[3, 3],
            strides=2,
            padding='same',
            name='deconv'
        )
        bn1 = tf.layers.batch_normalization(upsample, axis=axis, training=isTraining, name='bn')
        relu1 = tf.nn.relu(bn1, name='relu')
        if contlayer == None:
            return relu1
        cont = tf.concat([relu1, contlayer], 3)
        return cont

    def _buildConvSigmoid(self, inputs, filters):
        conv1 = tf.layers.conv2d(
            inputs=inputs,
            filters=filters,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            kernel_size=[3, 3],
            padding='same',
            name='conv1'
        )
        sig1 = tf.nn.sigmoid(conv1, name='sig')
        return sig1


