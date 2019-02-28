from BaseNN import *
from util import load_img_cond

class RelightBase(BaseNN):

    def __init__(self, featureSize=(64, 128, 256, 512), expandSize=(8, 32, 128, 128), imgSize=(128, 128), inputNum=5):
        """
        :param featureSize: list of feature sizes for unet encoder, from short to long.
                            decoder will use the reverse of this list.
        :param expandSize: list of layer size for FC layers to expand the light direction.
        :param imgSize:
        :param inputNum:
        """
        self.expandSize = expandSize  # fc expansion for light direction vector
        self.featureSize = featureSize
        self.imgSize = imgSize
        self.inputNum = inputNum

    def _initialize(self):
        """
        build input placeholders.
        :return:
        """
        inputNum = self.inputNum
        inputImgs = tf.placeholder(tf.int8, shape=(None, self.imgSize[0], self.imgSize[1], inputNum*3))
        inputDirs = tf.placeholder(tf.float32, shape=(None, 1, 1, inputNum * 3))
        relightDirs = tf.placeholder(tf.float32, shape=(None, 1, 1, 3))
        targets = tf.placeholder(tf.int8, shape=(None, self.imgSize[0], self.imgSize[1], 3))
        isTraining = tf.placeholder(tf.bool)
        return (inputImgs, inputDirs, relightDirs, isTraining, targets)

    def _relightNet(self, inputs):
        """
        inputs: (relightInput, relightDirs, isTraining)
        """

        relightInput, relightDirs, isTraining = inputs
        curLayer = relightInput
        layers = []
        layers.append(curLayer)

        # build downsampling encoder CNN
        print("build downsample encoder.")
        for i, feature in enumerate(self.featureSize):
            with tf.variable_scope("downblock%d" % i):
                curLayerOut = self._buildConvBnRelu(curLayer, feature, isTraining, strides=2)
                layers.append(curLayerOut)
                curLayer = curLayerOut
                print("downsample layer %d: " % i, curLayer)
        print("complete building downsample encoder.")

        innerLayer = layers.pop()  # the narrowest downsampled layer

        # encode the relight direction by FCN
        print("expanding light position")
        with tf.variable_scope('expandDirs'):
            curDir = tf.reshape(relightDirs, [-1, 3])  # all positions are 3 dimensional.
            for ii, feature in enumerate(self.expandSize):

                if ii == 0:
                    preF = 3  # previous feature vector size, initially is 3 (light position)
                else:
                    preF = self.expandSize[ii - 1]

                with tf.variable_scope("FCN%d" % ii):
                    # initialize weights using truncated normal, shape=[previous feature, next feature]
                    FCNW = tf.Variable(tf.truncated_normal([preF, feature], stddev=1.0 / np.sqrt(preF)), name='weights')
                    FCNB = tf.Variable(tf.zeros([feature]), name='biases')
                    curOutDir = tf.tanh(tf.matmul(curDir, FCNW) + FCNB)
                    curDir = curOutDir

            curDir = tf.reshape(curOutDir, [-1, 1, 1, self.expandSize[-1]])  # curOutDir is a (?, 128) tensor
            print("Expanded light position: ", curDir)

        # times each value in the 128 tensor to a 8x8 feature map filled with 1, to
        # create a (?, 8, 8, 128) feature map to concatenate.
        centerDir = curDir * tf.constant(1.0, tf.float32, shape=(1, self.imgSize[0] / (2 ** len(self.featureSize)),
                                                                 self.imgSize[0] / (2 ** len(self.featureSize)), 1))
        # join expanded conditional input with downsampled feature map, use it as input for decoder
        decoderIn = tf.concat([innerLayer, centerDir], 3)  # concat (?, 8, 8, 512) with (?, 8, 8, 128) along axis 3
        # output dimension: (?, 8, 8, 640)

        print("build center conv")
        with tf.variable_scope("convblock"):
            conv = self._buildConvBnRelu(decoderIn, self.featureSize[-1], isTraining)
            curLayer = tf.concat([conv, decoderIn], 3)

        layers.reverse()
        invFeature = []
        for i in range(0, len(self.featureSize)):
            invFeature.append(self.featureSize[i])
        invFeature.reverse()

        # build decoder CNN
        print("build upsample CNN")
        for i, feature in enumerate(invFeature):
            with tf.variable_scope("upblock%d" % i):
                curLayerOut = self._buildDeconvBnReluCont(curLayer, layers[i], feature, isTraining)
                curLayer = curLayerOut
                print("upsample layer %d: " % i, curLayer)

        with tf.variable_scope("out"):
            curLayer = self._buildConvBnRelu(curLayerOut, self.featureSize[0], isTraining)
            output = self._buildConvSigmoid(curLayer, 3)

        return output

    def _loss(self, graphInputs, graphOutputs):
        """
        calculate loss.
        :param graphInputs: input tensors from all placeholders.
        :param graphOutputs: output from the model (generated image tensor)
        :return: loss
        """
        inputImgs, inputDirs, relightDirs, isTraining, targets = graphInputs
        out = tf.cast(targets, tf.float32) / 255.0
        loss = tf.reduce_mean(tf.nn.l2_loss(graphOutputs - out)) # mse

        return loss

    def _BuildGraph(self, inputs):
        """
        Build whole graph
        :param inputs: inputs placeholders from self._initialize()
        :return:
        """
        inputImgs, inputDirs, relightDirs, isTraining, _ = inputs
        inImgs = tf.cast(inputImgs, tf.float32) / 255.0 * 2.0 - 1.0 # each pixel is between (-1, 1)
        inDirs = inputDirs * tf.constant(1.0, tf.float32, (1, self.imgSize[0], self.imgSize[1], 3 * self.inputNum))
        relightIn = tf.concat([inImgs, inDirs], axis=3)

        output = self._relightNet([relightIn, relightDirs, isTraining])
        return output

    def _get_train_ops(self, loss, learning_rate, beta1):
        """

        :param loss:
        :param learning_rate:
        :param beta:
        :return:
        """
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(loss)

        return train_opt



    def train(self, img_samples, n_epoch, lr, beta1, batch_iter_args, checkpoint_dir = 'checkpoint'):

        model_name = 'cUnet'

        graphInputs = self._initialize()
        inputImgs, inputDirs, relightDirs, isTraining, targets = graphInputs
        graphOutputs = self._BuildGraph(graphInputs)
        loss = self._loss(graphInputs, graphOutputs)
        train_opt = self._get_train_ops(loss, lr, beta1)

        steps = 0
        losses = []

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch_i in range(n_epoch):
                batch_iter = load_img_cond(*batch_iter_args)
                for batch_img, batch_cond in batch_iter:
                    batch_size = len(batch_img)

                    sess.run(train_opt, feed_dict={inputImgs: img_samples, inputDirs: batch_cond, isTraining:True,
                                                   targets: batch_img})

                    if steps % 1 == 0:
                        train_loss = loss.eval({inputImgs: img_samples, inputDirs: batch_cond, isTraining: False,
                                                targets: batch_img})
                        losses.append(train_loss)
                        print("Epoch {}/{}...".format(epoch_i+1, n_epoch),
                              "Loss: {:.4f}".format(train_loss))

            saver = tf.train.Saver()
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver.save(sess, os.path.join(checkpoint_dir, model_name+'.model'), global_step=2, write_meta_graph=False)



if __name__ == '__main__':


    import matplotlib.image as mpimg

    img_dir = './train/toy_multi_512'
    cond_file = './train/toy_multi_512.csv'
    sample_dir = './train/samples'
    sample_list = os.listdir(sample_dir)
    samples = np.empty((5, 512, 512, 4))
    for i, img in enumerate(sample_list):
        path = os.path.join(sample_dir, img)
        samples[i] = mpimg.imread(path)

    testNN = RelightBase()




    pass
