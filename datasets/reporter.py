""" reporter.py

    This is intended a as a simple drop-in evaluation analyzer for tensorflow.

    You can create a `tf_reporter` layer which accepts the following tensors:
    1. image:     [batch x W x H x 3]
    2. predicted: [batch]
    3. expected:  [batch]

    The `tf_reporter` is abstracted within the `EvalReporter` class which can
    be queried for the tf `op` which can be inserted into the evaluation
    session's execution graph.

    TODO: Detailed usage etc.

"""
import scipy
import base64
import numpy as np
import tensorflow as tf
from PIL import Image
from jinja2 import Template
from io import BytesIO 
import json
import os
import simplejson as json
import codecs
import scipy.io as scio
dataNew='datasets/dataNew_hierarchical.mat'
values_mat_labels=[]
values_mat_logits=[]
values_mat_labels_2=[]
values_mat_labels_3=[]
values_mat_labels_4=[]
values_mat_logits_2=[]
values_mat_logits_3=[]
values_mat_logits_4=[]
################################################################################

class EvalReporter(object):
    """ EvalReporter is a class which is constructed with a given batches set of
        tensors for the image, predictions and the expectation.
    """

    failure_histogram = None        # incorrectly predicted classes
    success_histogram = None        # correctly predicted classes
    op                = None        # `py_func` layer which dumps images
    labels=None
    logits=None
    values=None
    values_y=None
    def __init__(self, images=[], predicted=[], expected=[],expected_2=[],expected_3=[],expected_4=[],logit=[],logit_2=[],logit_3=[],logit_4=[]):
        """ constructor for the evaluation reporter.
            TODO: The lengths of the incoming tensors must match!
        """
        self.clear()
        self.op = tf.py_func(self._pyfunc, [images, predicted, expected,expected_2,expected_3,expected_4,logit,logit_2,logit_3,logit_4], tf.float32)


    def _pyfunc(self, images, predicted, expected,expected_2,expected_3,expected_4,logit,logit_2,logit_3,logit_4):
        """ _pyfunc is the python_func's "op" which will accept a list of the
            images, predicted classes and the expectations.  These at this point
            are NOT tensors, but are `numpy.ndarray`'s.
        """
        total = 0
        error = 0
        self.labels.append(expected.tolist())
        self.logits.append(predicted.tolist())
        self.values.append(expected)
        self.values_y.append(predicted)
        values_mat_labels.append(expected)
        values_mat_labels_2.append(expected_2)
        values_mat_labels_3.append(expected_3)
        values_mat_labels_4.append(expected_4)
        values_mat_logits.append(logit)      
        values_mat_logits_2.append(logit_2)
        values_mat_logits_3.append(logit_3)
        values_mat_logits_4.append(logit_4)  
        acc=0
        print(expected.shape)
        for i in range(100):
            if(np.argmax(logit[i])==expected[i]):
                acc+=1
        print(acc/100)
        for i in range(len(images)):
            im = scipy.misc.toimage(images[i])              # ndarray -> PIL.Image
            bs = BytesIO()                       # buffer to hold image
            im.save(bs, format="JPEG")                      # PIL.Image -> JPEG
            b64s = base64.b64encode(bs.getvalue())          # JPEG -> Base64
            total += 1
            #self.labels.append(expected.tolist())
            if expected[i] == predicted[i]:
                if predicted[i] in self.success_histogram:
                    self.success_histogram[predicted[i]].append(b64s)
                else:
                    self.success_histogram[predicted[i]] = [b64s]
            else:
                error += 1
                if predicted[i] in self.failure_histogram:
                    self.failure_histogram[predicted[i]].append([b64s, expected[i]])
                else:
                    self.failure_histogram[predicted[i]] = [[b64s, expected[i]]]

        return np.float32(((total - error)/total) if total > 0 else 0)


    def get_op(self):
        """ get_op returns the tensorflow wrapped `py_func` which will convert the local
            tensors into numpy ndarrays.
        """
        return self.op


    def clear(self):
        """ clear resets the histograms for `this` reporter.
        """
        self.failure_histogram = dict()
        self.success_histogram = dict()
        self.labels=[]
        self.logits=[]
        self.values=[]
        self.values_y=[]
    def write_html_file(self, file_path):
        """ write_html_file dumps the current histograms to the specified `file_path`.
        """
        report = Template("""<!DOCTYPE html>
<html>
<head>
    <title>Prediction Analyzer</title>
</head>
<body>
    <h1>Correct Predictions</h1><br>
    {%  for class, images in me.success_histogram.items() %}
        <h3>Class {{ class }}</h3><br>
        {%  for img in images %}
            <img src="data:image/jpeg;base64,{{img}}" title="class:{{class}}" />
        {%  endfor %}
    {%  endfor %}
    <hr><br><br>

    <h1>Incorrect Predictions</h1><br>
    {%  for class, groups in me.failure_histogram.items() %}
        <h3>Class {{ class }}</h3><br>
        {%  for group in groups %}
            <img src="data:image/jpeg;base64,{{group[0]}}" title="pred:{{class}} exp:{{group[1]}}" />
        {%  endfor %}
    {%  endfor %}
    <hr><br><br>
</body>i
</html>
""").render(me=self)
        with open(file_path, "w") as fout:
            fout.write(report)
        #print(self.values)
        #print(self.values_y)
        scio.savemat(dataNew, {'labels': values_mat_labels,'logits': values_mat_logits,'labels_2': values_mat_labels_2,'logits_2': values_mat_logits_2,'labels_3': values_mat_labels_3,'logits_3': values_mat_logits_3,'labels_4': values_mat_labels_4,'logits_4': values_mat_logits_4})
        #json.dump(self.labels, codecs.open('name_test.json', 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4) ### this saves the array in .json format
        #json.dump(self.logits, codecs.open('name_test_logits_single.json', 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
################################################################################
