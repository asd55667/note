>tf.train.feature
```
def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
```
>tf.train.Example
+ tf.gfile.GFile
```
with tf.gfile.GFile(path, 'rb') as file:
    data = file.read()
```
+ io.BytesIO
+ extract feature
> generate tfrecord
+ tf.python_io.TFRecordWriter
+ glob.glob
+ tf.Example.SerializeToString
+ writer.close

## Image date to tfrecord
```
import glob
import os
import io

import tensorflow as tf 
from PIL import Image

flags = tf.app.flags

flags.DEFINE_string('img_path', None, 'Absolute path of image ')  #define arg params
flags.DEFINE_string('output_path', None, 'Absolute path of record generated ')

FLAGS = flags.FLAGS  

def int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))



def create_examples(path):
	with tf.gfile.GFile(path, 'rb') as gfile:
		encode_img = gfile.read()
	img_io = io.BytesIO(encode_img)
	img = Image.open(img_io)
	width, height = img.size 
	label = path.split('.')[0].split('/')[-1]  #path (xxx/xxx/xxx/.jpg) 
	tf_examples = tf.train.Example(
		features=tf.train.Features(feature={
				'img': bytes_feature(encode_img),
				'height': int64_feature(height),
				'width': int64_feature(width),
				'label': bytes_feature(label.encode()),
				'format': bytes_feature(b'jpg')
				}))
	return tf_examples


def generate_record(path, out_path):
	writer = tf.python_io.TFRecordWriter(out_path)
	for img_path in glob.glob(path+'*.jpg'):    
		tf_examples = create_examples(img_path)
		writer.write(tf_examples.SerializeToString())
	writer.close()

def main(_):
	input_path = FLAGS.path
	output_path = FLAGS.output
	generate_record(input_path, output_path)		

if __name__ == '__main__':
	tf.app.run()	

```
```
$python tfrecord.py --img_path /media/wcw/Intel118G/Downloads/test/  \
                   --output_path /media/wcw/Intel118G/Downloads/whale_test  .record

```
>csv to tfrecord  MNIST的csv转为tfrecord由80M变到270M haochun
```
import pandas as pd 
import numpy as np 
import tensorflow as tf 
flags = tf.app.flags
flags.DEFINE_string('csv_path', None, 'Absolute path of csv file')
flags.DEFINE_string('output_path', None, 'Absolute path of tfrecord generated')
FLAGS = flags.FLAGS

def int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def generate_tfrecord(path, output_path):
	train_df = pd.read_csv(path)
	label = train_df['label'].values
	train = train_df.drop('label', axis=1).values
    
	writer = tf.python_io.TFRecordWriter(output_path)
    
	for i in range(train_df.shape[0]):
		digits_raw = train[i].tostring()
		example = tf.train.Example(features=tf.train.Features(feature={
			'digits': bytes_feature(digits_raw),
			'label': int64_feature(label[i])
			}))
		writer.write(record=example.SerializeToString())
	writer.close()

def main():
	csv_path = FLAGS.csv_path
	output_path = FLAGS.output_path
	generate_tfrecord(csv_path, output_path)

if __name__ == '__main__':
	main()                   
```    