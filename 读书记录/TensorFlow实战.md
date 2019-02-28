# 7 循环网络
## 7.1 Word2Vec
```向量空间模型（Vector Space Model)``` 
能学习到单词向量的相似性  （男 -> 女）  // （爸 -> 妈)
### SKip-Gram
从原句推测目标字词
序列 ··· a · b · c ···  (b -> a), (c -> a)
### CBOW(Continuous Bag of Words)
从目标字词推出原句


```预测模型 Neural Probabilistic Language Model 通常用最大似然```

```
def build_dataset(words):
	count = [['UNK', -1]]
	count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
	dictionary = {}
	for word, _ in count:
		dictionary[word] = len(dictionary)
	data = []
	unk_count = 0
	for word in words:
		if word in dictionary[word]
			index = dictionary[word]
		else:
			index = 0
			unk_count += 1
		data.append(index)
	count[0][1] = unk_count
	reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
	return data, count, dictionary, reverse_dictionary
data, count, dictionary ,reverse_dictionary = build_dataset(words)

del words


data_index = 0
def generate_batch(batch_size, num_skips, skip_window):
	global data_index
	assert batch_size % num_skips == 0
	assert num_skips <= 2 * skip_window
	batch = np.ndarray(shape=(batch_size), dtype=np.int32)
	labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
	span = 2 * skip_window + 1
	buffer = collections.deque(maxlen=sapn)  #bidirection queue

	for _ in range(span):
		buffer.append(data[data_index])
		data_index = (data_index + 1) % len(data)
	for i in range(batch_size // num_skips):
		target = skip_window
		targets_to_avoid = [skip_window]
		for j in range(num_skips):
			while target in targets_to_avoid:
				target = random.randint(0, span-1)
			targets_to_avoid.append(target)
			batch[i * num_skips + j] = buffer[skip_window]
			labels[i * num_skips + j, 0] = buffer[target]
		buffer.append(data[data_index])
		data_index = (data_index + 1) % len(data)
	return batch, labels	

graph = tf.Graph()
with graph.as_default():
	train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
	train_labels = tf.placeholder(tf.int32, shape=[generate_batch, 1])
	valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

	with tf.device('/cpu:0'):
		embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
	embed = tf.nn.embedding_lookup(embeddings, train_inputs)

	nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
		stddev=1.0/math.sqrt(embedding_size)))
	nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, 
							biases=nce_biases,
							labels=train_labels,
							num_sampled=num_sampled,
							num_classes=vocabulary_size))

optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
normalized_embeddings = embeddings / norm 
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

init = tf.global_variables_initializer()

with tf.Session(graph=graph) as sess:
	init.run()
	average_loss = 0
	for step in range(num_steps):
		batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
		feed_dict = {train_nputs: batch_inputs, train_labels: batch_labels}

		_, loss_val = sess.run([optimizer, loss], feed_dict=feed_dict)
		average_loss += loss_val

		if step % 10000 == 0:
			sim = similarity.eval()
			for i in range(valid_size):
				valid_word = reverse_dictionary[valid_examples[i]]
				top_k = 8
				nearest = (-sim[i, :]).argsort()[1:top_k+1]
				log_str = 'Nearest to %s:'%valid_word
				for k in range(top_k):
					close_word = reverse_dictionary[nearest[k]]
					log_str = '%s %s,' %(log_str, close_word)
				print(log_str)
final_embeddings = normalized_embeddings.eval()

def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
	assert low_dim_embs: shape[0] >= len(labels), 'More labels than embeddings'
	plt.figure(figsize=(18, 18))
	for i, label in enumerate(labels):
		x, y = low_dim_embs[i,:]
		plt.scatter(x, y)
		plt.annotate(label, xy=(x, y), 
			xytext=(5,2),
			textcoords='offset points',
			ha='right',
			va='bottom')


from sklearn.manifold import Tsne 
import matplotlib.pyplot as plt
tsne = Tsne(perplexity=30, n_components=2, init='pca', n_iter=5000)
plot_only = 100
low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
labels = [reverse_dictionary[i] for i in range(plot_only)]
plot_with_labels(low_dim_embs, labels)
```
## 7.2 LSTM

```
class PTBInput(object):
	def __init__(self, config, data, name=None):
		self.batch_size = batch_size = config.batch_size
		self.num_steps = num_steps = config.num_steps
		self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
		self.input_data, self.targets = reader.ptb_producer(data, 
															batch_size, 
															num_steps,
															name=name)

class PTBModule(object):
	def __init__(self, is_training, config, input_):
		self._input = input_

		batch_size = input_.batch_size
		num_steps = input_.num_steps
		size = config.hidden_size
		vocab_size = config.vocab_size

		def lstm_cell():
			return tf.contrib.nn.LSTMCell(size, forget_bias=0., state_is_tuple=Ture)
		attn_cell = lstm_cell
		if is_training and config.keep_prob < 1:
			def attn_cell():
				return tf.contrib.rnn.DropoutWrapper(lstm_cell(), 
													output_keep_prob=config.keep_prob)
			cell = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(config.num_layers)],
												state_is_tuple=True)

		self._initial_state = cell.zero_state(batch_size, tf.float32)

		with tf.device('/cpu:0'):
			embedding = tf.get_variable('embedding',	
										[vocab_size, size],
										dtype=tf.float32)
			inputs = tf.nn.embedding_lookup(embedding, input_.input_data)
		if is_training and config.keep_prob < 1:
			inputs = tf.nn.dropout(inputs, config.keep_prob)

		outputs = []
		state = self._initial_state

		with tf.variable_scope('RNN'):
			for time_step in range(num_steps):
				if time_step > 0:
					tf.get_variable_scope().reuse_variables()
				(cell_output, state) = cell(inputs[:, time_step, :], state)
				outputs.append(cell_output)

		output = tf.reshape(tf.concat(outputs, 1), [-1, size])
		softmax_w = tf.get_variable('softmax_w', 
									[size, vocab_size], 
									dtype=tf.float32)
		softmax_b = tf.get_variable('softmax_b', 
									[vocab_size], 
									dtype=tf.float32)
		logits = tf.matmul(output, softmax_w) + softmax_b
		loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
			[logits],
			[tf.reshape(input_.targets, [-1])],
			[tf.ones([batch_size * num_steps], dtype=tf.float32)])
		self.cost_ = cost = tf.reduce_sum(loss) / batch_size
		self._final_state = state

		if not is_training:
			return 

		self._lr = tf.Variable(0., trainable=False)
		tvars = tf.trainable_variables()
		grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
										config.max_grad_norm)
		optimizer = tf.train.GradientDescentOptimizer(self._lr)
		self._train_op = optimizer.apply_gradients(zip(grads, tvars),
			global_step=tf.contrib.framework.get_or_create_global_step())

		self._new_lr = tf.placeholder(tf.float32,
									shape=[], name='new_learning_rate')
		self._lr_update = tf.assign(self._lr, self._new_lr)

		def assign_lr(self, session, lr_value):
			session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

		@property
		def input(self):
			return self._input
		
		@property
		def initial_state(self):
			return self._initial_state
		
		@property
		def cost(self):
			return self._cost
		
		@property
		def final_state(self):
			return self._final_state
		
		@property
		def lr(self):
			return self._lr
		
		@property
		def train_op(self):
			return self._train_op
		


class SmallConfig(object):
	init_scale = 0.1
	learning_rate = 1.
	max_grad_norm = 5
	num_layers = 2
	num_steps = 20
	hidden_size = 200
	max_epoch = 4
	max_max_epoch = 13
	keep_prob = 1.
	lr_decay = 0.5
	batch_size = 20
	vocab_size = 10000

def run_epoch(session, model, eval_op=None, verbose=False):
	start_time = time.time()
	costs = 0.0 
	iters = 0
	state = session.run(model.initial_state)

	fetches = {'cost': model.cost, 'final_state': model.final_state}
	if eval_op is not None:
		fetches['eval_op'] = eval_op

	for step in range(model.input.epoch_size):
		feed_dict = {}
		for i, (c, h) in enumerate(model.initial_state):
			feed_dict[c] = state[i].c
			feed_dict[h] = state[i].h

		vals = session.run(fetches, feed_dict)
		cost = vals['cost']
		state = vals['final_state']

		costs += cost
		iters += model.input.num_steps

		if verbose and step % (model.input.epoch_size // 10) == 10:
			print('%.3f perplexity: %.3f speed: %.0f wps'%
				(step * 1. / model.input.epoch_size, np.exp(costs / iters),
				iters * model.input.batch_size / (time.time() - start_time)))

	return np.exp(costs / iters)

raw_data = reader.ptb_raw_data('simple-examples/data/')	
train_data, val_data, test_data, _ = raw_data

config = SmallConfig()
eval_config = SmallConfig()
eval_config.batch_size = 1
eval_config.num_steps = 1

with tf.Graph.as_default():
	initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

	with tf.name_scope('Train'):
		train_input = PTBInput(config=config, 
							data=train_data,
							name='TrainInput',)
		with tf.variable_scope('Model', reuse=None, initializer=initializer):
			m = PTBModule(is_training=True,
			 			config=config, 
			 			input_=train_input)
	with tf.name_scope('Valid'):
		valid_input = PTBInput(config=config,
								data=val_data,
								name='ValidInput')
		with tf.variable_scope('Model', reuse=True, initializer=initializer):
			mvalid = PTBModule(is_training=False,
								config=config,
								input_=valid_input)

	with tf.name_scope('Test'):
		test_input = PTBInput(config=eval_config,
								data=test_data,
								name='TestInput')
		with tf.variable_scope('Model', reuse=True, initializer=initializer):
			mtest = PTBModule(is_training=False,
							config=eval_config,
							input_=test_input)


sv = tf.train.Supervisor()
with sv.managed_session() as sess:
	for i in range(config.max_max_epoch):
		lr_decay = config.lr_decay ** max(i+1 - config.max_epoch, 0.0)
		m.assign_lr(sess, config.learning_rate * lr_decay)

		print('Epoch: %d Learning rate: %.3f'%(i+1, sess.run(m.lr)))
		train_perplexity = run_epoch(sess, m, eval_op=m.train_op, verbose=True)
		print('Epoch: %d Train Perplexity: %.3f'%(i+1, train_perplexity))
		valid_perplexity = run_epoch(sess, mvalid)
		print('Epoch: %d Valid Perplexity: %.3f'%(i+1, valid_perplexity))

	test_perplexity = run_epoch(sess, mtest)
	print('Test Perplexity: %.3f'% test_perplexity)
```
## 7.3 Bidirectional LSTM


```
x = tf.placeholder('float', [None, n_steps, n_input])
y = tf.placeholder('float', [None, n_classes])

weights = tf.Variable(tf.random_normal([2*n_hidden, n_classes]))
biases = tf.Variable(tf.random_normal([n_classes]))

def BiRNN(x, weights,biases):
	x = tf.transpose(x, [1, 0, 2])
	x = tf.reshape(x, [-1, n_input])
	x = tf.split(x, n_steps)

	lstm_fw_cell = tf.contrib.rnn.LSTMCell(n_hidden, forget_bias=1.0)
	lstm_bw_cell = tf.contrib.rnn.LSTMCell(n_hidden, forget_bias=1.)

	outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)
	return tf.matmul(outputs[-1], weights) + biases

pred = BiRNN(x, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

cost_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()	

with tf.Session() as sess:
	sess.run(init)
	step = 1
	while step * batch_size < max_samples:
		batch_x, batch_y = mnist.train.next_batch(batch_size)
		batch_x = batch_x.reshape((batch_size, n_steps, n_input))
		sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
		if step % displacy_step == 0:
			acc, loss = sess.run([accuracy, cost], feed_dict={x: batch_x, y: batch_y})
			print('Iter' + str(step*batch_size) + ', Minibatch loss= '+ '{:.6f}'.format(loss) + ', Training Accuracy= ' + '{:.5f}'.format(acc))
			step += 1
		print('Optimization Finished!')

	test_len = 10000
	test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
	test_label = mnist.test.labels[:test_len]
	print('Testing Accuracy: ', sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
```
