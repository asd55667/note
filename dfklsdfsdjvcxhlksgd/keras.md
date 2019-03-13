# 自定义Callback
```
class CustomCallback(keras.callbacks.Callback):
```
监控,设置训练中的参数, 例如,遇NaN终止, EarlyStopping, lrSchedule, ReducingOnPlateau,
method
> on_train_begin(self, log={})

可在训练开始期间自定义新参数, 
继承父类的params有'verbose', 'nb_epoch', 'batch_size', 'metrics'等, 均以字典形式存储,  

self.model.validation_data 属性在on_batch_begin后才能调用,
其中的数据有train_x, train_y, sample_weight
> on_epoch_begin(self, epoch, logs={})

在此方法下, log始终为空, 随着训练进度的推进, 依次增加epoch轮数
> on_epoch_end(self, epoch, logs={})

可观察每一轮的logs, 如'acc', 'loss', 'val_loss'等参数

on_batch* 与on_epoch*类似, 可观察每个batch的参数

```
    def on_train_begin(self, logs={}):
        self.auc = []
        self.losses = []
    
    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(loss.get('loss'))
        y_pred = self.model.predict(self.model.validation_data[0])
        self.auc.append(sklearn.metrics.roc_auc_score(self.model.validaiton_data[1], y_pred))
```



```
    def on_epoch_end(self, batch, logs={}):

        self.scores = {
            'recall_score': [],
            'precision_score': [],
            'f1_score': []
        }

        for batch_index in range(self.validation_steps):
            features, y_true = next(self.validation_generator)            
            y_pred = np.asarray(self.model.predict(features))
            y_pred = y_pred.round().astype(int) 
            self.scores['recall_score'].append(recall_score(y_true[:,0], y_pred[:,0]))
            self.scores['precision_score'].append(precision_score(y_true[:,0], y_pred[:,0]))
            self.scores['f1_score'].append(f1_score(y_true[:,0], y_pred[:,0]))
```

```
class NBatchLogger(Callback):
    """
    A Logger that log average performance per `display` steps.
    """
    def __init__(self, display):
        self.step = 0
        self.display = display
        self.metric_cache = {}

    def on_batch_end(self, batch, logs={}):
        self.step += 1
        for k in self.params['metrics']:
            if k in logs:
                self.metric_cache[k] = self.metric_cache.get(k, 0) + logs[k]
        if self.step % self.display == 0:
            metrics_log = ''
            for (k, v) in self.metric_cache.items():
                val = v / self.display
                if abs(val) > 1e-3:
                    metrics_log += ' - %s: %.4f' % (k, val)
                else:
                    metrics_log += ' - %s: %.4e' % (k, val)
            print('step: {}/{} ... {}'.format(self.step,
                                          self.params['steps'],
                                          metrics_log))
            self.metric_cache.clear()
```            


