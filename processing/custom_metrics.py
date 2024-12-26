import tensorflow as tf


class BinaryPrecision(tf.keras.metrics.Metric):
    def __init__(self, threshold = 0.5, name='binary_precision', **kwargs):
        super(BinaryPrecision, self).__init__(name=name, **kwargs)
        self.threshold = threshold
        self.precision_metric = tf.keras.metrics.Precision()
       
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_label = tf.cast(y_pred >= self.threshold, tf.int32)
        self.precision_metric.update_state(y_true, y_pred_label)
       
    def result(self):
        precision = self.precision_metric.result()
        return precision

    def reset_state(self):
        self.precision_metric.reset_states()
        
class BinaryRecall(tf.keras.metrics.Metric):
    def __init__(self, threshold = 0.5, name='binary_recall', **kwargs):
        super(BinaryRecall, self).__init__(name=name, **kwargs)
        self.threshold = threshold
        self.recall_metric = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_label = tf.cast(y_pred >= self.threshold, tf.int32)
        self.recall_metric.update_state(y_true, y_pred_label)

    def result(self):
        recall = self.recall_metric.result()
        return recall

    def reset_state(self):
        self.recall_metric.reset_states()        
        
class BinaryF1Score(tf.keras.metrics.Metric):
    def __init__(self, threshold = 0.5, name='binary_f1_score', **kwargs):
        super(BinaryF1Score, self).__init__(name=name, **kwargs)
        self.threshold = threshold
        self.precision_metric = tf.keras.metrics.Precision()
        self.recall_metric = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_label = tf.cast(y_pred >= self.threshold, tf.int32)
        self.precision_metric.update_state(y_true, y_pred_label)
        self.recall_metric.update_state(y_true, y_pred_label)

    def result(self):
        precision = self.precision_metric.result()
        recall = self.recall_metric.result()
        f1_score = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
        return f1_score

    def reset_state(self):
        self.precision_metric.reset_states()
        self.recall_metric.reset_states()