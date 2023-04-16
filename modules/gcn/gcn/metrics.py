import torch


def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    # loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    loss = torch.nn.functional.cross_entropy(preds, labels, reduction='none')
    # mask = tf.cast(mask, dtype=tf.float32)
    mask = mask.type(torch.float32)
    # mask /= tf.reduce_mean(mask)
    mask /= torch.mean(mask)
    loss *= mask
    return torch.mean(loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    # correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    correct_prediction = torch.argmax(preds, dim=1) == torch.argmax(labels, dim=1)
    # accuracy_all = tf.cast(correct_prediction, tf.float32)
    accuracy_all = correct_prediction.type(torch.float32)
    # mask = tf.cast(mask, dtype=tf.float32)
    mask = mask.type(torch.float32)
    # mask /= tf.reduce_mean(mask)
    mask /= torch.mean(mask)
    accuracy_all *= mask
    # return tf.reduce_mean(accuracy_all)
    return torch.mean(accuracy_all)
