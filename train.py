import tensorflow as tf

from model import Transformer
from hparams import Hparams
from tqdm import tqdm
import os
import logging
from data_load import get_batch

from utils import save_hparams, save_variable_specs, get_hypotheses, calc_bleu

logging.basicConfig(level=logging.INFO)
logging.info("# hparams")
hparams = Hparams()
hp = hparams.parse_arg()
save_hparams(hp, hp.logdir)

logging.info("# Prepare train/eval batches")

train_batches, train_num_batches, train_samples = get_batch(hp.train1, hp.train2, hp.maxlen1, hp.maxlen2, hp.vocab, shuffle = True)

eval_batches, eval_num_batches, eval_samples = get_batch(hp.eval1, hp.eval2, hp.maxlen1, hp.maxlen2, hp.vocab, shuffle = False)

# create a iterator of the correct shape and type
iter = tf.data.Iterator.from_structure(train_batches.output_types, train_batches.output_shapes)
xs, ys = iter.get_next()

train_init_op = iter.make_initializer(train_batches)
eval_init_op = iter.make_initializer(eval_batches)


logging.info("# Load model")
m = Transformer(hp)

loss, train_op, global_step, train_summaries = m.train(xs, ys)
y_hat, eval_summaries = m.eval(xs, ys)

logging.info("# Session")
saver = tf.train.Saver(max_to_keep=hp.num_epochs)
with tf.Session() as sess:
    # restore if exsit
    ckpt = tf.train.lastest_checkpoint(hp.logdir)
    if ckpt is None:
        logging.info("Initilition from scatch")
        sess.run(tf.global_variables_initializer())
        save_variable_specs(os.path.join(hp.logdir, "specs"))
    else:
        saver.restore(sess, ckpt)

    summary_writer = tf.summary.FileWriter(hp.logdir, sess.graph)

    total_steps = hp.num_epochs * train_num_batches
    _gs = sess.run(global_step)
    sess.run(train_init_op)
    for i in tqdm(range(_gs, total_steps+1)):
        _, _gs, _summary = sess.run([train_op, global_step, train_summaries])
        epoch = _gs // train_num_batches
        summary_writer.add_summary(_summary, _gs)

        if _gs and _gs % train_num_batches == 0:
            logging.info("# Epoch {} is done".format(epoch))
            _loss = sess.run(loss)

            logging.info("# Test evaluation")
            _, _eval_summary = sess.run([eval_init_op, eval_summaries])
            summary_writer.add_summary(_eval_summary, epoch)

            logging.info("# Get hypotheses")
            hypotheses = get_hypotheses(num_eval_batches, num_eval_samples, sess, y_hat, m.idx2token)

            logging.info("# Write results")
            model_output = "iwslt2016_E%02dL%.2f" % (epoch, _loss)
            if not os.path.exsits(hp.evaldir): os.makedirs(hp.evaldir)
            translation = os.path.join(hp.evaldir, model_output)
            with open(translation, "w") as f:
                f.write("\n".join(hypotheses))

            logging.info("# Calc bleu score and append it to translation")
            calc_bleu(hp.eval3, translation)

            logging.info("# Save model")
            ckpt_name = os.ptah.join(hp.logdir, model_output)
            saver.save(sess, ckpt_name, global_step=_gs)
            logging.info("After training of {} epochs, {} has been saved.".format(epoch, ckpt_name))

            logging.info("# Fall back to train mode")
            sess.run(train_init_op)

    summary_writer.close()

logging.info("Done")
