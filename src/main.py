from dataset import KnowledgeGraph
from model import TransE

import tensorflow as tf
import argparse


def main():
    parser = argparse.ArgumentParser(description='TransE')
    parser.add_argument('--mode', type=str, default='eval')
    parser.add_argument('--data_dir', type=str, default='../data/FB15k/')
    parser.add_argument('--embedding_dim', type=int, default=200)
    parser.add_argument('--margin_value', type=float, default=1.0)
    parser.add_argument('--score_func', type=str, default='L1')
    parser.add_argument('--batch_size', type=int, default=4800)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--n_generator', type=int, default=24)
    parser.add_argument('--n_rank_calculator', type=int, default=24)
    parser.add_argument('--hit_at_n', type=int, default=10)
    parser.add_argument('--ckpt_dir', type=str, default='../ckpt/')
    parser.add_argument('--summary_dir', type=str, default='../summary/')
    parser.add_argument('--max_epoch', type=int, default=500)
    parser.add_argument('--eval_freq', type=int, default=10)
    args = parser.parse_args()
    print(args)
    kg = KnowledgeGraph(data_dir=args.data_dir)
    kge_model = TransE(kg=kg, model_path=args.ckpt_dir,embedding_dim=args.embedding_dim, margin_value=args.margin_value,
                       score_func=args.score_func, batch_size=args.batch_size, learning_rate=args.learning_rate,
                       n_generator=args.n_generator, n_rank_calculator=args.n_rank_calculator, hit_at_n=args.hit_at_n)
    gpu_config = tf.GPUOptions(allow_growth=True)
    sess_config = tf.ConfigProto(gpu_options=gpu_config)
    if args.mode == 'test':
        saver = tf.train.Saver()
    else:
        saver = tf.train.Saver(tf.global_variables())
    with tf.Session(config=sess_config) as sess:
        if args.mode == 'eval':
            print('-----Loading from checkpoints-----')
            ckpt_file = tf.train.latest_checkpoint(args.ckpt_dir)
            saver.restore(sess, ckpt_file)
            kge_model.launch_evaluation(session=sess)
        elif args.mode == 'train':
            print('-----Initializing tf graph-----')
            tf.global_variables_initializer().run()
            print('-----Initialization accomplished-----')
            kge_model.check_norm(session=sess)
            summary_writer = tf.summary.FileWriter(logdir=args.summary_dir, graph=sess.graph)
            for epoch in range(args.max_epoch):
                print('=' * 30 + '[EPOCH {}]'.format(epoch) + '=' * 30)
                kge_model.launch_training(epoch=epoch, session=sess, summary_writer=summary_writer, saver=saver)
                # if (epoch + 1) % args.eval_freq == 0:
                #     kge_model.launch_evaluation(session=sess)
        elif args.mode == 'predict':
            print('-----Loading from checkpoints-----')
            ckpt_file = tf.train.latest_checkpoint(args.ckpt_dir)
            saver.restore(sess, ckpt_file)
            kge_model.launch_prediction(session=sess)
        else:
            print('Wrong mode!! (mode = train|test|predict)')


if __name__ == '__main__':
    main()
