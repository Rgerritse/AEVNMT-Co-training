import argparse

def add_arguments(parser):
    parser.add_argument("--in_dir", type=str, default=None, help="Path to input directory")
    parser.add_argument("--in_src", type=str, default=None, help="Path to input source file")
    parser.add_argument("--in_tgt", type=str, default=None, help="Path to input target file")

    parser.add_argument("--out_dir", type=str, default=None, help="Path to output directory")
    parser.add_argument("--out_src_train", type=str, default=None, help="Path to output train  source file")
    parser.add_argument("--out_tgt_train", type=str, default=None, help="Path to output train target file")
    parser.add_argument("--out_src_dev", type=str, default=None, help="Path to output train  source file")
    parser.add_argument("--out_tgt_dev", type=str, default=None, help="Path to output train target file")
    parser.add_argument("--out_src_test", type=str, default=None, help="Path to output train  source file")
    parser.add_argument("--out_tgt_test", type=str, default=None, help="Path to output train target file")

def split_data():
    src = []
    with open(FLAGS.in_dir + FLAGS.in_src, 'r') as f_in_src:
        for line in f_in_src:
            src.append(line)

    tgt = []
    with open(FLAGS.in_dir + FLAGS.in_tgt, 'r') as f_in_tgt:
        for line in f_in_tgt:
            tgt.append(line)

    dev_index = len(src) // 10 * 8
    test_index = len(src) // 10 * 9

    with open(FLAGS.out_dir + FLAGS.out_src_train, 'w+') as f_out_src_train:
        for sent in src[:dev_index]:
            f_out_src_train.write(sent)

    with open(FLAGS.out_dir + FLAGS.out_tgt_train, 'w+') as f_out_tgt_train:
        for sent in tgt[:dev_index]:
            f_out_tgt_train.write(sent)

    with open(FLAGS.out_dir + FLAGS.out_src_dev, 'w+') as f_out_src_dev:
        for sent in src[dev_index:test_index]:
            f_out_src_dev.write(sent)

    with open(FLAGS.out_dir + FLAGS.out_tgt_dev, 'w+') as f_out_tgt_dev:
        for sent in tgt[dev_index:test_index]:
            f_out_tgt_dev.write(sent)

    with open(FLAGS.out_dir + FLAGS.out_src_test, 'w+') as f_out_src_test:
        for sent in src[test_index:]:
            f_out_src_test.write(sent)

    with open(FLAGS.out_dir + FLAGS.out_tgt_test, 'w+') as f_out_tgt_test:
        for sent in tgt[test_index:]:
            f_out_tgt_test.write(sent)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    FLAGS, unparsed = parser.parse_known_args()
    print(FLAGS)
    split_data()
    # preprocess()
