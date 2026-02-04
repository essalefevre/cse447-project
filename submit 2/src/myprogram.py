#!/usr/bin/env python
import os
import string
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import pickle


class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """

    @classmethod
    def load_training_data(cls):
        # your code here
        # this particular model doesn't train
        data = []
        with open('src/training_data.txt', 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n')  # remove newline but keep the text
                if line:
                    data.append(line)
        return data

    @classmethod
    def load_test_data(cls, fname):
        # your code here
        data = []
        with open(fname) as f:
            for line in f:
                inp = line[:-1]  # the last character is a newline
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, data, work_dir):
        # your code here
        self.k = 2  # context length
        self.counts = {}

        for line in data:
            for i in range(len(line) - self.k):
                context = line[i:i+self.k]
                next_char = line[i+self.k]

                if context not in self.counts:
                    self.counts[context] = {}
                self.counts[context][next_char] = (
                    self.counts[context].get(next_char, 0) + 1
                )
        pass

    def run_pred(self, data):
        # your code here
        preds = []
        all_chars = string.ascii_letters
        for inp in data:
            # this model just predicts a random character each time
            # top_guesses = [random.choice(all_chars) for _ in range(3)]
            # preds.append(''.join(top_guesses))

            context = inp[-self.k:] if len(inp) >= self.k else inp

            if context in self.counts:
                next_chars = sorted(
                    self.counts[context].items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                top3 = [c for c, _ in next_chars[:3]]
            else:
                # fallback: common characters -- this is used every time since we don't have training data rn
                top3 = ['e', ' ', 'a']

            preds.append(''.join(top3))
        return preds

    def save(self, work_dir):
        # your code here
        # this particular model has nothing to save, but for demonstration purposes we will save a blank file
        # with open(os.path.join(work_dir, 'model.checkpoint'), 'wt') as f:
        #     f.write('dummy save')
        with open(os.path.join(work_dir, 'model.checkpoint'), 'wb') as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, work_dir):
        # your code here
        # this particular model has nothing to load, but for demonstration purposes we will load a blank file
        # with open(os.path.join(work_dir, 'model.checkpoint')) as f:
        #     dummy_save = f.read()
        # return MyModel()
        model = cls()
        with open(os.path.join(work_dir, 'model.checkpoint'), 'rb') as f:
            model.__dict__ = pickle.load(f)
        return model


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instatiating model')
        model = MyModel()
        print('Loading training data')
        train_data = MyModel.load_training_data()
        print('Training')
        model.run_train(train_data, args.work_dir)
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(args.work_dir)
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
