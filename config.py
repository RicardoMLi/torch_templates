import argparse


class Args:
    @staticmethod
    def parse():
        parser = argparse.ArgumentParser()

        return parser

    @staticmethod
    def initialize(parser):
        parser.add_argument('--output_dir', default='./checkpoints/',
                            help='the output dir for the model checkpoints')

        parser.add_argument('--data_dir', default='./data/', help='input data dir')
        parser.add_argument('--log_dir', default='./logs/', help='log dir')
        parser.add_argument('--seed', default=42, type=int, help='random seed')
        parser.add_argument('--patience', default=10, type=int, help='early stopping patience')
        parser.add_argument('--batch_size_train', default=512, type=int, help='train batch size')
        parser.add_argument('--batch_size_test', default=256, type=int, help='test batch size')
        parser.add_argument('--epochs', default=60, type=int, help='the number of training epochs')
        parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
        parser.add_argument('--train_size', default=0.8, type=float, help='the rate of training samples')

        return parser

    def get_parsr(self):
        parser = self.parse()
        parser = self.initialize(parser)

        return parser.parse_args()


