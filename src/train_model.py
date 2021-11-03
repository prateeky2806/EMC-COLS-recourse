from argparse import ArgumentParser
import os
import torch
from torch import nn
import torch.utils.data as data_utils

import pytorch_lightning as pl
from pytorch_lightning.metrics import functional as FM
from pytorch_lightning.callbacks import ModelCheckpoint

from model.mlp import MLP
# from utils.data_io import DataIO
# from loaders.data_loaders.pandas_interface import PandasDataLoader


class PL_Trainer(pl.LightningModule):

    def __init__(self, input_dim, hidden_dim=20, output_dim=2):
        super().__init__()
        self.save_hyperparameters()
        self.model = MLP(input_dim, hidden_dim, output_dim)
        self.loss_fn = self.configure_loss(loss_type='CE')

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        logits = self.model(x)
        return logits

    def shared_step(self, batch, batch_idx):
        x, y = batch
        z = self.model(x)
        loss = self.loss_fn(z, y.long())
        y_hat = torch.argmax(z, dim=1)
        acc = FM.accuracy(y_hat, y.int())
        return loss, acc, z

    def training_step(self, batch, batch_idx):
        loss, acc, z = self.shared_step(batch, batch_idx)
        metrics = {'train_acc': acc, 'train_loss': loss}
        self.log_dict(metrics, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc, z = self.shared_step(batch, batch_idx)
        metrics = {'val_acc': acc, 'val_loss': loss}
        self.log_dict(metrics, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, acc, _ = self.shared_step(batch, batch_idx)
        metrics = {'test_acc': acc, 'test_loss': loss}
        self.log_dict(metrics, prog_bar=True)

    def configure_loss(self, loss_type='CE'):
        if loss_type == 'CE':
            return nn.CrossEntropyLoss()
        else:
            raise ValueError(" Loss can only be Cross Entropy currently")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.00001)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parser


def prepare_data_for_training(pandas_loader, split='train'):
    if split == 'train':
        data = pandas_loader.train_df
    elif split == 'val':
        data = pandas_loader.val_df
    elif split == 'test':
        data = pandas_loader.test_df
    else:
        raise ValueError('split can only be train, val or test.')

    y_data = torch.tensor(data[pandas_loader.outcome_name].values, dtype=torch.float32)
    x_data = data.drop(pandas_loader.outcome_name, 1)
    x_data = torch.tensor(pandas_loader.transform_data(x_data, encode=True, normalise=True, return_numpy=True), dtype=torch.float32)
    return x_data, y_data

def get_training_data_loaders(args):
    """Get pytorch data-loaders for the adult dataset."""
    from utils.data_io import DataIO
    from utils.helpers import setup_data
    from loaders.data_loaders.pandas_interface import PandasDataLoader

    pandas_data_obj = setup_data(args)

    x_train, y_train = prepare_data_for_training(pandas_data_obj, split='train')
    x_val, y_val = prepare_data_for_training(pandas_data_obj, split='val')
    x_test, y_test = prepare_data_for_training(pandas_data_obj, split='test')
    input_dim = x_train.shape[1]

    train_tensor = data_utils.TensorDataset(x_train, y_train)
    val_tensor = data_utils.TensorDataset(x_val, y_val)
    test_tensor = data_utils.TensorDataset(x_test, y_test)

    train_loader = data_utils.DataLoader(dataset=train_tensor, batch_size=args.batch_size, shuffle=True)
    val_loader = data_utils.DataLoader(dataset=val_tensor, batch_size=args.batch_size, shuffle=False)
    test_loader = data_utils.DataLoader(dataset=test_tensor, batch_size=args.batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, input_dim

def cli_main():
    pl.seed_everything(1234)

    parser = ArgumentParser()
    # Training args.
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--hidden_dim', type=int, default=20)
    parser.add_argument('--num_classes', type=int, default=2)
    # parser.add_argument('--learning_rate', type=float, default=0.0001)

    # Data Loading args
    parser.add_argument('--data_name', default='adult_binary', type=str)
    parser.add_argument('--project_dir', default='../', type=str)
    parser.add_argument('--data_folder', default='data', type=str)
    parser.add_argument('--save_data', action='store_true', help='Save the data splits.')
    parser.add_argument('--dump_negative_data', action='store_true', help='Test the test data and dump only negative class in test.')
    parser.add_argument('--verbose', action='store_true')
    # Logger args
    parser.add_argument('--name', type=str, default='test')
    parser.add_argument('--project', type=str, default='test-project')
    parser = pl.Trainer.add_argparse_args(parser)
    parser = PL_Trainer.add_model_specific_args(parser)
    args = parser.parse_args()

    train_loader, val_loader, test_loader, input_dim = get_training_data_loaders(args)
    model = PL_Trainer(input_dim, args.hidden_dim, args.num_classes)

    checkpoint_callback = ModelCheckpoint(monitor='val_acc',
                                        dirpath=os.path.join(args.project_dir, 'trained_models/'),
                                        filename=f'{args.data_name}' + '-default',
                                        save_top_k=1,
                                        mode='max')
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback])
    trainer.fit(model, train_loader, val_loader)

    result = trainer.test(test_dataloaders=test_loader)
    print(result)


if __name__ == '__main__':
    cli_main()
