# -*- coding:utf-8 -*-
"""
File: logger.py
File Created: 2021-02-17
Author: nyLiao
"""
import os
from abc import ABC
from datetime import datetime
import uuid
import json
from dotmap import DotMap

import torch


PATH_SAVE_LOCAL = "./saved_models/"


def prepare_opt(parser):
    # Parser to dict
    opt_parser = vars(parser.parse_args())
    # Config file to dict
    try:
        config_path = opt_parser['config']
        with open(config_path, 'r') as config_file:
            opt_config = json.load(config_file)
        # Merge dicts to dotmap
        return DotMap({**opt_parser, **opt_config})
    except:
        return DotMap({**opt_config})


class Logger(ABC):
    def __init__(self, prj_name, save_path=PATH_SAVE_LOCAL, flag_run=''):
        super(Logger, self).__init__()

        # init log directory
        self.seed_str = str(uuid.uuid4())[:6]
        self.seed = int(self.seed_str, 16)
        if not flag_run:
            flag_run = datetime.now().strftime("%m%d") + '-' + self.seed_str
        elif flag_run.count('date') > 0:
            flag_run = flag_run.replace('date', datetime.now().strftime("%m%d"))
        else:
            pass
        self.dir_save = os.path.join(save_path, prj_name, flag_run)

        self.path_existed = os.path.exists(self.dir_save)
        os.makedirs(self.dir_save, exist_ok=True)

        # init log file
        self.flag_run = flag_run
        self.file_log = self.path_join('log.txt')
        self.file_config = self.path_join('config.json')

    def path_join(self, *args):
        """
        Generate file path in current directory.
        """
        return os.path.join(self.dir_save, *args)

    def print(self, s):
        """
        Print string to console and write log file.
        """
        print(s)
        with open(self.file_log, 'a') as f:
            f.write(str(s) + '\n')

    def print_on_top(self, s):
        """
        Print string on top of log file.
        """
        print(s)
        with open(self.file_log, 'a') as f:
            pass
        with open(self.file_log, 'r+') as f:
            temp = f.read()
            f.seek(0, 0)
            f.write(str(s) + '\n')
            f.write(temp)

    def save_opt(self, opt):
        with open(self.file_config, 'a') as f:
            json.dump(opt.toDict(), fp=f, indent=4, sort_keys=False)
        print("Option saved.")
        print("Config path: {}".format(self.file_config))
        print("Option dict: {}\n".format(opt.toDict()))

    def load_opt(self):
        with open(self.file_config, 'r') as config_file:
            opt = DotMap(json.load(config_file))
        print("Option loaded.")
        print("Config path: {}".format(self.file_config))
        print("Option dict: {}\n".format(opt.toDict()))
        return opt


class ModelLogger(ABC):
    """
    Log, save, and load training states, with given path and certain prefix.
    """
    def __init__(self, logger, prefix='model', state_only=True):
        super(ModelLogger, self).__init__()
        self.logger = logger
        self.prefix = prefix
        self.state_only = state_only

        self.model = None
        self.optimizer = None
        self.scheduler = None

        # Metric
        self.metric_best = 0
        self.metric_name = 'acc'
        self.metric_better = lambda curr: curr > self.metric_best
        self.epoch_best = -1

    @property
    def state_dict(self):
        return self.model.state_dict()

    def __set_model(self, model):
        self.model = model
        return self.model

    def regi_model(self, model, save_init=True):
        """
        Get model from parameters.

        Args:
            model: model instance
            save_init (bool, optional): Whether save initial model. Defaults to True.
        """
        self.__set_model(model)
        if save_init:
            self.save('0')

    def regi_state(self, optimizer=None, scheduler=None, save_init=True):
        self.optimizer = optimizer
        self.scheduler = scheduler
        if save_init:
            self.save_state('0')

    def load_model(self, *suffix, model=None, path_force=None):
        """
        Get model from file.
        """
        name = '_'.join((self.prefix,) + suffix)
        if path_force is None:
            path = self.logger.path_join(name + '.pth')
        else:
            path = os.path.join(path_force, name + '.pth')

        if self.state_only:
            if model is None:
                model = self.model
            state_dict = torch.load(path)
            model.load_state_dict(state_dict)
        else:
            model = torch.load(path)
        return self.__set_model(model)

    def load_state(self, *suffix, optimizer=None, scheduler=None):
        name = '_'.join((self.prefix,) + suffix)
        path = self.logger.path_join(name + '.pt')
        state_dict = torch.load(path)

        if optimizer is None:
            optimizer = self.optimizer
        if optimizer is not None:
            optimizer.load_state_dict(state_dict["optimizer"])
            self.optimizer = optimizer

        if scheduler is None:
            scheduler = self.scheduler
        if scheduler is not None:
            scheduler.load_state_dict(state_dict["scheduler"])
            self.scheduler = scheduler

        self.metric_best = state_dict["metric_best"]
        self.epoch_best = state_dict["epoch_best"]

        state_dict_ret = {
            "epoch": state_dict["epoch"],
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "metric_best": self.metric_best,
            "epoch_best": self.epoch_best,
        }
        return state_dict_ret

    def get_last_epoch(self):
        """
        Get last saved model epoch.

        Returns:
            int: number of last epoch
        """
        name_pre = '_'.join((self.prefix,) + ('',))
        last_epoch = -2

        for fname in os.listdir(self.logger.dir_save):
            fname = str(fname)
            if fname.startswith(name_pre) and fname.endswith('.pth'):
                suffix = fname.replace(name_pre, '').replace('.pth', '')
                if suffix == 'init':
                    this_epoch = -1
                elif suffix.isdigit():
                    # correct the `epoch + 1` in `save_epoch()`
                    this_epoch = int(suffix) - 1
                else:
                    this_epoch = -2
                if this_epoch > last_epoch:
                    last_epoch = this_epoch
        return last_epoch

    def save(self, *suffix):
        """
        Save model with given name string.
        """
        name = '_'.join((self.prefix,) + suffix)
        path = self.logger.path_join(name + '.pth')

        # self.model.cpu()
        if self.state_only:
            torch.save(self.state_dict, path)
        else:
            torch.save(self.model, path)

    def save_state(self, *suffix, epoch=-1):
        name = '_'.join((self.prefix,) + suffix)
        path = self.logger.path_join(name + '.pt')

        state_dict = {
            "epoch": epoch,
            "optimizer": self.optimizer.state_dict() if self.optimizer is not None else None,
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "metric_best": self.metric_best,
            "epoch_best": self.epoch_best,
        }
        torch.save(state_dict, path)

    def save_epoch(self, epoch, period=1):
        """
        Save model each epoch period.

        Args:
            epoch (int): Current epoch. Start from 0 (display as epoch + 1).
            period (int, optional): Save period. Defaults to 1 (save every epochs).
        """
        if (epoch + 1) % period == 0:
            self.save(str(epoch+1))
            self.save_state(str(epoch+1), epoch=epoch)

    def print_best(self, top=False):
        if top:
            self.logger.print_on_top(f'[best]  Epoch {self.epoch_best+1:>3},  {self.metric_name}: {self.metric_best:>.4f}')
        else:
            self.logger.print(f'[best]  Epoch {self.epoch_best+1:>3},  {self.metric_name}: {self.metric_best:>.4f}')

    def save_best(self, metric_curr, epoch=-1, verbose=True):
        """
        Save model with best accuracy.

        Args:
            metric_curr (int/float): Current accuracy.
        """
        if self.metric_better(metric_curr):
            self.metric_best = metric_curr
            self.epoch_best = epoch

            if epoch > 5:
                self.save('best')
                # self.save_state('best', epoch=epoch)

            if verbose:
                self.print_best(top=False)

        return self.metric_best
