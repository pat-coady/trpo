"""
Logging and Data Scaling Utilities

Written by Patrick Coady (pat-coady.github.io)
"""
import numpy as np
import os
import shutil
import glob
import csv


class Scaler(object):
    def __init__(self, obs_dim, alpha=0.05):
        self.vars = np.zeros(obs_dim) + 1.0
        self.means = np.zeros(obs_dim)
        self.first_pass = True
        self.alpha = alpha

    def update_scale(self, x):

        if self.first_pass:
            self.vars = np.var(x, axis=0)
            self.means = np.mean(x, axis=0)
            self.first_pass = False
        else:
            self.vars += self.alpha * (np.var(x, axis=0) - self.vars)
            self.means += self.alpha * (np.mean(x, axis=0) - self.means)

    def get_scale(self):
        return self.means, np.sqrt(self.vars)


class ConstantScaler(object):
    """ Dumb scaler, scale and offset set at initialization """
    def __init__(self, obs_dim, scale=1.0, offset=0.0):
        self.scale = np.ones(obs_dim) * scale
        self.offset = np.zeros(obs_dim) + offset

    def update(self, x):
        pass  # no updates for constant scaler

    def get(self):
        return self.scale, self.offset


class Logger(object):
    """ Simple training logger: saves to file and optionally prints to stdout """
    def __init__(self, logname, now):
        """
        Args:
            logname: name for log (e.g. 'Hopper-v1')
            now: unique sub-directory name (e.g. date/time string)
        """
        path = os.path.join('log-files', logname, now)
        os.makedirs(path)
        filenames = glob.glob('*.py')  # put copy of all python files in log_dir
        for filename in filenames:     # for reference
            shutil.copy(filename, path)
        path = os.path.join(path, 'log.csv')

        self.write_header = True
        self.row = {}
        self.f = open(path, 'w')
        self.writer = None  # DictWriter created with first call to write() method

    def write(self, display=True):
        """ Write 1 log entry to file, and optionally to stdout
        Log fields preceded by '_' will not be printed to stdout

        Args:
            display: boolean, print to stdout
        """
        if display:
            self.disp(self.row)
        if self.write_header:
            fieldnames = [x for x in self.row.keys()]
            self.writer = csv.DictWriter(self.f, fieldnames=fieldnames)
            self.writer.writeheader()
            self.writer.writerow(self.row)
            self.write_header = False
        else:
            self.writer.writerow(self.row)
        self.row = {}

    @staticmethod
    def disp(log):
        """Print metrics to stdout"""
        log_keys = [k for k in log.keys()]
        log_keys.sort()
        print('***** Iteration {}, Mean R = {:.0f} *****'.format(log['_Iteration'],
                                                                 log['_MeanReward']))
        for key in log_keys:
            if key[0] != '_':  # don't display log items with leading '_'
                print('{:s}: {:.3g}'.format(key, log[key]))
        print('\n')

    def log(self, items):
        """ Update fields in log (does not write to file, used to collect updates.

        Args:
            items: dictionary of items to update
        """
        self.row.update(items)

    def close(self):
        """ Close log file - log cannot be written after this """
        self.f.close()
