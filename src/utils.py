import numpy as np
import os
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


class Logger(object):
    def __init__(self, path, filename):
        path_filename = os.path.exists(os.path.join(path, filename+'.csv'))
        self.write_header = True
        self.row = {}
        assert not os.path.exists(path_filename), 'Logfile {} already exists'.format(filename)
        if not os.path.isdir(path):
            os.makedirs(path)
        self.f = open(path_filename, 'w')
        self.writer = None  # DictWriter created with first call to write() method

    def write(self, log, display=True):
        if display:
            self.disp_log(log)
        if self.write_header:
            fieldnames = [x for x in log.keys()]
            self.writer = csv.DictWriter(self.f, fieldnames=fieldnames)
            self.writer.writeheader()
            self.writer.writerow(log)
            self.write_header = False
        else:
            self.writer.writerow(log)
        self.row = {}

    @staticmethod
    def disp_log(log):
        """Print metrics to stdout"""
        log_keys = [k for k in log.keys()]
        log_keys.sort()
        print('***** Iteration {}, Mean R = {:.0f} *****'.format(log['_Iteration'],
                                                                 log['_MeanReward']))
        for key in log_keys:
            if key[0] != '_':  # don't display log items with leading '_'
                print('{:s}: {:.3g}'.format(key, log[key]))
        print('\n')

    def log_items(self, items):
        self.row.update(items)

    def close(self):
        self.f.close()
