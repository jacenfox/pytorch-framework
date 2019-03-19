import numpy as np
import pandas as pd
import os


try:
    import matplotlib
    matplotlib.use('Agg')  # Force matplotlib to not use any Xwindows backend.
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError


def callback_fun(deepnet_training_obj, network_output, data, target):
    # common process
    return


class CallbackSaveSamples(object):

    def __init__(self, call_me_every_n_epoch=20):
        pass

    def batch(self, deepnet_training_obj, network_output, target, data, loss_dict, is_training):
        callback_fun()

    def epoch(self, deepnet_training_obj, network_output, target, data, loss_dict, is_training):
        self.num_samples_already_saved = 0


class CallbackLossCurves(object):

    def __init__(self):
        self.loss_epoches_tr = pd.DataFrame()
        self.loss_epoches_vld = pd.DataFrame()
        self.loss_minibatch = pd.DataFrame()
        self.already_loaded_from_ckpt = False

    def batch(self, deepnet_training_obj, network_output, target, data, loss_dict, is_training):
        data_dict = {}
        for key in loss_dict.keys():
            try:
                data_dict[key] = loss_dict[key].data.cpu().numpy()
            except:
                print(key, type(loss_dict[key]))
                data_dict[key] = loss_dict[key]
        self.loss_minibatch = self.loss_minibatch.append(data_dict, ignore_index=True)

    def epoch(self, deepnet_training_obj, network_output, target, data, loss_dict, is_training):
        model_path = deepnet_training_obj.model_path
        self.log_dir = os.path.join(model_path, 'log')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        elif os.path.exists(os.path.join(self.log_dir, 'loss.train.csv')) and os.path.exists(os.path.join(self.log_dir, 'loss.validation.csv')) and (not self.already_loaded_from_ckpt):
            self.loss_epoches_tr = pd.read_csv(os.path.join(self.log_dir, 'loss.train.csv'))
            self.loss_epoches_vld = pd.read_csv(os.path.join(self.log_dir, 'loss.validation.csv'))
            self.already_loaded_from_ckpt = True
        if is_training:
            self.loss_epoches_tr = self.loss_epoches_tr.append(self.loss_minibatch.mean().to_frame().T, ignore_index=True)
            self.loss_epoches_tr.to_csv(os.path.join(self.log_dir, 'loss.train.csv'), index=False, float_format='%.5f')
        else:
            self.loss_epoches_vld = self.loss_epoches_vld.append(self.loss_minibatch.mean().to_frame().T, ignore_index=True)
            self.loss_epoches_vld.to_csv(os.path.join(self.log_dir, 'loss.validation.csv'), index=False, float_format='%.5f')

        if not is_training:
            self.plot()

        self.loss_minibatch = pd.DataFrame(data={})

    def plot(self):
        df_tr = self.loss_epoches_tr
        df_vld = self.loss_epoches_vld
        log_dir = self.log_dir
        for key in df_tr.columns:
            try:
                data = {key + '_tr': df_tr[key],
                        key + '_vld': df_vld[key],
                        }
                df = pd.DataFrame(data=data)
                df_smooth = df.rolling(5, axis=0, center=True, min_periods=1).mean()
                np.seterr(invalid='ignore')
                ylim = (np.minimum(np.min(df_tr[key].values), np.min(df_vld[key].values)) - 1e-8,
                        np.maximum(np.nanpercentile(df_tr[key].values, 95), np.nanpercentile(df_vld[key].values, 95)) + 1e-8)
                df.plot(ylim=ylim, color=['b', 'orange'], alpha=0.2)
                plt.plot(df_smooth.values[:, 0], linewidth=3, color='b')
                plt.plot(df_smooth.values[:, 1], linewidth=3, color='orange')
                plt.grid(); plt.ylim(ylim)
                plt.savefig(os.path.join(log_dir, key + '.pdf'))
                plt.clf(), plt.close()
            except:
                plt.clf(), plt.close()
                print('[Warning] Cannot plot: %s' % (key))
                pass


class CallbackGradientCurves(object):

    def __init__(self):
        self.grad_record = pd.DataFrame()
        self.grad_record_minibatch = pd.DataFrame()
        self.call_me_every_n_epoch = 1
        self.already_loaded_from_ckpt = False

    def batch(self, deepnet_training_obj, network_output, target, data, loss_dict, is_training):
        self.i_epoch = deepnet_training_obj.epoch
        if self.i_epoch % self.call_me_every_n_epoch != 0:
            return
        if not is_training:
            return

        data_dict = {}
        for key in deepnet_training_obj.model.model.grad_record.keys():
            data_dict[key] = deepnet_training_obj.model.model.grad_record[key].mag.data.cpu().numpy()
        self.grad_record_minibatch = self.grad_record_minibatch.append(data_dict, ignore_index=True)

    def epoch(self, deepnet_training_obj, network_output, target, data, loss_dict, is_training):
        self.i_epoch = deepnet_training_obj.epoch
        if (self.i_epoch + 1) % self.call_me_every_n_epoch != 0:
            return
        if not is_training:
            return
        model_path = deepnet_training_obj.model_path
        self.output_dir = os.path.join(model_path, 'gradient')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        elif os.path.exists(os.path.join(self.output_dir, 'gradient_records.csv')) and (len(self.grad_record) == 0):
            self.grad_record = pd.read_csv(os.path.join(self.output_dir, 'gradient_records.csv'))
            self.already_loaded_from_ckpt = True
        self.grad_record = self.grad_record.append(self.grad_record_minibatch.mean().to_frame().T, ignore_index=True)
        self.grad_record.to_csv(os.path.join(self.output_dir, 'gradient_records.csv'), index=False, float_format='%.15f')
        self.plot(df=self.grad_record, fname=os.path.join(self.output_dir, 'gradient_records.pdf'))
        self.grad_record_minibatch = pd.DataFrame(data={})

    def plot(self, df, fname):
        df = self.grad_record
        ylim = (np.nanpercentile(df.values, 5) - 1e-50, np.nanpercentile(df.values, 95) + 1e-50)
        try:
            df.plot(logy=True, ylim=ylim); plt.grid()
            plt.savefig(fname, bbox_inches='tight')
            plt.clf(), plt.close()
        except ValueError:
            print('[Warning] gradient has Nan or Inf')
