import keras.callbacks
import tqdm
import ipywidgets
import threading
import warnings
from IPython.display import display

import matplotlib
matplotlib.use('nbagg')
import matplotlib.pyplot as plt

def fit(model, *fit_args, plots=['loss'], **fit_kwargs):
    """
    Interactive model training in jupyter notebook
    
    Just replace
        model.fit(model, x, y, nb_epoch=42, ...)
    with
        fit(model, x, y, nb_epoch=42, ...)
    to get interactive training
    """
    nbagg_backend = matplotlib.backends.backend.lower()=='nbagg'
    if not nbagg_backend:
        warnings.warn('''\nUse nbAgg backend if possible to prevent stacking display of training plots.
        Add import matplotlib; matplotlib.use(\'nbagg\') and restart kernel''')
    
    nb_epoch = fit_kwargs.get('nb_epoch',10)
    
    progress_bar = tqdm.tqdm_notebook(total=nb_epoch)

    class LossHistory(keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.fig=plt.figure()
            self.vals=dict()
        def on_epoch_end(self, epoch, logs={}):
            progress_bar.update()
            for k in logs.keys():
                val_list=self.vals.get(k,[])
                val_list.append(logs[k])
                self.vals[k]=val_list
        def on_train_end(self, logs={}):
            progress_bar.close()
            
    cancel_button = ipywidgets.Button(description='Cancel')
    
    def cancel_fitting():
        model.stop_training=True
        learning_thread.join(5)
        print('Cancelled')

    cancel_button.on_click(lambda sender: cancel_fitting())

    loss_history=LossHistory()

    display(cancel_button)

    if nbagg_backend:
        axis=None
        def plot_loss(sender):
            nonlocal axis
            if axis is None:
                axis=plt.gca()
            axis.clear()
            for n,l in loss_history.vals.items():
                if n in plots:
                    axis.semilogy(l, label=n)
            axis.legend(loc='best', frameon=False)
            plt.show()
    else:
        def plot_loss(sender):
            for n,l in loss_history.vals.items():
                if n in plots:
                    plt.semilogy(l, label=n)
            plt.legend(loc='best', frameon=False)
            plt.show()

    plot_button=ipywidgets.Button(description='Plot')
    plot_button.on_click(plot_loss)

    display(plot_button)

    callbacks=fit_kwargs.get('callbacks',[])
    callbacks=callbacks.copy()
    callbacks.append(loss_history)
    fit_kwargs['callbacks']=callbacks
    fit_kwargs['verbose']=False
    learning_thread=threading.Thread(target=lambda:model.fit(*fit_args, **fit_kwargs))
    learning_thread.start()