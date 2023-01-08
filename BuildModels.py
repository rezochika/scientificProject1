import sys

import absl.logging

from Models.dnn import build_dnn

absl.logging.set_verbosity(absl.logging.ERROR)
if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")
modelPath = 'modelLSTM6420230105/'
rebuild = False
ws = 7
lr = 0.008886502  # # 0.011353
ep = 5000
plot = False
Bi = False
dnnlayers = 64
for arg in sys.argv:
    if arg == 'rebuild=1':
        rebuild = True
    elif arg.startswith("modelPath"):
        modelPath = arg.replace("modelPath=", "")
    elif arg.startswith("epochs"):
        ep = int(arg.replace("epochs=", ""))
    elif arg.startswith("layers"):
        dnnlayers = int(arg.replace("layers=", ""))
    elif arg == 'plot=1':
        plot = True
    elif arg == 'bi=1':
        Bi = True

build_dnn(ws=ws, lr=lr, ep=ep, bi=Bi, dnnlayers=dnnlayers, model_path=modelPath)
