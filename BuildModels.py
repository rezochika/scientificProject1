import sys
import warnings
from datetime import datetime

import absl.logging

import sqlconnect
from Models.dnn import build_dnn
from Models.ols import build_ols
from Models.sarimax import build_sarimax

absl.logging.set_verbosity(absl.logging.ERROR)
if not sys.warnoptions:
    warnings.simplefilter("ignore")
ws = 7
lr = 0.008886502  # # 0.011353
ep = 5000
verbose = 0
print(datetime.now())
start = datetime.now()
modelPath = 'CompiledModels/' + datetime.now().strftime("%Y%m%d/")
df, df1 = sqlconnect.getdatafromsql()
build_ols(model_path=modelPath, model_id=1)
build_sarimax(model_path=modelPath, df=df.copy(),  model_id=10)
build_dnn(ws=ws, lr=lr, ep=ep, bi=False, dnnlayers=16, model_path=modelPath, df=df.copy(), verbose=verbose, model_id=2)
build_dnn(ws=ws, lr=lr, ep=ep, bi=True, dnnlayers=16, model_path=modelPath, df=df.copy(), verbose=verbose, model_id=3)
build_dnn(ws=ws, lr=lr, ep=ep, bi=False, dnnlayers=32, model_path=modelPath, df=df.copy(), verbose=verbose, model_id=4)
build_dnn(ws=ws, lr=lr, ep=ep, bi=True, dnnlayers=32, model_path=modelPath, df=df.copy(), verbose=verbose, model_id=5)
build_dnn(ws=ws, lr=lr, ep=ep, bi=False, dnnlayers=64, model_path=modelPath, df=df.copy(), verbose=verbose, model_id=6)
build_dnn(ws=ws, lr=lr, ep=ep, bi=True, dnnlayers=64, model_path=modelPath, df=df.copy(), verbose=verbose, model_id=7)
build_dnn(ws=ws, lr=lr, ep=ep, bi=False, dnnlayers=128, model_path=modelPath, df=df.copy(), verbose=verbose, model_id=8)
build_dnn(ws=ws, lr=lr, ep=ep, bi=True, dnnlayers=128, model_path=modelPath, df=df.copy(), verbose=verbose, model_id=9)
print(datetime.now())
print(datetime.now()-start)
