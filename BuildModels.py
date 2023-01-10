import sys
from datetime import datetime
import absl.logging
import warnings
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

modelPath = 'CompiledModels/' + datetime.now().strftime("%Y%m%d/")
df, df1 = sqlconnect.getdatafromsql()
build_dnn(ws=ws, lr=lr, ep=ep, bi=False, dnnlayers=32, model_path=modelPath, df=df.copy())
build_dnn(ws=ws, lr=lr, ep=ep, bi=True, dnnlayers=32, model_path=modelPath, df=df.copy())
build_dnn(ws=ws, lr=lr, ep=ep, bi=False, dnnlayers=64, model_path=modelPath, df=df.copy())
build_dnn(ws=ws, lr=lr, ep=ep, bi=True, dnnlayers=64, model_path=modelPath, df=df.copy())
build_dnn(ws=ws, lr=lr, ep=ep, bi=True, dnnlayers=128, model_path=modelPath, df=df.copy())
build_dnn(ws=ws, lr=lr, ep=ep, bi=False, dnnlayers=128, model_path=modelPath, df=df.copy())
build_sarimax(model_path=modelPath, df=df.copy())
build_ols(model_path=modelPath, df=df.copy())
