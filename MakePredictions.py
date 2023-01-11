import sys
import warnings
from datetime import datetime

import absl.logging

import sqlconnect
from Models.dnn import predict_dnn
from Models.ols import predict_ols
from Models.sarimax import predict_sarimax

absl.logging.set_verbosity(absl.logging.ERROR)
if not sys.warnoptions:
    warnings.simplefilter("ignore")
ws = 7
lr = 0.008886502  # # 0.011353
ep = 5000
verbose = 0
print(datetime.now())
start = datetime.now()
modelPath = 'CompiledModels/20230111/'# + datetime.now().strftime("%Y%m%d/")
df, df1 = sqlconnect.getdatafromsql()
model_id, res = predict_ols(model_path=modelPath, model_id=1)
results = {model_id: res}
model_id, res = predict_sarimax(model_path=modelPath, df=df.copy(), df1=df1.copy(),  model_id=10)
results[model_id] = res
model_id, res = predict_dnn(ws=ws, lr=lr, ep=ep, bi=False, dnnlayers=16, model_path=modelPath, df=df.copy(), df1=df1.copy(), verbose=verbose, model_id=2)
results[model_id] = res
model_id, res = predict_dnn(ws=ws, lr=lr, ep=ep, bi=True, dnnlayers=16, model_path=modelPath, df=df.copy(), df1=df1.copy(), verbose=verbose, model_id=3)
results[model_id] = res
model_id, res = predict_dnn(ws=ws, lr=lr, ep=ep, bi=False, dnnlayers=32, model_path=modelPath, df=df.copy(), df1=df1.copy(), verbose=verbose, model_id=4)
results[model_id] = res
model_id, res = predict_dnn(ws=ws, lr=lr, ep=ep, bi=True, dnnlayers=32, model_path=modelPath, df=df.copy(), df1=df1.copy(), verbose=verbose, model_id=5)
results[model_id] = res
model_id, res = predict_dnn(ws=ws, lr=lr, ep=ep, bi=False, dnnlayers=64, model_path=modelPath, df=df.copy(), df1=df1.copy(), verbose=verbose, model_id=6)
results[model_id] = res
model_id, res = predict_dnn(ws=ws, lr=lr, ep=ep, bi=True, dnnlayers=64, model_path=modelPath, df=df.copy(), df1=df1.copy(), verbose=verbose, model_id=7)
results[model_id] = res
model_id, res = predict_dnn(ws=ws, lr=lr, ep=ep, bi=False, dnnlayers=128, model_path=modelPath, df=df.copy(), df1=df1.copy(), verbose=verbose, model_id=8)
results[model_id] = res
model_id, res = predict_dnn(ws=ws, lr=lr, ep=ep, bi=True, dnnlayers=128, model_path=modelPath, df=df.copy(), df1=df1.copy(), verbose=verbose, model_id=9)
results[model_id] = res
for x in results.values():
    print(x['2023-01-11'])

print(datetime.now())
print(datetime.now()-start)
