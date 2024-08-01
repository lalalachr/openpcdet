import pickle

with open('/root/OpenPCDet/output/custom_models/pointpillar/default/eval/epoch_300/test/default/result.pkl', 'rb') as f:
    data = pickle.load(f)

print(data)