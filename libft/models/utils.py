import pickle


def save_model(model, scaler=None, path='model.ft'):
    json = {'model': model, 'scaler': scaler}
    file = open(path, "wb")
    pickle.dump(json, file)
    file.close()


def load_model(path):
    file = open(path, 'rb')
    json = pickle.load(file)
    file.close()

    return json['model'], json['scaler']
