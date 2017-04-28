import os
import pickle
from glob import glob


def create_if_need(path):
    if not os.path.exists(path):
        os.makedirs(path)


def unpickle_data(path):
    with open(path, "rb") as fout:
        data = pickle.load(fout)
        return data


def pickle_data(data, path):
    with open(path, "wb") as fout:
        pickle.dump(data, fout)


def masked_files(mask):
    if isinstance(mask, list) and all(["*" in x for x in mask]):
        files = [file for file_mask in mask for file in masked_files(file_mask)]
    elif isinstance(mask, str):
        files = glob(mask)
    else:
        raise Exception("Unknown instance")
    return sorted(files)


def masked_if_need(mask):
    if isinstance(mask, list) and all(["*" not in x for x in mask]):
        return mask
    return masked_files(mask)


def save_history(history, out_dir):
    pickle_data(history, os.path.join(out_dir, "history.pkl"))


def save_model(sess, saver, save_dir, model_global_step=None):
    create_if_need(save_dir)
    save_path = os.path.join(save_dir, "model.cpkl")
    saver.save(sess, save_path, global_step=model_global_step)
