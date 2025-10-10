import sys
from mslite_model import MSLiteModel

ARGS_SIZE = 6
ARG_INDEX_MODELS_PATH = 1
ARG_INDEX_SAVE_MODELS_PATH = 2
ARG_INDEX_CONFIG_FILES_PATH = 3
ARG_INDEX_MODELS_LIST_CONFIG = 4
ARG_INDEX_SO_PATH = 5
ARG_INDEX_DEVICE_ID = 6

if __name__ == '__main__':
    if len(sys.argv) < ARGS_SIZE:
        raise RuntimeError("args is wrong.")
    models_path = sys.argv[ARG_INDEX_MODELS_PATH]
    save_models_path = sys.argv[ARG_INDEX_SAVE_MODELS_PATH]
    config_files_path = sys.argv[ARG_INDEX_CONFIG_FILES_PATH]
    models_list_config = sys.argv[ARG_INDEX_MODELS_LIST_CONFIG]
    so_path = sys.argv[ARG_INDEX_SO_PATH]
    device_id = int(sys.argv[ARG_INDEX_DEVICE_ID])
    print("models_path: ", models_path)
    print("save_models_path: ", save_models_path)
    print("config_file_path: ", config_files_path)
    print("models_list_config: ", models_list_config)
    print("so_path: ", so_path)
    print("device_id: ", device_id)
    results = []
    with open(models_list_config) as f:
        for line in f:
            if line[0] == "#":
                continue
            if line[-1] == "\n":
                line = line[:-1]
            mslite_model = MSLiteModel(models_path, save_models_path, config_files_path, so_path, line, "ascend",
                                       device_id)
            try:
                mslite_model.Convert()
                mslite_model.Build()
                mslite_model.Predict()
            except:
                raise RuntimeError("model run failed.")
            results.append(mslite_model.GetPredictResult())
    print("=" * 150)
    print(f"{'Model Name': <30} {'Build Time(ms)': <30} {'Predict Time(ms)': <30} {'Accuracy(%)': <30} {'Result': <30}")
    for res in results:
        print("{:<30}".format(res["model_name"]), "{:<30}".format(res["build_time"]),
              "{:<30}".format(res["predict_time"]), "{:<30}".format(res["accuracy_threshold"]),
              "{:<30}".format(res["result"]))
    print("=" * 150)
