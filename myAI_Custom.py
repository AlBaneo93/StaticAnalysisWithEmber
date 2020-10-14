#!/usr/bin/env python3

from collections import OrderedDict
import os
import pandas as pd
import myEmber as ember
import lightgbm as lgb
import numpy as np
import shutil as shu
import myExtractor as extractor
import json
import score
import argparse
# for ROC Curve
from sklearn import metrics
import matplotlib.pyplot as plt
# GET GPU State
import torch


class myClass:
    def __init__(self, dataDir=None, output=None, params=None, erroutput=None):
        self.dataDir = dataDir
        self.output = output
        self.params = params
        self.erroutput = erroutput
        self.is_optimize = False
        self.result = []

    def train(self):
        if self.is_optimize:
            print("begin Optimize")
            self.optimize(self.dataDir, self.params)

        self.model = ember.train_model(self.output, self.params)
        self.model.save_model(os.path.join(self.output, "model.dat"))

        print(" {:=^200} ".format('AI Model Created'))

    # 예측할 PE파일이 있는 경로
    # model.txt 파일이 있는 경로
    def predict(self, preDataDir, modelPath):
        if not os.path.exists(modelPath):
            raise Exception(
                "There is not a model in {} path".format(modelPath))

        model = lgb.Booster(self.params, model_file=modelPath)
        pre_result = []
        file_name = []
        err_cnt = 0
        err_list = []

        for root, dirs, files in os.walk(preDataDir):
            for file in files:
                if file.endswith("vir"):
                    with open(os.path.join(root, file), "rb") as bin:
                        try:
                            tmp = ember.predict_sample(model, bin.read())
                            pre_result.append(
                                tmp[0]
                            )
                            file_name.append(file)
                        except Exception as e:
                            # print("Error Occured {}".format(e))
                            err_cnt += 1
                            err_list.append(file)
                            raise Exception(e)

        # get result, threshold
        self.result = np.where(np.array(pre_result) > 0.5, 1, 0)

        print("prediction finished")
        # result save
        series = OrderedDict([("hash", file_name), ("label", self.result)])
        res = pd.DataFrame.from_dict(series)
        res.to_csv(os.path.join(self.output, "result.csv"),
                   index=False, header=["hash", "label"])

        with open(os.path.join(self.output, "error.txt"), "w", encoding="utf-8") as f:
            for idx, line in enumerate(err_list):
                f.writelines("{}, {}".format(idx, line))
        return self.getSCore(os.path.join(self.output, "result.csv"), os.path.join(preDataDir, "label.csv"))

    # jsonl파일들을 가지고 feature vector를 만든다
    def preProcessing(self):
        ember.create_vectorized_features(self.dataDir, 2)
        shu.move(os.path.join(self.dataDir, "X.dat"), self.output)
        shu.move(os.path.join(self.dataDir, "y.dat"), self.output)
        print('Created the data in {}'.format(self.output))

    # make feature.jsonl files
    def make_feaures(self, output):
        """
        : param dataDir : 특징을 추출해낼 vir파일의 경로
        : param output : 만들어진 feature를 저장할 경로
         """
        for root, dirs, files in os.walk(self.dataDir):
            for directory in dirs:
                base = os.path.join(root, directory)
                # if os.path.exists(os.path.join(base, "{}_feature.jsonl".format(base.split("/")[-1]))):
                #     os.remove(os.path.join(
                #         base, "{}_feature.jsonl".format(base.split("/")[-1])))
                extractor.Extractor(base, os.path.join(
                    base, "label.csv"), os.path.join(base, "{}_feature.jsonl".format(base.split("/")[-1])), self.erroutput).run()

        print("make features in {}".format(output))

    # optimize hparams
    def optimize(self, dataDir, params):
        """
        : param dataDir : X.dat, y.dat가 있는 폴더
         """
        params = ember.optimize_model(dataDir, params)
        with open(os.path.join(self.output, 'best_params.txt'), 'w', encoding='utf-8') as f:
            print("{:=^200}".format("optimize data saved"))
            f.write(params)
        print("Best Parameters: ")
        print(json.dumps(params, indent=2))

    # TODO
    def make_ROC(self, path, score):
        tmp = {}
        with open(path, 'r', encoding='utf-8') as f:
            tmp = {line.split(",")[0].strip(): int(line.split(",")[1].strip())
                   for line in f.readlines()}

        # np parsing
        labels = np.array([tmp[item] for item in sorted(tmp.keys())])
        score = np.array(score)
        # roc_curve(test, score)
        fpr, tpr, thresholds = metrics.roc_curve(labels, score)
        roc_auc = metrics.auc(fpr, tpr)
        print("roc_auc value :", roc_auc)
        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Model ROC Curve')
        # plt.legend(loc="lower right")
        plt.show()
        roc_curve_path = os.path.join(self.output, 'prediction_result.png')
        plt.savefig(roc_curve_path, dpi=plt.gcf().dpi)
        print("{:=^150}".format("save graph in "+roc_curve_path))

    def getSCore(self, resultPath, answerPath):
        return score.Score(resultPath, answerPath).run()

    def run(self, type):
        if type == "t":
            self.train()
        elif type == "p":
            self.predict()
        else:
            self.preProcessing


# parse the hyper parameter
def parse(params):
    ret = {}
    if params:
        for item in params.split(","):
            aa = item.strip().split("=")
            ret.update({aa[0].strip(): aa[1].strip()})

    return ret


def printResult(TP, TN, FP, FN, overDetection, missDetection, correction):
    print("{:=^150}".format("PREDICTION RESULT"))
    print("TP : {} TN : {} FP : {} FN : {}".format(TP, TN, FP, FN))
    print("overDetection : {} missDetection : {} correction : {}".format(
        overDetection, missDetection, correction))
    print("{:=^150}".format(""))


if __name__ == "__main__":
    # 현재 경로로 작업 디렉토리 변경
    os.chdir(os.getcwd())

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, default="./data", type=str)
    parser.add_argument("--out", required=True, default="./output", type=str)
    parser.add_argument("--pre", type=str)
    parser.add_argument("--param", type=str)
    parser.add_argument("--opti", type=bool, default=False)
    args = parser.parse_args()

    num_gpus = torch.cuda.device_count()

    # add hyper parameter in lgbm
    # See : https://lightgbm.readthedocs.io/en/latest/Parameters.html

    params = parse(args.param)
    params.update({"application": "binary",
                   "device": "gpu", "num_gpu": num_gpus})

    ai = myClass(dataDir=args.data, output=args.out, params=params,
                 erroutput=os.path.join(args.out, "error_output.txt"))
    # argparse의 subparse를 이용하여 prediction과 training 분기 처리
    try:
        # ai.make_feaures(os.path.join(args.out, "/features"))
        # ai.preProcessing()
        ai.train()
        TP, TN, FP, FN, overDetection, missDetection, correction = ai.predict(
            preDataDir="/home/cs206869/tmp/predict/2020_01", modelPath=os.path.join(args.out, "model.dat"))
        # printResult(TP, TN, FP, FN, overDetection, missDetection, correction)
        ai.make_ROC(os.path.join(
            "/home/cs206869/tmp/predict/2020_01", "label.csv"), ai.result)
    except Exception as e:
        # print("Error occured while running {} process".format())
        print("Error Message : {}".format(e))
