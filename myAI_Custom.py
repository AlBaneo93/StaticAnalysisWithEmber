from collections import OrderedDict
import os
import pandas as pd
import myEmber as ember
import lightgbm as lgb
import numpy as np
import shutil as shu
import myExtractor as extractor
import json


class myClass:
    def __init__(self, dataDir=None, output="./output", params={"application": "binary"}):
        self.dataDir = dataDir
        self.params = params
        self.ouput = output

    def train(self):
        self.model = ember.train_model("/DATA/2017_01", self.params)
        self.model.save_model(os.path.join(self.ouput, "model.dat"))

        print(" {:=^200} ".format('AI Model Created'))

    # 예측할 PE파일이 있는 경로
    # model.txt 파일이 있는 경로
    def predict(self, preDataDir, modelPath):
        if not os.path.exists(modelPath):
            raise Exception("{} 경로에 모델이 없습니다".format(modelPath))

        model = lgb.Booster(model_file=modelPath)
        pre_result = []
        file_name = []
        err_cnt = 0
        err_list = []

        for file in os.walk(preDir)[2]:
            with open(os.path.abspath(file, "rb")).read() as bin:

                try:
                    pre_result.append(
                        ember.predict_sample(model, bin)
                    )
                    file_name.append(file)
                except Exception as e:
                    print("Error Occured {}".format(e))
                    err_cnt += 1
                    err_list.append(file)

        # get result
        pre_result = np.where(np.array(pre_result) > 0.5, 1, 0)

        # result save
        series = OrderedDict([("hash", file_name), ("label", pre_result)])
        res = pd.DataFrame.from_dict(series)
        res.to_csv(os.path.join(self.output, "result.csv"),
                   index=False, header=["hash", "label"])
        with open(os.path.join(self.output, "error.txt"), "w", encoding="utf-8") as f:
            for idx, line in enumerate(err_list):
                f.writelines("{}, {}".format(idx, line))

    # jsonl파일들을 가지고 feature vector를 만든다
    # def preProcessing(self, feature_dataDir, outputDir):
    def preProcessing(self):
        ember.create_vectorized_features(self.dataDir, 2)
        shu.move(os.path.join(self.dataDir, "X.dat"), self.ouput)
        shu.move(os.path.join(self.dataDir, "y.dat"), self.ouput)
        print('Created the data in {}'.format(self.ouput))

    # make feature.jsonl files
    def make_feaures(self, output):
        """
        : param dataDir : 특징을 추출해낼 vir파일의 경로
        : param output : 만들어진 feature를 저장할 경로
         """
        for root, dirs, files in os.walk(self.dataDir):
            for directory in dirs:
                base = os.path.join(root, directory)
                extractor.Extractor(base, os.path.join(
                    base, "label.csv"), os.path.join(base, "{}_feature.jsonl".format(os.path.dirname(directory)))).run()

        print("make features in {}".format(output))

    # optimize hparams
    def optimize(self, dataDir):
        """
        : param dataDir : X.dat, y.dat가 있는 폴더
         """
        params = ember.optimize_model(dataDir)
        print("Best Parameters: ")
        print(json.dumps(params, indent=2))
        print("optimization!")

    def run(self, type):
        if type == "t":
            self.train()
        elif type == "p":
            self.predict()
        else:
            self.preProcessing


if __name__ == "__main__":
    dataRoot = "/DATA"
    outputDirectory = "/home/cs206869/tmp/output"

    params = {}

    ai = myClass(dataDir=dataRoot, output=outputDirectory)
    try:
        ai.make_feaures(os.path.join(outputDirectory, "/features"))
        # ai.preProcessing()
        # ai.train()  # TODO : model create to use multi feature
        # ai.predict(preDataDir="", modelPath=os.path.join(outputDirectory, "model.dat"))

    except Exception as e:
        # print("Error occured while running {} process".format())
        print("Error Message\n{}".format(e))
