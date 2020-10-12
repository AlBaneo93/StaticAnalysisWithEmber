import os

"""
percentage = result / answer * 100
3번째 자리에서 반올림
"""


class Score:
    def __init__(self, resultPath, answeresultPath):
        self.resultPath = resultPath
        self.answerPath = answeresultPath

    def content_to_dict(self, file):
        dict = {}
        lines = file.readlines()
        for idx in range(len(lines)):
            data = lines[idx].split(",")
            dict[data[0]] = data[1]
        return dict

    def run(self):
        result = self.content_to_dict(
            open(self.resultPath, "r", encoding="utf-8"))
        answer = self.content_to_dict(
            open(self.answerPath, "r", encoding="utf-8"))

        result_orderd_key_list = sorted(result.keys())
        answer_orderd_key_list = sorted(answer.keys())

        correct = 0
        incorrect = 0
        total = len(result_orderd_key_list)
        TP = 0
        TN = 0
        FP = 0
        FN = 0

        # 비교
        for idx in range(len(result_orderd_key_list)):
            rkey = result_orderd_key_list[idx]
            akey = answer_orderd_key_list[idx]

            if result[rkey] == answer[akey]:
                if answer[akey] == 1 and result[rkey] == 1:
                    # TP
                    TP += 1
                elif answer[akey] == 0 and result[rkey] == 0:
                    # TN
                    TN += 1
            else:
                if answer[akey] == 0 and result[rkey] == 1:
                    FP += 1
                elif answer[akey] == 1 and result[rkey] == 0:
                    FN += 1

        overDetection = 0
        missDetection = 0
        correction = 0

        try:
            overDetection = FP / (FP + TN) * 100
        except ZeroDivisionError:
            overDetection = 0
        try:
            missDetection = FN / (FN + TP) * 100
        except ZeroDivisionError:
            missDetection = 0
        try:
            correction = (TP + TN) / (TP + TN + FP + FN) * 100
        except ZeroDivisionError:
            correction = 0

        return TP, TN, FP, FN, overDetection, missDetection, correction
# print("정탐률 : ", correction, ", 과탐률 : ",    overDetection, ", 미탐률 : ", missDetection)
