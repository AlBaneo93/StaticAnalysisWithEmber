import re


class Score:
    def __init__(self, resultPath, answerPath):
        self.resultPath = resultPath
        self.answerPath = answerPath
        self.answer_dict = []

    def content_to_dict(self, lines):
        return {line.split(",")[0]: line.split(",")[1] for line in lines}

    def run(self):
        result = self.content_to_dict(
            open(self.resultPath, "r", encoding="utf-8").readlines())
        answer = self.content_to_dict(
            open(self.answerPath, "r", encoding="utf-8").readlines())

        result_orderd_key_list = sorted(result.keys())
        answer_orderd_key_list = sorted(answer.keys())

        correct = 0
        incorrect = 0
        total = len(result_orderd_key_list)
        TP = 0
        TN = 0
        FP = 0
        FN = 0

        rnum = len(result_orderd_key_list)
        anum = len(answer_orderd_key_list)
        # 비교
        for idx in range(min(rnum, anum)):

            rkey = result_orderd_key_list[idx]
            akey = answer_orderd_key_list[idx]
            rval = int(answer[akey])
            aval = int(result[rkey])

            if rval == aval:
                if aval == 1 and rval == 1:
                    # TP
                    TP += 1
                elif aval == 0 and rval == 0:
                    # TN
                    TN += 1
            else:
                if aval == 0 and rval == 1:
                    FP += 1
                elif aval == 1 and rval == 0:
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
