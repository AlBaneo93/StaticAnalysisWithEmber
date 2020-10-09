"""
This python module refer to Ember Porject(https://github.com/endgameinc/ember.git)
"""
import os
import sys
from feature import PEFeatureExtractor
import tqdm
import jsonlines
import pandas as pd
import multiprocessing


class Extractor:
    def __init__(self, datadir, label, output, erroutput):
        self.datadir = datadir
        self.output = output
        self.erroutput = erroutput
        self.data = pd.read_csv(label, names=["hash", "y"])
        self.features = PEFeatureExtractor()

    def extract_features(self, sample):
        """
        Extract features.
        If error is occured, return None Object
        """
        extractor = PEFeatureExtractor(self.features)
        fullpath = os.path.join(os.path.join(self.datadir, sample))
        try:
            binary = open(fullpath, "rb").read()

            feature = extractor.raw_features(binary)
            feature.update({"sha256": sample})  # sample name(hash)
            feature.update(
                {"label": self.data[self.data.hash == sample].values[0][1]}
            )  # label

        except KeyboardInterrupt:
            sys.exit()
        except Exception as e:
            # print("errror exception\n")
            ''' file name extract when error occured while feature extract from pe binary '''
            with open(self.erroutput, "a", encoding="utf-8") as f:
                if sample.endswith('.vir'):
                    f.writelines(sample)
                    f.writelines("\n")
            return None

        return feature

    def extract_unpack(self, args):
        """
        Pass thorugh function unpacking arguments
        """
        return self.extract_features(args)

    def extractor_multiprocess(self):
        """
        Ready to do multi Process
        Note that total variable in tqdm.tqdm should be revised
        Currently, I think that It is not safely. Because, multiprocess pool try to do FILE I/O.
        """
        pool = multiprocessing.Pool()
        queue = multiprocessing.Queue()
        queue.put("safe")
        end = len(next(os.walk(self.datadir))[2])
        error = 0

        extractor_iterator = ((directory)
                              for directory in os.listdir(self.datadir))
        with jsonlines.open(self.output, "w") as f:
            for x in tqdm.tqdm(
                pool.imap_unordered(self.extract_unpack, extractor_iterator), total=end
            ):
                if not x:
                    """
                    To input error class or function
                    """
                    error += 1
                    continue
                msg = queue.get()
                if msg == "safe":
                    f.write(x)
                    queue.put("safe")

        pool.close()

    def run(self):
        self.extractor_multiprocess()
