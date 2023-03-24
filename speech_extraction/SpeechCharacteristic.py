characteristics = (
    "NN",
    "RB",
    "PRP",
    "VBP",
    "JJ",
    "TO",
    "VB",
    "DT",
    "NNS",
    "IN",
    "WRB",
    "VBD",
    "VBN",
    "RP",
    "CC",
    "VBG",
    "JJR",
    "RBR",
    "WDT",
    "MD",
    "VBZ",
    "WP",
    "EX",
    "PRP",
    "CD",
    "PDT",
    "JJS",
    "POS",
    "FW",
    "RBS",
    "NNP",
)


class SpeechCharacteristic:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.words_in_phrase = 0.0
        self.sentences_in_speech = 0.0
        self.letter_in_words = 0.0
        self.Noun = 0.0
        self.RB = 0.0
        self.PRP = 0.0
        self.VBP = 0.0
        self.JJ = 0.0
        self.TO = 0.0
        self.VB = 0.0
        self.DT = 0.0
        self.NNS = 0.0
        self.IN = 0.0
        self.WRB = 0.0
        self.VBD = 0.0
        self.NN = 0.0
        self.VBN = 0.0
        self.RP = 0.0
        self.CC = 0.0
        self.VBG = 0.0
        self.JJR = 0.0
        self.RBR = 0.0
        self.WDT = 0.0
        self.MD = 0.0
        self.VBZ = 0.0
        self.WP = 0.0
        self.EX = 0.0
        self.PRP = 0.0
        self.CD = 0.0
        self.PDT = 0.0
        self.JJS = 0.0
        self.POS = 0.0
        self.FW = 0.0
        self.RBS = 0.0
        self.NNP = 0.0
        self.punctuation = 0.0
