import urllib.request
import numpy as np
from six import BytesIO
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub

LabelMap = {
    0: "unlabeled",
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic",
    11: "fire",
    12: "street",
    13: "stop",
    14: "parking",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    26: "hat",
    27: "backpack",
    28: "umbrella",
    29: "shoe",
    30: "eye",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports",
    38: "kite",
    39: "baseball",
    40: "baseball",
    41: "skateboard",
    42: "surfboard",
    43: "tennis",
    44: "bottle",
    45: "plate",
    46: "wine",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted",
    65: "bed",
    66: "mirror",
    67: "dining",
    68: "window",
    69: "desk",
    70: "toilet",
    71: "door",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    83: "blender",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy",
    89: "hair",
    90: "toothbrush",
    91: "hair",
    92: "banner",
    93: "blanket",
    94: "branch",
    95: "bridge",
    96: "building",
    97: "bush",
    98: "cabinet",
    99: "cage",
    100: "cardboard",
    101: "carpet",
    102: "ceiling",
    103: "ceiling",
    104: "cloth",
    105: "clothes",
    106: "clouds",
    107: "counter",
    108: "cupboard",
    109: "curtain",
    110: "desk",
    111: "dirt",
    112: "door",
    113: "fence",
    114: "floor",
    115: "floor",
    116: "floor",
    117: "floor",
    118: "floor",
    119: "flower",
    120: "fog",
    121: "food",
    122: "fruit",
    123: "furniture",
    124: "grass",
    125: "gravel",
    126: "ground",
    127: "hill",
    128: "house",
    129: "leaves",
    130: "light",
    131: "mat",
    132: "metal",
    133: "mirror",
    134: "moss",
    135: "mountain",
    136: "mud",
    137: "napkin",
    138: "net",
    139: "paper",
    140: "pavement",
    141: "pillow",
    142: "plant",
    143: "plastic",
    144: "platform",
    145: "playingfield",
    146: "railing",
    147: "railroad",
    148: "river",
    149: "road",
    150: "rock",
    151: "roof",
    152: "rug",
    153: "salad",
    154: "sand",
    155: "sea",
    156: "shelf",
    157: "sky",
    158: "skyscraper",
    159: "snow",
    160: "solid",
    161: "stairs",
    162: "stone",
    163: "straw",
    164: "structural",
    165: "table",
    166: "tent",
    167: "textile",
    168: "towel",
    169: "tree",
    170: "vegetable",
    171: "wall",
    172: "wall",
    173: "wall",
    174: "wall",
    175: "wall",
    176: "wall",
    177: "wall",
    178: "water",
    179: "waterdrops",
    180: "window",
    181: "window",
    182: "wood"
}
inclusionList = [
    "unlabeled", "shoe", "hat", "backpack", "umbrella",
    "tennis", "bottle", "book",
    "cup", "laptop", "cell phone", "cloth",
    # "scissors",
    # "paper", "mirror", "towel", "mouse", "keyboard", "skateboard", "handbag", "tie", "baseball", "bicycle"
]
numberObjectMap = {
    "unlabeled": 27, "bicycle": 26, "shoe": 12, "hat": 13, "backpack": 15, "umbrella": 28,
    "handbag": 16, "tie": 17, "baseball": 20, "skateboard": 21, "tennis": 22, "bottle": 19, "book": 4,
    "cup": 23, "laptop": 7, "mouse": 8, "keyboard": 9, "cell phone": 6, "scissors": 18, "cloth": 14,
    "paper": 5, "mirror": 25
}


def loadImageIntoNumpyArray(imageData):

    image = Image.open(BytesIO(imageData))
    (imWidth, imHeight) = image.size
    return np.array(image.getdata()).reshape(
        (1, imHeight, imWidth, 3)).astype(np.uint8)


class ObjectRecogintion:

    def __init__(self):
        # 'CenterNet HourGlass104 Keypoints 512x512'
        self.hubModel = hub.load(
            "https://tfhub.dev/tensorflow/centernet/hourglass_512x512/1")
        return

    def predict(self, imageData):
        """
        return {"classname": class_name, "score": score }
        Top Prediction
        """
        imageNp = loadImageIntoNumpyArray(imageData)
        results = self.hubModel(imageNp)
        result = {key: value.numpy() for key, value in results.items()}
        itemTypeLevel2Id = numberObjectMap["unlabeled"]
        # itemTypeLevel2=ItemTypeLevel2.objects.get(level2Id=itemTypeLevel2Id)
        # prediction = {"classname": "unlabeled", "score": 0,"itemTypeLevel1Id":itemTypeLevel2.level1Id.level1Id,"imgResult":itemTypeLevel2Id}
        prediction = {"classname": "unlabeled",
                      "score": 0, "imgResult": itemTypeLevel2Id}
        for i in range(result["detection_classes"][0].size):
            label = result["detection_classes"][0][i]
            score = result['detection_scores'][0][i]
            if LabelMap[label] in inclusionList and score > 0.7:
                itemTypeLevel2Id = numberObjectMap[LabelMap[label]]
                # itemTypeLevel2=ItemTypeLevel2.objects.get(level2Id=itemTypeLevel2Id)
                # prediction = {"classname": LabelMap[label], "score": score,"itemTypeLevel1Id":itemTypeLevel2.level1Id.level1Id,"imgResult":itemTypeLevel2Id}
                prediction = {
                    "classname": LabelMap[label], "score": score, "imgResult": itemTypeLevel2Id}
                return prediction
        return prediction


def test():
    # read the image url
    url = 'https://shop.r10s.jp/komiyakasa/cabinet/komiyashoten/women/kasane/imgrc0079149018.jpg'
    resp = urllib.request.urlopen(url)
    # read image as an numpy array
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    obr = ObjectRecogintion()
    prediction = obr.predict(image)

    print(prediction)
    return
