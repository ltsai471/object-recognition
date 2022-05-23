from flask import Flask, request, Response
from flask.views import MethodView
import urllib
import numpy as np
import cv2
import json
import sys
import ObjectRecognition

app = Flask(__name__)
obrModel = ObjectRecognition.ObjectRecogintion()


class ObjectAPI(MethodView):
    def get(self):
        url2 = "https://media.decathlon.tw/media/catalog/product/7/3/737c48d1-1bda-4bb6-b170-ed52d114d22c_8519380.jpg"
        resp2 = urllib.request.urlopen(url2)
        image2 = np.asarray(bytearray(resp2.read()), dtype='uint8')
        np.set_printoptions(threshold=sys.maxsize)
        print(image2)
        imgPrediction = obrModel.predict(image2)
        print(imgPrediction)
        # return Response(
        #     json.dumps({'imagePrediction': imgPrediction['classname']}),
        #     mimetype='application/json'
        # )
        return Response(
            json.dumps({'imgResult': imgPrediction['imgResult'],
                        'imgDetail': imgPrediction['classname']}),
            mimetype='application/json'
        )

    def post(self):
        req = request.get_json()
        imgType = req['imgType']
        imageData = req['image']

        # transform img format to 3d array
        try:
            # image = np.asarray(bytearray(imageData), dtype='uint8')
            imgPrediction = obrModel.predict(imageData)  # activate obr model
            print(imgPrediction)
            # return Response(
            #     json.dumps({'imgResult': imgPrediction['imgResult'],
            #                 'imgDetail': imgPrediction['classname']}),
            #     mimetype='application/json'
            # )
            return Response(
                json.dumps({'imgResult': imgPrediction['imgResult'],
                            'imgDetail': imgPrediction['classname'],
                             'itemTypeLevel2Id': imgPrediction['imgResult']}),
                mimetype='application/json'
            )
        except:
            return Response(
                json.dumps({'imgResult': '0',
                            'imgDetail': '',
                            'itemTypeLevel2Id': '0'}),
                mimetype='application/json'
            )


app.add_url_rule('/objectRecognition/',
                 view_func=ObjectAPI.as_view('objectRecognition'))

if __name__ == '__main__':
    app.run(debug=True)
