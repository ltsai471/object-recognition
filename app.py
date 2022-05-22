from flask import Flask, jsonify, request, Response
from flask.views import MethodView
import urllib
import numpy as np
import cv2
import json
import sys
# from . import ocr
import ObjectRecognition

app = Flask(__name__)
obrModel = ObjectRecognition.ObjectRecogintion()


class ObjectAPI(MethodView):
    def get(self):
        # url1 = "https://9.share.photo.xuite.net/huanting77/193b7e8/14659719/776948065_m.jpg"
        # resp = urllib.request.urlopen(url1)
        # image = np.asarray(bytearray(resp.read()), dtype='uint8')
        # image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        # ocrResult = ocr.studentIdRecognition(image)
        # print(ocrResult)
        url2 = "https://media.decathlon.tw/media/catalog/product/7/3/737c48d1-1bda-4bb6-b170-ed52d114d22c_8519380.jpg"
        resp2 = urllib.request.urlopen(url2)
        image2 = np.asarray(bytearray(resp2.read()), dtype='uint8')
        np.set_printoptions(threshold=sys.maxsize)
        print(image2)
        imgPrediction = obrModel.predict(image2)
        # return Response({'image':image2, 'ocrResult': '', 'imagePrediction': imgPrediction['classname']})
        return Response(
            json.dumps({'ocrResult': '',
                        'imagePrediction': imgPrediction['classname']}),
            mimetype='application/json'
        )

    def post(self):
        req = request.get_json()
        imgType = req['imgType']
        imageData = req['imageData']
        try:
            image = np.asarray(bytearray(imageData), dtype='uint8')
            # if imgType == 1:
            #     # img = student ID Card
            #     image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            #     imgPrediction = ocr.studentIdRecognition(
            #         image)  # activate ocr function
            #     print(imgPrediction)
            #     return Response({'imgResult': imgPrediction})
            # else:  # img = object image

            imgPrediction = obrModel.predict(image)  # activate obr model
            print(imgPrediction)
            return Response(
                json.dumps({'imgResult': imgPrediction['classname']}),
                mimetype='application/json'
            )
        except:
            return Response(
                json.dumps({'imgResult': '0'}),
                mimetype='application/json'
            )

# @api_view(['POST'])
# def imageRecognition(request):
#     if request.method == 'POST':
#         imgType = request.data['imgType']
#         imageData = request.data['image']
#         # transform img format to 3d array
#         image = np.asarray(bytearray(imageData), dtype='uint8')
#         if imgType == 1:
#             # img = student ID Card
#             image = cv2.imdecode(image, cv2.IMREAD_COLOR)
#             imgPrediction = ocr.studentIdRecognition(
#                 image)  # activate ocr function
#             print(imgPrediction)
#             return Response({'imgResult': imgPrediction})
#         else:  # img = object image
#             imgPrediction = obrModel.predict(image)  # activate obr model
#             print(imgPrediction)
#             return Response({'imgResult': imgPrediction['imgResult'], 'imgDetail': imgPrediction['classname'],
#                             'itemTypeLevel1Id': imgPrediction['itemTypeLevel1Id']})


# @api_view(['GET'])
# def testimageRecognition(request):  # 測試傳入影像array，啟用ocr與影像辨識功能
#     if request.method == 'GET':
#         url1 = "https://9.share.photo.xuite.net/huanting77/193b7e8/14659719/776948065_m.jpg"
#         resp = urllib.request.urlopen(url1)
#         image = np.asarray(bytearray(resp.read()), dtype='uint8')
#         image = cv2.imdecode(image, cv2.IMREAD_COLOR)
#         ocrResult = ocr.studentIdRecognition(image)
#         print(ocrResult)
#         url2 = "https://media.decathlon.tw/media/catalog/product/7/3/737c48d1-1bda-4bb6-b170-ed52d114d22c_8519380.jpg"
#         resp2 = urllib.request.urlopen(url2)
#         image2 = np.asarray(bytearray(resp2.read()), dtype='uint8')
#         imgPrediction = obrModel.predict(image2)
#         print(imgPrediction)
#         return Response({'ocrResult': ocrResult, 'imagePrediction': imgPrediction['classname']})

app.add_url_rule('/objectRecognition/',
                 view_func=ObjectAPI.as_view('objectRecognition'))

if __name__ == '__main__':
    app.run(debug=True)
