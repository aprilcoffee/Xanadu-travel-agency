from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import service_pb2_grpc

stub = service_pb2_grpc.V2Stub(ClarifaiChannel.get_grpc_channel())

from clarifai_grpc.grpc.api import service_pb2, resources_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2


import api_key

YOUR_CLARIFAI_API_KEY = api_key.clarify_apikey()
YOUR_APPLICATION_ID = api_key.clarify_APPLICATION_ID()


SAMPLE_URL = "https://live.staticflickr.com/7372/12502775644_acfd415fa7_w.jpg"

# This is how you authenticate.
metadata = (("authorization", f"Key {YOUR_CLARIFAI_API_KEY}"),)

#for i in range(10):
#t = 'http://167.235.56.4:5000/static/landscape/image'+str(i)+'.jpg'
#SAMPLE_URL = t

request = service_pb2.PostModelOutputsRequest(
    # This is the model ID of a publicly available General model. You may use any other public or custom model ID.
    model_id="general-image-recognition",
    user_app_id=resources_pb2.UserAppIDSet(app_id=YOUR_APPLICATION_ID),
    inputs=[
        resources_pb2.Input(
            data=resources_pb2.Data(image=resources_pb2.Image(url=SAMPLE_URL))
        )
    ],
)
response = stub.PostModelOutputs(request, metadata=metadata)

if response.status.code != status_code_pb2.SUCCESS:
    print(response)
    raise Exception(f"Request failed, status code: {response.status}")

for concept in response.outputs[0].data.concepts:
    print("%12s: %.2f" % (concept.name, concept.value))
