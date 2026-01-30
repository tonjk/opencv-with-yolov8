
# pip install inference-sdk
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
import cv2
import os
from dotenv import load_dotenv
load_dotenv()

# create an inference client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.getenv("ROBOFLOW_API_KEY")
    )

configuration = InferenceConfiguration(
    confidence_threshold=0.02,
    # iou_threshold=.45
    )
CLIENT.configure(configuration)

# run inference on a local image
results = CLIENT.infer(inference_input="images/Thailand-Trains-4.jpg",
                       model_id="crowd-detection-vajqw/1")

def run(image_path, save_path):
    # create dir if not exist
    os.makedirs(save_path, exist_ok=True)


    im = cv2.imread(image_path)
    # print(f"image shape: {im.shape}")
    results = CLIENT.infer(image_path, model_id="crowd-detection-vajqw/1")

    # draw bounding boxes
    for res in results['predictions']:
        # x1, y1, x2, y2 = int(res['x']), int(res['y']), int(res['x'] + res['width']), int(res['y'] + res['height'])
        x1, y1, x2, y2 = int(res['x']-res['height']/2), int(res['y']-res['width']/2), int(res['x'] + res['height']/2), int(res['y'] + res['width']/2)
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # save image
    cv2.imwrite(save_path + "/" + image_path.split("/")[-1], im)

    # return im

if __name__ == "__main__":
    run("images/ppl.jpg", "roboflow_results")
