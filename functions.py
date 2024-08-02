from datetime import datetime
import re
import cv2
import replicate
import time
import numpy as np
def extract_coordinates(text):
    #This pattern matches coordinates in the form "0.123" and allows for variations in whitespace and the presence of "=", ",", or ":"
    pattern = r'[\s=:]+([\d\.]+),[\s]*([\d\.]+)[\s,]+[\s=:]+([\d\.]+),[\s]*([\d\.]+)'
    # Find all matches in the text
    matches = re.findall(pattern, text)
    # Convert each match from tuples of strings to a tuple of floats
    return [(float(x0), float(y0), float(x1), float(y1)) for x0, y0, x1, y1 in matches]

def calculate_predi_time(Version, input_data,price):
    client = replicate.Client()

    prediction = client.predictions.create(
        version=Version,
        input=input_data
    )
    print("Prediction created with details:", prediction)
    # Replace 'your_api_key_here' with your actual Replicate API key
    api_key = 'r8_X7BImFgGCsTYYNYiRsHNlipu0zf3pG12fITxR'

    prediction_id = prediction.id
    print(f"Prediction ID: {prediction_id}\n")
    print("type: ", type(prediction))
    # time.sleep(7) 
    # prediction_result = client.predictions.get(prediction_id)
    # print(prediction_result)


    while True:
        prediction_result = client.predictions.get(prediction_id)
        if prediction_result.status == 'succeeded':
            # Process the successful prediction
            break
        elif prediction_result.status == 'failed':
            # Handle the failure
            break
        else:
            # Wait some time before checking again
            time.sleep(5)

    # Check the status or result of your prediction
    if prediction_result.status == 'succeeded':
        result = prediction_result.metrics
        print(result, "\n")
        predic_time = prediction_result.metrics['predict_time']
        print(f"Predict Time: {predic_time} seconds")
        
    elif prediction_result.status == 'failed':
        error_message = prediction_result.error
        print(error_message)
    else:
        # If it's still running, you might want to check again later
        print("Prediction is still processing.")

    total_cost = predic_time * price
    return total_cost,predic_time


