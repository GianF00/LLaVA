from matplotlib import pyplot as plt
import openai
import time
from openai import OpenAI
import os 
import json
import base64
import requests
import psutil
import nltk
import cv2
from nltk.tokenize import word_tokenize
from tabulate import tabulate
from functions import  drawing_boxes, extract_coordinates_gpt4,calculate_image_tokens, calculate_iou,calculate_cost
import replicate
#nltk.download('punkt')  # Ensure the tokenizer is downloaded
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# OpenAI API Key
api_key = "skriv din API nyckeln för att utföra förfrågningar" 

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

start = time.time()
# Path to your image
image_path = "ordning3.jpg"
#image_path = "bord1.jpg"
#image_path = "ordning4_ny.jpg"
#image_path = "ordning5.jpg"
input_image = open(f"{image_path}", "rb")

# Query
#query = "Describe the coordinates of the utensils inside the dish rack. Describe the coordinates x0,y0,x1,y1 of the pot lid, the blue bowl, the scissors, the dish rack, the shelf above the sink, the spatula and of the cheese slicer."
#ordning3
query = "Calculate the coordinates in pixels in the format x0,y0,x1,y1 of the white plate, the white bowl, the glass, the fork, the spoon, the knife, the dish rack, the shelf above the dish rack, the white coffe mugg in the shelf and the blue coffe mugg in the shelf. Present the result like this: x0,y0,x1,y1 new line next coordinate new line and so on."

#ordning4_ny
#query = "Calculate the coordinates in pixels in the format x0,y0,x1,y1 of the pot lid, the blue bowl, the scissors, the dish rack, the shelf above the sink, the spatula and of the cheese slicer. Present the result like this: x0,y0,x1,y1 new line next coordinate new line and so on."

#ordning5
#query = "Calculate the coordinates in pixels in the format x0,y0,x1,y1 of the blue bowl, the scissors, the dish rack, the shelf above the sink, the spoon, the fork, spaghetti scoop, the grater. Present the result like this: x0,y0,x1,y1 new line next coordinate new line and so on."

#query = "Calculate the coordinates in pixels in the format x0,y0,x1,y1 of the spoon, the yellow paper, the pen, the pencil, the eraser and the glasses. Present the result like this: x0,y0,x1,y1 new line next coordinate new line and so on."

# Getting the base64 string
base64_image = encode_image(image_path)

headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {api_key}"
}
payload = {
  "model": "gpt-4-turbo",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": query
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}"
          }
        }
      ]
    }
  ],
  "max_tokens": 300
}

# imagen ordning 3
reference = "The image shows a kitchen scen where there is a dish drainer. Inside the dish drainer there are a white plate, a glass, a white bowl, two forks a spoon and a knife.\
              \nOn top of the dish drainer there is a white shelf with a white coffe mug and a blue coffe mug. The wall of the kitchen is made of white majolica."  # Correct answer

#imagen ordning4
#reference = "The image shows a kitchen scen where there is a dish rack. In the white dish rack there is a blue bowl, a silver pot lid, white scissors, a spatula and a cheese slicer. On top of the dish rack there is an empty white shelf"
# Reference ordning5:
#reference = "The image shows a kitchen scen where there is a dish rack. In the white dish rack there is the blue bowl, the scissors, the dish rack, the shelf above the sink, the spoon, the fork, spaghetti scoop, the grater"

#reference = "In the white table we have a spoon, the yellow paper, the pen, the pencil, the eraser and the glasses"


f = open('GPT4_data.txt', 'a')

response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
print(response.json())
end = time.time()
temp_json = response.json()
#in candidate stores the text generated by the model
candidate = temp_json['choices'][0]['message']['content']
print("\nOutput message: ",candidate)

########================= getting the coordinates from the text ===================###########
pred_coordinates = extract_coordinates_gpt4(candidate)

# Print or use the extracted coordinates
print("\n")
for coord in pred_coordinates:
    print(coord)
print("\n")


# names = extract_names(candidate)
# print("list of names: ", names, "\n")
#############=================== end of getting the coordinates from the text ==========#####

####================ DRAWING THE GROUND TRUTH BOUNDING BOXES ==========############

img = cv2.imread(image_path)
numOfTimes = 10
labels = "objects"
coordinates = []
for _ in range(numOfTimes):
    x_val, y_val, w_val, h_val = drawing_boxes(labels,img)
    coordinates.append([x_val,y_val, (x_val + w_val), (y_val+h_val)])
    #print("Num of iter: ", x)

##Add two list together:
# result = coordinates[0] + coordinates[1]
# print("\n",result)

result_gt = []
sz = len(coordinates)
for x in range(sz):
    #merged_result = []

    # Add each element of list_float to itself and append to merged_result
    # for num in list_float:
    #     merged_result.append(num + num)
    result_gt.extend(coordinates[x])
    # Append the merged_result to mrgd_list_float
    # mrgd_lis_float.append(merged_result)

print("\nresult ground truth bounding box",result_gt)
print("\nCoordinates: ", coordinates)
for coord in coordinates:
    temp = cv2.rectangle(img, (coord[0], coord[1]), (coord[2], coord[3]), (0, 255, 0), 2)

cv2.imshow("Ground truth bounding boxes", temp)
cv2.imwrite('gt_image.jpg',temp)      
cv2.waitKey(0)
#print(f"{x= }	{y= }	{w= }	{h= }")
#cv2.waitKey(0)

#rectangle DRAW THE BOUNDING BOXES BASED ON THE COORDINATES 
# img_with_gruthrubbox = cv2.rectangle(img, (x,y),(x + w,y + h), (0,255,0),2)
# cv2.imshow("with the box", img_with_gruthrubbox)
# cv2.waitKey(0)
cv2.destroyAllWindows()

####================ END DRAWING THE GROUND TRUTH BOUNDING BOXES ==========############



####================ DRAWING THE PREDICTED BOUNDING BOXES ==========############

# Load the image you want to draw on
img = cv2.imread('gt_image.jpg')  # Replace 'your_image.jpg' with your actual image file
# if img.shape[1] != 606 or img.shape[0] != 640:
#     img = cv2.resize(img, (606, 640))

# Iterate over the bounding box coordinates and draw each rectangle
pr = []
for bbox in pred_coordinates:
    pr.extend([bbox[0], bbox[1], bbox[2], bbox[3]])
    # Draw rectangle on the image
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)

# Display the image with rectangles
cv2.imshow('Image with Bounding Boxes', img)
cv2.imwrite('GPT4_ordning3_ny3_3.jpg',img) 
print("List of the predicted bboxes: ",pr)     
cv2.waitKey(0)
cv2.destroyAllWindows()

###=================== END OF DRAWING THE PREDICTED BBOXES ===========0#####

###=================== CALUCLATING THE IOU ===========0#####
# iou_results = help_iou_more_coordinates(result_gt,pr)
# print(f"iou: {iou_results}\n")
###=================== END CALUCLATING THE IOU ===========0#####

###============= CALCULATING THE PRECISION, RECALL, TP, FP, FN =====####
pr_boxes = [pr[i:i+4] for i in range(0, len(pr), 4)]
print("Predicted bboxes",pr_boxes, "\n")
true_positives = 0
false_positives = 0
iou_threshold = 0.60
matches = []
# matched_ground_truth = []  # Lista för att hålla reda på vilka GT-boxar som matchats
gt_matched = set()  # För att hålla reda på vilka GT-boxar som matchats

# Calculate True Positives and False Positives
for p_idx, p_box in enumerate(pr_boxes):
    match_found = False
    for gt_idx, gt_box in enumerate(coordinates):
        iou = calculate_iou(p_box, gt_box)
        if iou >= iou_threshold:
            if gt_idx not in gt_matched:
                true_positives += 1
                gt_matched.add(gt_idx)
                match_found = True

                # Draw bounding box on the image if IoU is greater than 0.80
                cv2.rectangle(img, (p_box[0], p_box[1]), (p_box[2], p_box[3]), (255, 0, 0), 2)                
                break

    if not match_found:
        false_positives += 1

# Calculate False Negatives
false_negatives = len(coordinates) - len(gt_matched)
precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

print(f"True Positives (TP): {true_positives}")
print(f"False Positives (FP): {false_positives}")
print(f"False Negatives (FN): {false_negatives}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"Matched ground truth: {gt_matched}")
cv2.imshow('Image with Bounding Boxes', img)
cv2.imwrite('GPT43.jpg',img) 
print("List of the predicted bboxes: ",pr)     
cv2.waitKey(0)
cv2.destroyAllWindows()
###============= END CALCULATING THE PRECISION, RECALL, TP, FP, FN =====####

####========= DRAWING THE DIAGRAM ===========####
metrics = ['True Positives', 'False Positives', 'False Negatives', 'Precision', 'Recall']
values = [true_positives, false_positives, false_negatives, precision, recall]

# Create bar chart
plt.figure(figsize=(10, 6))
plt.bar(metrics, values, color=['green', 'red', 'blue', 'purple', 'orange'])

# Adding the title and labels
plt.title('Object Detection Metrics')
plt.xlabel('Metrics')
plt.ylabel('Values / Scores')
for i, v in enumerate(values):
    plt.text(i, v + 0.01, f"{v:.2f}", ha='center')

# Show the plot
plt.tight_layout()
plt.show()
####========= DRAWING THE DIAGRAM ===========####

###============= CALCULATING THE AMOUNT OF THE TOKEN COST =====####
image_height, image_width = img.shape[0], img.shape[1]
print(calculate_image_tokens(image_width,image_height) , "Tokens")
tokens_img = calculate_image_tokens(image_width,image_height)
cost = calculate_cost(image_width,image_height)
###============= END OF CALCULATING THE AMOUNT OF THE TOKEN COST =====####




####========================Extractiong of the objects============###
with open('items.json', 'r') as file:
    data = json.load(file)
    item_list = data['items']

# Normalize the description to lower case to ensure case-insensitive matching
description_lower = candidate.lower()

# Find which items from the JSON list are mentioned in the description
found_items = [item for item in item_list if item in description_lower]

# Clean items to ensure proper formatting for API query
found_items_cleaned = [item.strip() for item in found_items]  # Remove extra spaces
found_items_query = ','.join(found_items_cleaned)
print("\nFound Items:", found_items_query)
print("Number of Items Found:", len(found_items))
print("Total Number of Items:", len(item_list))
#print("Found {} out of {} items.".format(len(found_items)))
#####======================ENd extraction of the objects=============###

#####=============DINO================#####
# output = replicate.run(
#     "adirik/grounding-dino:efd10a8ddc57ea28773327e881ce95e20cc1d734c589f7dd01d2036921ed78aa",    
#     input={
#         "image": input_image,
#         "query": found_items_query,
#         "box_threshold": 0.41,
#         "text_threshold": 0.26,
#         "show_visualisation": True,   
#     }
# )

#resultatet blir en JSON format med en länk som visar input bilden med resultatet
# print("\n",output)
###===============END OF DINO============####
time_exec = end - start
print("time of execution: ", time_exec )

########================ CALCULATING THE METEOR AND BLEU
# # Tokenize the reference and candidate
# tokenized_reference = word_tokenize(reference)
# tokenized_candidate = word_tokenize(candidate)

# # Calculate METEOR score
# score1 = meteor_score([tokenized_reference], tokenized_candidate)
# print(f"METEOR Score: {score1:.3f}\n")

# # Create a SmoothingFunction object
# chencherry = SmoothingFunction()

# # Now, let's calculate the BLEU score with smoothing
# bleu_score = sentence_bleu([reference], candidate, 
#                            smoothing_function=chencherry.method1)  # reference tokens must be a list of lists

# print(f"BLEU Score: {bleu_score}\n")

f.write(f"\nimage: {image_path}, below are the measures for this image using the model LLaVA\n")
f.write("\nQuery: " + query + "\n")
f.write("\nReference text provided by the user:\n" + reference +"\n")
f.write("Candidate text provided by the modell:\n" + candidate + "\n")
f.write("Execution time: " + repr(time_exec)+ "second\n")
#f.write("METEOR Score: "+ repr(score1)+ "\n")
#f.write("BLEU score: "+repr(bleu_score))

mydata = [
    ["Execution time", f"{time_exec}"], 
#    ["METEOR Score", f"{score1}"],
#    ["BLEU score", f"{bleu_score}"],
    ["Predicted coordinates from GPT4", f"{coordinates}"],
    #["IOU", f"{iou_results}"],
    ["Cost of token image", f"{tokens_img}"],
    ["Num of found items", f"{len(coordinates)}"],
    ["Precision", f"{precision}"],
    ["Recall", f"{recall}"],
    ["True positive", f"{true_positives}"],
    ["False positive", f"{false_positives}"],
    ["False negative", f"{false_negatives}"],
    ["Matched ground truth:", f"{gt_matched}"],
    ["iou threshold: ", f"{iou_threshold}"],
    ["cost of the picture: ", f"{cost}"]

]
 
# create header
head = [f"{image_path}","GPT4 Turbo Vision"]
print(tabulate(mydata, headers=head, tablefmt="grid"))
f.write(tabulate(mydata, headers=head, tablefmt="grid"))