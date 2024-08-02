import time
import replicate
import supervision as sv
import cv2 
import numpy as np
import json
from nltk.tokenize import word_tokenize
#nltk.download('punkt')  # Ensure the tokenizer is downloaded
from tabulate import tabulate
from functions import calculate_iou, calculate_precision_recall, drawing_boxes,calculate_predi_time, get_single_image_results, help_iou_more_coordinates,intersection_over_union
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

client = replicate.Client()

#img = "gt_bordny9_ny1.jpg"
#img = "gt_bordny8_ny.jpg"
# img = "gt_ordning3.jpg"
img = "gt_ordning11.jpg"
temp_img = cv2.imread(img)
image_height, image_width = temp_img.shape[0], temp_img.shape[1]

start_describ = time.time()             #start of the time before reading the picture
input_image = open(f"{img}", "rb")
f = open('LLaVA_data.txt', 'a')

#version 2
query = (
    "Describe in detail the main objects present in the picture."
    "ensure that each object is reported only once."
    "Provide the result in a JSON format where each object has 'name' keys and the JSON object has the name 'objects'. Do not write the word 'json' after ``` at the beginning of the json format"
)

f.write("Query: " + query + "\n")
#temperature: 10, 30 och 60%
input_data1={
    "image": input_image,
    "top_p": 1,
    "prompt": query,
    "history": [],
    "max_tokens": 1024,
    "temperature": 0.60
}
temper = input_data1["temperature"]

output = replicate.run(
    "yorickvp/llava-v1.6-vicuna-13b:0603dec596080fa084e26f0ae6d605fc5788ed2b1a0358cd25010619487eae63",
    input=input_data1
)


res = "".join(output)
end_describ = time.time()
tot = end_describ - start_describ           #calculating the time that takes describe the picture

print("\n",res,"\n")
####========================= Extracting the objects in JSON format ===============######
new_res = res.replace('json', '').replace("```", "")
print(f"\n {new_res}")
with open("sample.json", "w") as file:
    file.write(new_res)

with open("sample.json", "r") as in_file:
    json_object = json.load(in_file)

print(json_object)
####======================== End of extracting the objects in JSON format ==========#####
f.write("text provided by the modell:\n" + repr(json_object) + "\n" + "temperature: "+ str(temper)+"\n")

print(type(json_object))
print(json_object['objects'])

#bordny8_ny
gt = [[197, 323, 362, 485], [78, 392, 116, 535], [43, 415, 441, 623], [0, 2, 457, 175], [90, 78, 195, 156], [217, 83, 323, 161]]  

####====== CALCULATING THE COST OF THE GPU with REPLICATE TIME =====#####
Total_cost,Predic_time = calculate_predi_time("0603dec596080fa084e26f0ae6d605fc5788ed2b1a0358cd25010619487eae63",input_data1,0.000725)
###======= END OF CALCULATING THE COST OF THE GPU with REPLICATE TIME =======### 


f.write(f"image: {img}, below are the measures for this image using the model LLaVA")
#f.write("\nReference text provided by the user:\n" + reference +"\n")
f.write("\n")
#f.write("Candidate text provided by the modell:\n" + candidate + "\n")

mydata = [
    ["Predict time usage of the Nvidia A40 (Large) GPU hardware (LLaVA)", f"{Predic_time} seconds"],
    ["Cost of the usage Nvidia A40 (Large) GPU hardware (LLaVA)", f"{Predic_time} * 0.000725/s = {Total_cost} dollar"],
    ["time to describe the scen: ", f"{tot}"]
]
 
# create header
head = [f"{img}","LLaVA"]
print(tabulate(mydata, headers=head, tablefmt="grid"))
f.write(tabulate(mydata, headers=head, tablefmt="grid"))


f.close()
