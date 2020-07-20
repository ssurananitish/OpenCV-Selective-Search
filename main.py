# python main.py --image ".\Input_Images\dog.jpg" 
# python main.py --image ".\Input_Images\dog.jpg" --method="quality"

import argparse
import random
import cv2

# Command Line Arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the input image")
ap.add_argument("-m", "--method", type=str, default="fast", choices=["fast", "quality"], help="selective search method")
args = vars(ap.parse_args())

# Loading the image and in-builty cv2 selective search implementation
original_image = cv2.imread(args["image"])
selective_search = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
selective_search.setBaseImage(original_image)

if args["method"] == "fast":
    selective_search.switchToSelectiveSearchFast()
else:
    selective_search.switchToSelectiveSearchQuality()

# Running the selective search on the image
rects = selective_search.process()
print("Total Region Proposals: {}".format(len(rects)))

# Visualizing the output
for i in range(0,len(rects), 100):
    output = original_image.copy()
    for (x,y,w,h) in rects[i:i+100]:
        color = [random.randint(0,255) for j in range(0,3)]
        cv2.rectangle(output, (x,y), (x+w,x+h), color, 3)

    cv2.imshow("Output_Window", output)
    cv2.imwrite("./Output_Images/Output.jpg", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

