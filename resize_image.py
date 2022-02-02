import cv2
from os import listdir
from os.path import isfile, join
import argparse
from pathlib import Path
import time
import multiprocessing
import numpy as np

def convert_image(list_of_files, in_folder, out_folder, width, height, starting_value):

    count = 0

    for file in list_of_files:
        if ".jpg" in file or ".jpeg" in file:

            abs_filepath = str(Path(in_folder+file).resolve())
            print(abs_filepath)
            img = cv2.imread(abs_filepath, cv2.IMREAD_UNCHANGED)
    
            if img is not None:
                print("Original Dimensions : {}".format(img.shape))

                dim = (width, height)
                resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

                im_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

                cv2.imwrite( out_folder + "frame_resized_{}.jpg".format(starting_value), resized)     # save frame as JPEG file
                starting_value = starting_value + 1
                count = count + 1
                print("Resized Dimensions : {}".format(resized.shape))
                time.sleep(1)
                print("Remaining {} images".format(len(list_of_files) - count))
            else:
                print("Can't open image {}".format(abs_filepath))
                break
    

def convert_images(in_folder, out_folder, width, height):

    list_of_files = [f for f in listdir(in_folder) if isfile(join(in_folder, f))]

    threads = 32
    splitted = np.array_split(list_of_files, threads)

    jobs = []
    for i in range(0, threads):  
        print("Adding thread {}".format(i))
        starting_value = i*len(splitted[i])
        print("Starting idx {}".format(starting_value))
        process  = multiprocessing.Process(target=convert_image, args=(splitted[i], in_folder, out_folder, width, height,starting_value))
        jobs.append(process)


    for j in jobs:
        j.start()

    # Ensure all of the threads have finished
    for j in jobs:
        j.join()

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--in_folder", help="path to the folder that contains the original images")
    parser.add_argument("--out_folder", help="path to the folder in which the resized images will be saved")
    parser.add_argument("--width", help="output width",default=300)
    parser.add_argument("--height", help="output height", default=300)
    args = parser.parse_args()
    
    if args.in_folder[-1] != "/":
        args.in_folder = args.in_folder + "/"

    if args.out_folder[-1] != "/":
        args.out_folder = args.out_folder + "/"

    if args.in_folder == None and args.out_folder == None and args.width and args.height:
        parser.print_help()

    convert_images(args.in_folder, args.out_folder, args.width, args.height)