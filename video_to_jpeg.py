import argparse

import cv2

def extractImages(pathIn, pathOut):
    count = 1067
    vidcap = cv2.VideoCapture(pathIn)

    success = True

    while success:
        success,image = vidcap.read()
        print ('Read a new frame: ', success)
        print ('count: ', count)

        if success == True:
            cv2.imwrite( pathOut + "frame%d.jpg" % count, image)     # save frame as JPEG file
        count = count + 1

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--pathIn", help="path to video")
    parser.add_argument("--pathOut", help="path to images")
    args = parser.parse_args()
    
    if args.pathIn == None and args.pathOut == None:
        parser.print_help()

    extractImages(args.pathIn, args.pathOut)