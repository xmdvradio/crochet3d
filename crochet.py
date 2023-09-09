import numpy as np
import cv2 as cv
import csv

def generatecloud(colours, n, r): #n = number of colours in the pallet, r = radius of the blocking

    def generatenames(pallet):
        
        #gets the colour names and their respective xyv codes
        with open("colours.csv", "r") as f:
            r = csv.reader(f)
            names = []
            colours = []
            for colour in r:
                names.append(colour[1])
                colours.append(colour[3:])

        colours = np.array([colours], dtype="uint8")
        h, s, v = cv.cvtColor(colours, cv.COLOR_BGR2HSV)[0].T

        ang = (h*2*np.pi)/180
        x = (np.cos(ang)*(s/2))+127.5 #find the x
        y = (np.sin(ang)*(s/2))+127.5 #find the y
        colours = np.column_stack([x, y, v])

        namepallet = []
        for colour in pallet:
            deltas = colours - colour
            distances = np.linalg.norm(deltas, ord=2, axis=1.)
            i = np.argmin(distances)
            namepallet.append(names[i]) 

        return namepallet

    f = cv.cvtColor(colours, cv.COLOR_RGB2HSV)[0] #converts to hsv colours
    #if colours are weird check that its rotating the histogram correctly
    f = np.rot90(f, -1) #make what 3 big long lists -> hue, sat, val

    h, s, v = f #split into variables
    ang = (h*2*np.pi)/180
    x = (np.cos(ang)*(s/2))+127.5 #find the x
    y = (np.sin(ang)*(s/2))+127.5 #find the y
    f = x, y, v #reassign to f
    f = np.rot90(np.array(f, dtype=int), 1) #replace with a column stack i think

    colours, counts = np.unique(f, return_counts=True, axis=0) #return all individual colours and the amount of times they change

    pallet = np.array([[0,0,0] for _ in range(n)], dtype=int)
    cnum = 0

    while cnum < n: #while theres not enough colours

        i = np.argmax(counts) #get the index of the most common colour
        c = colours[i]
        pallet[cnum] = c #put the colour in the pallet at cnum
        counts[i] = 0 #zeros the count preventing reselection unless we run out of colours

        if cnum > 0:
            distances = np.sort(np.linalg.norm(pallet - c, ord=2, axis=1.)) #calculate the distances in accending order
            dis = distances[np.where(distances > 0.)[0][0]] #calculate the lowest distance that isnt zero
            if dis >= r: cnum += 1  #work out if its too close
        else: cnum += 1 #make the c num of the pallet the current colour

    names = generatenames(pallet)
    npallet = XYV2RGB(pallet)
    image = cv.cvtColor(np.array([npallet]), cv.COLOR_RGB2BGR)
    image = cv.resize(image, (720,720), interpolation=cv.INTER_NEAREST)
    cv.imwrite("pallet.jpg", image)

    cloud = []
    for color in f:
        distances = np.linalg.norm(pallet - color, ord=2, axis=1.)
        c = np.argmin(distances) #gets the index of the closest pallet colour

        cloud.append(npallet[c])

    cloud = np.array(cloud)

    return cloud/255, npallet/255, names

def XYV2RGB(a): #takes an array of XYV colours and outputs a BGR array

    a = np.array(a, dtype=int)

    #dont totally trust this bit -> got the hue by trial and error becasue following the docs didnt work

    xy = a[:,:2] #get a list of the x, y coords (dodgey as fuck)
    a = np.rot90(a, -1) #rotate the rest to make it a list of x, y, v
    x, y, v = a

    h = (np.arctan2(y-128, x-128)+np.pi)
    h = ((h + np.pi) % (2 * np.pi))*(90/np.pi)
    s = np.linalg.norm(xy-128, axis=1)*2 #100% a dimension error for this one

    h[h>180] = 180
    s[s>255] = 255
    v[v>255] = 255

    a = np.column_stack([h, s[::-1], v])[::-1]

    a = np.array([a], dtype="uint8")

    a = cv.cvtColor(a, cv.COLOR_HSV2RGB)[0]

    return a