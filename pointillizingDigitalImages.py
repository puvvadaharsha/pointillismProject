"""
Harsha Puvvada
CSC 481 Final Project
Converts digital image to Pointillistic Image
Functions are on top and main method is at the bottom of script
Dependencies: vectorField.py
"""

import math,random,cv2, scipy, bisect, progressbar, colorsys
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from vectorField import vectorField


#functions used/main script at bottom
def limitSize(img, ratio):
    '''Downsamples image based on given ratio'''
    if ratio == 1:
        return img
    else:
        height, width, depth = img.shape
        shape = (int(width * ratio), int(height * ratio))
        return cv2.resize(img, shape, interpolation=cv2.INTER_AREA)

def paletteGenerator(img, colors):
    '''Finds the dominant colors using Kmeans Clustering given image and number of colors'''
    #convert from bgr to rgb
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #reshape to list of pixels
    img = img.reshape((img.shape[0] * img.shape[1], 3))
    kmeans = KMeans(n_clusters = colors)
    out = kmeans.fit(img)
    return out.cluster_centers_

def showPalette(palette, paletteSize, name):
    '''Displays the cluster centers as colors and saves it into file'''
    cols = paletteSize
    rows = int(math.ceil(len(palette) / cols))
    res = np.zeros((rows * 80, cols * 80, 3), dtype=np.uint8)
    for y in range(rows):
        for x in range(cols):
            if y * cols + x < len(palette):
                color = [int(c) for c in palette  [y * cols + x]]
                cv2.rectangle(res, (x * 80, y * 80), (x * 80 + 80, y * 80 + 80), color, -1)
    fig = plt.figure()
    plt.axis("off")
    plt.imshow(res)
    
    if name == 'boosted':
        plt.title('Boosted color palette')
        fig.savefig('./output/boostedPaletteColors.jpg')
    else:
        plt.title('Original color palette')
        fig.savefig('./output/originalPaletteColors.jpg')
    return res

def boostColors(palette):
    '''boosts colors in palette'''
    boostedPalette = []
    for item in palette:
        b = item[0]; g = item[1]; r = item[2]
        h,s,v = colorsys.rgb_to_hsv(r,g,b)
        #increase hue, saturation and brightness
#        h = h * 1.2
        s = s * 1.2
        v = v * 1.4
        newr,newg,newb = colorsys.hsv_to_rgb(h,s,v)
        boostedPalette.append([newr,newg,newb])
    return np.asarray(boostedPalette, dtype=np.float64)


def randomized_grid(h, w, scale):
    '''creates a randomized grid on which to paint. This is the canvas'''
    assert (scale > 0)
    r = scale//2
    grid = []
    for i in range(0, h, scale):
        for j in range(0, w, scale):
            y = random.randint(-r, r) + i
            x = random.randint(-r, r) + j

            grid.append((y % h, x % w))

    random.shuffle(grid)
    return grid

def compute_color_probabilities(pixels, palette, k=9):
    '''compute the probabilities of the colors in grid'''
    distances = scipy.spatial.distance.cdist(pixels, palette)
    maxima = np.amax(distances, axis=1)

    distances = maxima[:, None] - distances
    summ = np.sum(distances, 1)
    distances /= summ[:, None]

    distances = np.exp(k*len(palette)*distances)
    summ = np.sum(distances, 1)
    distances /= summ[:, None]

    return np.cumsum(distances, axis=1, dtype=np.float32)

def color_select(probabilities, palette):
    '''picks the nearest color that is most similar to original image'''
    r = random.uniform(0, 1)
    i = bisect.bisect_left(probabilities, r)
    return palette[i] if i < len(palette) else palette[-1]

######################  Main Method ########################################

###read in image
img = cv2.imread("./images/image3.jpg")

#### Step 1: Downsample Image
#downsample by 0.5 of image to speed up Kmeans
ratio = 0.2
shrunkImg = limitSize(img, ratio)
#Save Shrunk Image
cv2.imwrite('./output/shrunkImage.jpg', shrunkImg)


### Step 2: Compute color palette using Kmeans
paletteSize = 20
palette = paletteGenerator(shrunkImg, paletteSize)
originalPaletteColors = showPalette(palette, paletteSize, 'original')


### Step 3: Boost color saturation by converting from rgb to hsv and then increasing saturation
boosted = boostColors(palette)

### Step 4: Get vector field of image using Scharr Image derivative
#convert image to grayscale to compute image gradient 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gradient = vectorField.from_gradient(gray)


### Step 5: Smooth vector field using Gaussian Blur
radius = 0; #0 is automatic
gradient.smooth(radius)


### Step 6: Paint image
# cartoonized version is used as a base. Made using medianBlur
res = cv2.medianBlur(img, 11)
cv2.imwrite('./output/baseImage.jpg', res)


# define a randomized grid of locations for the brush strokes
grid = randomized_grid(img.shape[0], img.shape[1], scale=3)
batch_size = 10000

bar = progressbar.ProgressBar()
for h in bar(range(0, len(grid), batch_size)):
    # get the pixel colors at each point of the grid
    pixels = np.array([img[x[0], x[1]] for x in grid[h:min(h + batch_size, len(grid))]])
    # precompute the probabilities for each color in the palette
    # lower values of k means more randomnes
    color_probabilities = compute_color_probabilities(pixels, boosted, k=9)    
    #define stroke scale:
    stroke_scale = int(math.ceil(max(img.shape) / 900))
    
    for i, (y, x) in enumerate(grid[h:min(h + batch_size, len(grid))]):
        color = color_select(color_probabilities[i], boosted)
        angle = math.degrees(gradient.direction(y, x)) + 90
        length = int(round(stroke_scale + stroke_scale * math.sqrt(gradient.magnitude(y, x))))
        # draw ellipse stroke
        #cv2.ellipse(res, (x, y), (length, stroke_scale), angle, 0, 360, color, -1, cv2.LINE_AA)
        
        # draw circle stroke
        cv2.circle(res, (x,y), stroke_scale, color, thickness=-1, lineType= 8, shift=0)
### Step 7: save Final Image
cv2.imwrite('./output/pointillizedImage.jpg', res)



