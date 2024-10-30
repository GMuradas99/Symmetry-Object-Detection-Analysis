import cv2
import numpy as np
import matplotlib.pyplot as plt

from os.path import join, exists
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

### HELPER FUNCTIONS

def getImgPath(imgName: str):
    """Returns the correct path for the image
    """
    if exists(join('images/images_art',imgName)):
        return join('images/images_art',imgName)
    elif exists(join('images/images_nat', imgName)):
        return join('images/images_nat',imgName)
    else:
        return "error"   

def imRead(path: str):
    """Returns numpy array of the image on the path
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if len(img.shape) != 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

# Remove negative coordinates from list
def removeNegativeCoordinates(points, height, width):
    result = []
    for point in points:
        x = point[0]
        y = point[1]
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x > width:
            x = width
        if y > height:
            y = height
        result.append((x,y))
    return result

### DRAWERS ###

def drawAxis(mask, row, thickness = 3, color = 1):
    """Draws symmetry axis of desired thickness and color on the mask
    """
    pts = getBBPoints(row)

    if 'axis' in row:
        orientation = row['axis']
    else:
        orientation = "vertical_axis"
    
    # Drawing symmetry axis
    if orientation == "vertical_axis":
        startAxis = ((pts[0][0] + pts[1][0])/2  , (pts[0][1] + pts[1][1])/2)
        endAxis = ((pts[2][0] + pts[3][0])/2  , (pts[2][1] + pts[3][1])/2) 
    elif orientation == "horizontal_axis":
        startAxis = ((pts[0][0] + pts[3][0])/2  , (pts[0][1] + pts[3][1])/2)
        endAxis = ((pts[1][0] + pts[2][0])/2  , (pts[1][1] + pts[2][1])/2)    

    cv2.line(mask, (int(startAxis[0]),int(startAxis[1])), (int(endAxis[0]),int(endAxis[1])), color, thickness)

    return mask

def drawRectangle(img: np.ndarray, points: list, thickness = 1):
    """Draws rectangle on the image
    """
    cv2.line(img, (int(points[0][0]),int(points[0][1])), (int(points[1][0]),int(points[1][1])), [0,255,0], thickness)
    cv2.line(img, (int(points[1][0]),int(points[1][1])), (int(points[2][0]),int(points[2][1])), [0,255,0], thickness)
    cv2.line(img, (int(points[2][0]),int(points[2][1])), (int(points[3][0]),int(points[3][1])), [0,255,0], thickness)
    cv2.line(img, (int(points[3][0]),int(points[3][1])), (int(points[0][0]),int(points[0][1])), [0,255,0], thickness)

def drawBB(img: np.ndarray, row, thicknessAxis = 1, thicknessRectangle = 1, colorAxis = [255,0,0]):
    """Draws the bounding box of the row on an img
    """
    # Getting bounding box points
    pts = getBBPoints(row)

    # Drawing symmetry axis
    if row['axis'] == "vertical_axis":
        startAxis = ((pts[0][0] + pts[1][0])/2  , (pts[0][1] + pts[1][1])/2)
        endAxis = ((pts[2][0] + pts[3][0])/2  , (pts[2][1] + pts[3][1])/2) 
    elif row['axis'] == "horizontal_axis":
        startAxis = ((pts[0][0] + pts[3][0])/2  , (pts[0][1] + pts[3][1])/2)
        endAxis = ((pts[1][0] + pts[2][0])/2  , (pts[1][1] + pts[2][1])/2)        
    else:
        print(row['axis'])
    cv2.line(img, (int(startAxis[0]),int(startAxis[1])), (int(endAxis[0]),int(endAxis[1])), colorAxis, thicknessAxis)

    # Drawing rencangle of bounding box
    drawRectangle(img, pts, thicknessRectangle)

    return img

### OPERATIONS ###

def crop_image(img: np.ndarray, row) -> np.ndarray:
    """
    Crops and rotates a given image based on the bounding box parameters provided in the row.
    Args:
        img (numpy.ndarray): The input image to be cropped.
        row (dict): A dictionary containing the bounding box parameters:
            - 'width_box' (int): The width of the bounding box.
            - 'height_box' (int): The height of the bounding box.
            - 'centerX' (int): The x-coordinate of the center of the bounding box.
            - 'centerY' (int): The y-coordinate of the center of the bounding box.
            - 'rotation' (float): The rotation angle of the bounding box in degrees.
    Returns:
        numpy.ndarray: The cropped and rotated image.
    """

    width = row['width_box']
    height = row['height_box']
    centerX = row['centerX']
    centerY = row['centerY']
    rotation = row['rotation']

    # Calculating the bounding box
    pts = [(centerX-width/2 , centerY-height/2), (centerX+width/2 , centerY-height/2), 
           (centerX+width/2 , centerY+height/2), (centerX-width/2 , centerY+height/2)]
    
    # Removing negative coordinates
    pts = removeNegativeCoordinates(pts, img.shape[0], img.shape[1])

    # Rotating and cropping image
    rotationMatrix = cv2.getRotationMatrix2D((centerX,centerY),rotation,1)
    img = cv2.warpAffine(img,rotationMatrix,(img.shape[1], img.shape[0]))
    img = img[int(pts[0][1]):int(pts[0][1])+int(height), int(pts[0][0]):int(pts[0][0]+width)]

    return img

def transformKeypoints(keypoints: list, rotationMatrix: np.ndarray):
    """Returns a list with all the transformmed keypoints
    """
    result = []
    for keypoint in keypoints:
        rotatedPoint = rotationMatrix.dot(np.array(keypoint + (1,)))
        result.append((rotatedPoint[0],rotatedPoint[1]))

    return result

def getBBPoints(row, integers: bool = False):
    """Returns the rotated points for the bounding box of the indicated row
    """
    # Points for bounding box
    pts = [(row['centerX']-row['width_box']/2 , row['centerY']-row['height_box']/2), (row['centerX']+row['width_box']/2 , row['centerY']-row['height_box']/2), 
           (row['centerX']+row['width_box']/2 , row['centerY']+row['height_box']/2), (row['centerX']-row['width_box']/2 , row['centerY']+row['height_box']/2)]
    
    # Rotating box
    center = (row['centerX'], row['centerY'])
    rotation_matrix = cv2.getRotationMatrix2D(center, -row['rotation'], 1)
    pts = transformKeypoints(pts, rotation_matrix)

    # Return integers if needed
    if integers:
        intPoints = []
        for pt in pts: 
            intPoints.append([round(pt[0]), round(pt[1])])
        return intPoints

    return pts

def addBBtoMask(mask: np.ndarray, row):
    """Adds 1 to all the pixels inside the bounding box (VERY SLOW!!!)
    """
    pts = getBBPoints(row)
    polygon = Polygon(pts)
    for i in range(len(mask[0])):
        for j in range(len(mask)):
            point = Point(i,j)
            if (polygon.contains(point)):
                mask[j,i] += 1

def getRowMask(shape: tuple, row, num: int=1):
    """Returns a mask of the selected shape with the bounding box indicated by the bounding box
    """
    # Extracting mask of 1s
    points = getBBPoints(row)
    p = np.array(points)
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, np.int32([p]), color=1)
    # Transforming mask into the desired number
    mask = mask.astype(np.float32)
    mask = mask * num

    return mask

def getBBMask(imgName: str, df):
    """Draws all the bounding boxes and axis from a single image
    """
    matches = df.loc[df['file_name'] == imgName]
    img = cv2.imread(getImgPath(imgName))
    mask = np.zeros((img.shape[:2]), np.uint8)
    for _,row in matches.iterrows():
        nextBB = getRowMask(img.shape[:2], row)
        mask = np.add(mask, nextBB)

    return mask

def getBoundingBoxHeatmap(imgName: str, df):
    """Returns image superposed with heatmap of labeled symmetry
    """
    img = cv2.imread(getImgPath(imgName))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mask = getBBMask(imgName,df)

    # Normalizing and adding thmask
    normalized = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    blur = cv2.GaussianBlur(normalized,(15,15), 11)
    heatmap_img = cv2.applyColorMap(blur, cv2.COLORMAP_RAINBOW)

    super_imposed_img = cv2.addWeighted(heatmap_img, 0.5, img, 0.5, 0)

    return super_imposed_img

# ADD DESCRIPTOR
def getBBMaskScore(imgName: str, column, df):
    matches = df.loc[df['file_name'] == imgName]
    img = cv2.imread(getImgPath(imgName))
    score = np.zeros((img.shape[:2]), np.float32)
    potentialScore = np.zeros((img.shape[:2]), np.float32)
    maximum = round(df[column].max(),0)
    for _,row in matches.iterrows():
        percieved = getRowMask(img.shape[:2], row, num = row[column])
        potential = getRowMask(img.shape[:2], row, num = maximum)
        score = np.add(score, percieved)
        potentialScore = np.add(potentialScore, potential)

    return score, potentialScore

def displayHeatmaps(heatmaps: list, titles: list, saveFigPath: str = None):
    """Displays all given heat maps
    """
    if len(heatmaps) == 1:
        plt.imshow(heatmaps[0], cmap='rainbow')
        plt.title(titles[0])
        plt.colorbar()
        if saveFigPath is not None:
            plt.savefig(saveFigPath)
        plt.show()
    else:
        fig, ax = plt.subplots(1,len(heatmaps),figsize=(15, 5))
        for idx, heatmap in enumerate(heatmaps):
            im2,_ = ax[idx].imshow(heatmap, cmap='rainbow'), ax[idx].set_title(titles[idx])
        fig.colorbar(im2, ax=ax, orientation='vertical')
        if saveFigPath is not None:
            plt.savefig(saveFigPath)
        plt.show()

def getHeatmap(imageName: str, column, df):
    """Returns the desired heatmap
    """
    # Calculating
    score, potentialScore = getBBMaskScore(imageName, column, df)
    divisiblePotentialScore = potentialScore.copy()
    divisiblePotentialScore[divisiblePotentialScore == 0] = 1
    result = score / divisiblePotentialScore

    # Display
    img = cv2.imread(getImgPath(imageName))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    super_imposed_img = applyHeatmap(img, result)

    return super_imposed_img, result

def applyHeatmap(img: np.ndarray, mask: np.ndarray):
    """Applies mask to img as a heatmap and returns the visualization of the heatmap
    """
    normalized = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    blur = cv2.GaussianBlur(normalized,(15,15), 11)
    heatmap_img = cv2.applyColorMap(blur, cv2.COLORMAP_RAINBOW)
    super_imposed_img = cv2.addWeighted(heatmap_img, 0.5, img, 0.5, 0)

    return super_imposed_img

### DISPLAYING ###

def displayBB(row):
    """Displays the bounding box of the row
    """
    # Opening image
    img = cv2.imread(getImgPath(row['file_name']))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Draw bounding box and symmetry axis
    drawBB(img, row)    

    # Display
    plt.imshow(img)
    plt.show()

def displayAllBB(imgName: str, df):
    """ Draws all the bounding boxes and axis from a single image
    """
    matches = df.loc[df['file_name'] == imgName]
    img = cv2.imread(getImgPath(imgName))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for _,row in matches.iterrows():
        drawBB(img,row)
    
    return img

def removeNegativeCoordinates(points: list, height: int, width: int):
    """Remove negative coordinates from list
    """
    result = []
    for point in points:
        x = point[0]
        y = point[1]
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x > width:
            x = width
        if y > height:
            y = height
        result.append((x,y))
    return result

def cropBoundingBox(image: np.ndarray, row):
    """Crops image according to bounding box
    """
    # Calculating the bounding box
    pts = [(row['centerX']-row['width_box']/2 , row['centerY']-row['height_box']/2), (row['centerX']+row['width_box']/2 , row['centerY']-row['height_box']/2), 
            (row['centerX']+row['width_box']/2 , row['centerY']+row['height_box']/2), (row['centerX']-row['width_box']/2 , row['centerY']+row['height_box']/2)]
    
    pts = removeNegativeCoordinates(pts, image.shape[0], image.shape[1])

    rotationMatrix = cv2.getRotationMatrix2D((int(row['centerX']),int(row['centerY'])),row['rotation'],1)
    rotate = cv2.warpAffine(image,rotationMatrix,(image.shape[1], image.shape[0])) 
    cropped = rotate[int(pts[0][1]):int(pts[0][1])+int(row['height_box']), int(pts[0][0]):int(pts[0][0]+row['width_box'])]

    return cropped