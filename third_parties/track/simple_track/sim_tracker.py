# Based on Simple Object Tracking with OpenCV by pyimagesearch 
# https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/

import numpy as np
from scipy.stats import variation
from collections import OrderedDict
        

class ObjectTracker():
    # Initializes tracker
    def __init__(self, area_thresh = 0.15, window_size=20, tolerance=15):
        self.ptr = 0
        self.tolerance = tolerance
        self.centroids = OrderedDict()
        self.areas = OrderedDict()
        self.changed = OrderedDict()
        self.disappeared = OrderedDict()
        self.thresh = area_thresh
        self.window = window_size
        self.bboxes = OrderedDict()

    # Registers an object in the tracking list
    def register(self, centroid, area, bbox=None):
        self.centroids[self.ptr] = centroid
        self.areas[self.ptr] = area
        self.disappeared[self.ptr] = 0
        self.changed[self.ptr] = False
        self.bboxes[self.ptr] = bbox
        self.ptr += 1

    # Removes an object from the tracking list
    def remove(self, id): 
        del self.centroids[id]
        del self.areas[id]
        del self.disappeared[id]
        del self.changed[id]
        if id in self.bboxes:
            del self.bboxes[id]

    # Use the bounding box coordinates to derive the centroid
    def compute_centroids(self, coord_objs):
        centroids = np.zeros((len(coord_objs), 2), dtype='int')
        for i in range(len(coord_objs)):
            c_x = int((coord_objs[i][0] + coord_objs[i][2]) / 2.0)
            c_y = int((coord_objs[i][1] + coord_objs[i][3]) / 2.0)
            centroids[i] = (c_x, c_y)
        return centroids

    # Use the bounding box coordinates to derive the bounding boxes area
    def compute_areas(self, coord_objs):
        areas = np.zeros(len(coord_objs), dtype = float)
        for i in range(len(coord_objs)):
            areas[i] = (coord_objs[i][2] - coord_objs[i][0]) * (coord_objs[i][3] - coord_objs[i][1])
        return areas
    

    def tracking(self, coord_objs, max_euclidean_distance_to_bind=40):
        if len(coord_objs) == 0:
            for id in list(self.disappeared.keys()):
                self.disappeared[id] += 1
                if self.disappeared[id] > self.tolerance:
                    self.remove(id)

            # return self.centroids, self.areas, self.bboxes
            return {}, {}, {}
                    
        centroids = self.compute_centroids(coord_objs)
        areas = self.compute_areas(coord_objs)

        # There are no objects to track yet
        if len(self.centroids) == 0:
            # Registering objects found in the current frame
            for c in range(len(centroids)):
                self.register(centroids[c], areas[c], coord_objs[c])

        # Objects are already being tracked
        else:
            # List of object ids and corresponding centroids and areas
            object_ids = list(self.centroids.keys())
            object_centroids = list(self.centroids.values())

            # Compute the distance between each pair of object centroids and new centroids
            # Columns of matrix D: distance from the first element of "centroids" to all others of "object_centroids" ... 
            # D = dist.cdist(np.array(object_centroids), centroids, metric = 'euclidean') #from scipy.spatial import distance as dist
            D = np.array([[np.linalg.norm(i-j) for j in centroids] for i in np.array(object_centroids)])

            # Smallest value in each row
            # Shortest distance between each element of "object_centroids" and all elements of "centroids"
            row_min = D.min(axis=1)

            # Sort the row indexes based on their minimum values
            rows = row_min.argsort()
            checker = np.sort(row_min)
            rows = rows[checker < max_euclidean_distance_to_bind]

            # Smallest value in each column
            # Shortest distance between each element of "centroids" and all elements of "object_centroids"
            col_min = D.argmin(axis=1)
            # Sorting using thre previously computed row index list
            cols = col_min[rows]

            # Set of column and row indexes already examined
            usedCols = set()
            usedRows = set()

            # Loop over the combination of the index tuples
            for (row, col) in zip(rows, cols):
                # If the value has already been examined, just ignore it
                if col in usedCols or row in usedRows:
                    continue

                # Otherwise, grab the object ID for the current row
                objectID = object_ids[row]
                # Update centroid
                self.centroids[objectID] = centroids[col]
                # Update area
                self.areas[objectID] = areas[col]
                # Update bbox
                self.bboxes[objectID] = coord_objs[col]
                # Reset the disappeared counter
                self.disappeared[objectID] = 0

                # Adds examined column and row
                usedCols.add(col)
                usedRows.add(row)

            # Computes columns and rows that have not been examined
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)

            # If the number of object centroids is equal or greater than the number of current centroids
            if D.shape[0] >= D.shape[1]:
                # Check if some of these objects have potentially disappeared
                for row in unusedRows:
                    objectID = object_ids[row]
                    self.disappeared[objectID] += 1

                    # If the number of consecutive frames without the object has been extrapolated, delete it
                    if self.disappeared[objectID] > self.tolerance:
                        self.remove(objectID)
            else:
                # If the number of current centroids is greater than the number of object centroids
                # Register each new object centroid as a trackable object
                for col in unusedCols:
                    self.register(centroids[col], areas[col], coord_objs[col])

        # Return the set of trackable objects
        return self.centroids, self.areas, self.bboxes