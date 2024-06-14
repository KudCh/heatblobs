#!/usr/bin/env python3
# coding: utf-8
#
# Original implementation by Gregory SOETENS (BINFO UniLu)
# using https://github.com/jankovicsandras/autoimagemorph as a starting point
# and also previous code by Kristina KUDRYAVTSEVA (BiCS UniLu).
# Code refactored by Luis LEIVA (UniLu) while trying to decipher what the students did.

import sys
import argparse
import csv
import io
import cv2
import time
from datetime import datetime
import numpy as np
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from matplotlib.path import Path
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import median_filter
from scipy.spatial import Delaunay
from scipy.stats import gaussian_kde
import scipy
import copy
import os
from scipy.spatial import delaunay_plot_2d

# EDIT:
import glob

class Triangle:
    def __init__(self, vertices):
        if isinstance(vertices, np.ndarray) == 0:
            raise ValueError("Input argument is not of type np.array.")
        if vertices.shape != (3, 2):
            raise ValueError("Input argument does not have the expected dimensions.")
        if vertices.dtype != np.float64:
            raise ValueError("Input argument is not of type float64.")
        self.vertices = vertices

    # Credit to https://github.com/zhifeichen097/Image-Morphing for the following approach (which is a bit more efficient than my own)!
    def getPoints(self):
        width = round(max(self.vertices[:, 0]) + 2)
        height = round(max(self.vertices[:, 1]) + 2)
        mask = Image.new('P', (width, height), 0)
        ImageDraw.Draw(mask).polygon(tuple(map(tuple, self.vertices)), outline=255, fill=255)
        coordArray = np.transpose(np.nonzero(mask))

        return coordArray

class Morpher:
    def __init__(self, leftImage, leftTriangles, rightImage, rightTriangles):
        if type(leftImage) != np.ndarray:
            raise TypeError('Input leftImage is not an np.ndarray')
        if leftImage.dtype != np.uint8:
            raise TypeError('Input leftImage is not of type np.uint8')
        if type(rightImage) != np.ndarray:
            raise TypeError('Input rightImage is not an np.ndarray')
        if rightImage.dtype != np.uint8:
            raise TypeError('Input rightImage is not of type np.uint8')
        if type(leftTriangles) != list:
            raise TypeError('Input leftTriangles is not of type List')
        for j in leftTriangles:
            if isinstance(j, Triangle) == 0:
                raise TypeError('Element of input leftTriangles is not of Class Triangle')
        if type(rightTriangles) != list:
            raise TypeError('Input leftTriangles is not of type List')
        for k in rightTriangles:
            if isinstance(k, Triangle) == 0:
                raise TypeError('Element of input rightTriangles is not of Class Triangle')
        self.leftImage = copy.deepcopy(leftImage)
        self.newLeftImage = copy.deepcopy(leftImage)
        self.leftTriangles = leftTriangles  # Not of type np.uint8
        self.rightImage = copy.deepcopy(rightImage)
        self.newRightImage = copy.deepcopy(rightImage)
        self.rightTriangles = rightTriangles  # Not of type np.uint8

    def get_image_at_alpha(self, alpha):
        for leftTriangle, rightTriangle in zip(self.leftTriangles, self.rightTriangles):
            self.interpolatePoints(leftTriangle, rightTriangle, alpha)
        return ((1 - alpha) * self.newLeftImage + alpha * self.newRightImage).astype(np.uint8)

    def interpolatePoints(self, leftTriangle, rightTriangle, alpha):
        targetTriangle = Triangle(leftTriangle.vertices + (rightTriangle.vertices - leftTriangle.vertices) * alpha)
        targetVertices = targetTriangle.vertices.reshape(6, 1)
        tempLeftMatrix = np.array([[leftTriangle.vertices[0][0], leftTriangle.vertices[0][1], 1, 0, 0, 0],
                                   [0, 0, 0, leftTriangle.vertices[0][0], leftTriangle.vertices[0][1], 1],
                                   [leftTriangle.vertices[1][0], leftTriangle.vertices[1][1], 1, 0, 0, 0],
                                   [0, 0, 0, leftTriangle.vertices[1][0], leftTriangle.vertices[1][1], 1],
                                   [leftTriangle.vertices[2][0], leftTriangle.vertices[2][1], 1, 0, 0, 0],
                                   [0, 0, 0, leftTriangle.vertices[2][0], leftTriangle.vertices[2][1], 1]])
        tempRightMatrix = np.array([[rightTriangle.vertices[0][0], rightTriangle.vertices[0][1], 1, 0, 0, 0],
                                    [0, 0, 0, rightTriangle.vertices[0][0], rightTriangle.vertices[0][1], 1],
                                    [rightTriangle.vertices[1][0], rightTriangle.vertices[1][1], 1, 0, 0, 0],
                                    [0, 0, 0, rightTriangle.vertices[1][0], rightTriangle.vertices[1][1], 1],
                                    [rightTriangle.vertices[2][0], rightTriangle.vertices[2][1], 1, 0, 0, 0],
                                    [0, 0, 0, rightTriangle.vertices[2][0], rightTriangle.vertices[2][1], 1]])
        try:
            lefth = np.linalg.solve(tempLeftMatrix, targetVertices)
            righth = np.linalg.solve(tempRightMatrix, targetVertices)
            leftH = np.array([[lefth[0][0], lefth[1][0], lefth[2][0]], [lefth[3][0], lefth[4][0], lefth[5][0]], [0, 0, 1]])
            rightH = np.array([[righth[0][0], righth[1][0], righth[2][0]], [righth[3][0], righth[4][0], righth[5][0]], [0, 0, 1]])
            leftinvH = np.linalg.inv(leftH)
            rightinvH = np.linalg.inv(rightH)
            targetPoints = targetTriangle.getPoints()

            # Credit to https://github.com/zhifeichen097/Image-Morphing for the following code block that I've adapted. Exceptional work on discovering
            # RectBivariateSpline's .ev() method! I noticed the method but didn't think much of it at the time due to the website's poor documentation..
            xp, yp = np.transpose(targetPoints)
            leftXValues = leftinvH[1, 1] * xp + leftinvH[1, 0] * yp + leftinvH[1, 2]
            leftYValues = leftinvH[0, 1] * xp + leftinvH[0, 0] * yp + leftinvH[0, 2]
            leftXParam = np.arange(np.amin(leftTriangle.vertices[:, 1]), np.amax(leftTriangle.vertices[:, 1]), 1)
            leftYParam = np.arange(np.amin(leftTriangle.vertices[:, 0]), np.amax(leftTriangle.vertices[:, 0]), 1)
            leftImageValues = self.leftImage[int(leftXParam[0]):int(leftXParam[-1] + 1), int(leftYParam[0]):int(leftYParam[-1] + 1)]

            rightXValues = rightinvH[1, 1] * xp + rightinvH[1, 0] * yp + rightinvH[1, 2]
            rightYValues = rightinvH[0, 1] * xp + rightinvH[0, 0] * yp + rightinvH[0, 2]
            rightXParam = np.arange(np.amin(rightTriangle.vertices[:, 1]), np.amax(rightTriangle.vertices[:, 1]), 1)
            rightYParam = np.arange(np.amin(rightTriangle.vertices[:, 0]), np.amax(rightTriangle.vertices[:, 0]), 1)
            rightImageValues = self.rightImage[int(rightXParam[0]):int(rightXParam[-1] + 1), int(rightYParam[0]):int(rightYParam[-1] + 1)]

            self.newLeftImage[xp, yp] = RectBivariateSpline(leftXParam, leftYParam, leftImageValues, kx=1, ky=1).ev(leftXValues, leftYValues)
            self.newRightImage[xp, yp] = RectBivariateSpline(rightXParam, rightYParam, rightImageValues, kx=1, ky=1).ev(rightXValues, rightYValues)
        except:
            return

class AnimationCreator:
    def __init__(self, images, polygons, duration, fps, text='', smoothing=None):
        total_frames = fps*duration
        total_transition_frames = (total_frames-len(images))
        n_transitions = (len(images)-1)

        self.images = images
        self.polygons = polygons
        self.duration = duration
        self.fps = fps
        self.smoothing = smoothing
        self.num_morph_frames = total_transition_frames//n_transitions
        self.overlay_text = text

        print('Number of intermediate frames: ', self.num_morph_frames)
        print('Total frames: ', total_frames)

    def loadTriangles(self, limg, rimg, polygons_left, polygons_right):
        leftTriList = []
        rightTriList = []

        leftArray = polygons_left
        rightArray = polygons_right
        delaunayTri = Delaunay(leftArray)

        _ = delaunay_plot_2d(delaunayTri)
        plt.show()
        plt.savefig("delaunay.png")
        plt.close()
        
        leftNP = leftArray[delaunayTri.simplices]
        rightNP = rightArray[delaunayTri.simplices]
        
        for left, right in zip(leftNP, rightNP):
            leftTriList.append(Triangle(left))
            rightTriList.append(Triangle(right))
        
        return leftTriList, rightTriList

    def init_morph(self, start_image_path, end_image_path, polygons_left, polygons_right):
        left_image_arr = start_image_path

        right_image_raw = end_image_path
        right_image_raw = cv2.resize(right_image_raw, (left_image_arr.shape[1], left_image_arr.shape[0]), interpolation=cv2.INTER_CUBIC)
        right_image_arr = np.asarray(right_image_raw)

        triangle_left, triangle_right = self.loadTriangles(left_image_arr, right_image_arr, polygons_left, polygons_right)

        triangle_tuple = (triangle_left, triangle_right)

        # Morpher objects for color layers BGR
        morphers = [
            Morpher(left_image_arr[:, :, 0], triangle_tuple[0], right_image_arr[:, :, 0], triangle_tuple[1]),
            Morpher(left_image_arr[:, :, 1], triangle_tuple[0], right_image_arr[:, :, 1], triangle_tuple[1]),
            Morpher(left_image_arr[:, :, 2], triangle_tuple[0], right_image_arr[:, :, 2], triangle_tuple[1])
        ]

        return morphers


    def morph_process(self, morphers):
        for i in range(1, self.num_morph_frames):
            alpha = i / self.num_morph_frames

            # image calculation and smoothing BGR
            if self.smoothing > 0 :
                print('Smoothing present: ', self.smoothing)
                out_image = np.dstack([
                    np.array(median_filter(morphers[0].get_image_at_alpha(alpha),self.smoothing)),
                    np.array(median_filter(morphers[1].get_image_at_alpha(alpha),self.smoothing)),
                    np.array(median_filter(morphers[2].get_image_at_alpha(alpha),self.smoothing)),
                ])
            else :
                out_image = np.dstack([
                    np.array(morphers[0].get_image_at_alpha(alpha)),
                    np.array(morphers[1].get_image_at_alpha(alpha)),
                    np.array(morphers[2].get_image_at_alpha(alpha)),
                ])
            #cv2.imshow('frame {}'.format(i), out_image)
            #cv2.waitKey(0)
            self.video.write(cv2.putText(img=cv2.resize(out_image, (len(self.images[0][0]), len(self.images[0]))),
                                org=(int(0.3*len(self.images[0][0])), int(0.9*len(self.images[0]))), text=self.overlay_text,
                                color=(255, 255, 255), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1))


    def save(self, outfile):
        if not self.images:
            raise ValueError('No images could be processed. Exiting ...')

        print(f'Preparing to morph {len(self.images)} images ...', file=sys.stderr)

        #vid_size = (self.images[0].shape[:2]) # Doesn't work
        vid_size = (len(self.images[0][0]), len(self.images[0]))

        vid_writer = cv2.VideoWriter_fourcc(*'mp4v')
        self.video = cv2.VideoWriter(outfile, vid_writer, self.fps, vid_size)

        for idx in range(len(self.images) - 1):
            print(f'Morphing from image {idx+1} to image {idx+2} ...', file=sys.stderr)
            self.morph_process(
                self.init_morph(self.images[idx], self.images[idx + 1], self.polygons[idx], self.polygons[idx + 1])
            )

        self.video.release()


class HeatmapFrames:
    images, polygons = [], []

    def __init__(self, move_events, map_width=1280, map_height=768, dpi=100, kde_bins=200):
        self.move_events = move_events
        self.map_width = map_width
        self.map_height = map_height
        self.dpi = dpi
        self.kde_bins = kde_bins


    def slice_moves(self, first_ts, until_ts):
        moves = []
        for (x, y, t) in self.move_events:
            if t < first_ts:
                continue
            if t > until_ts:
                break
            moves.append([x,y,t])

        return moves


    def get_coordinates_kde(self, moves):
        X = np.array([p[0] for p in moves])
        Y = np.array([p[1] for p in moves])

        my_data = np.vstack([X, Y])
        #print('Bins: ', self.kde_bins)
        xi, yi = np.mgrid[0:self.map_width:self.kde_bins * 1j, 0:self.map_height:self.kde_bins * 1j]

        kernel = gaussian_kde(my_data)
        zi = kernel(np.vstack([xi.flatten(), yi.flatten()]))

        return [xi, yi, zi]


    def create_polygons(self, data, axes, x_scale=1, y_scale=1):
        plt.gca().set_axis_off()
        # FIXME: Why we can't set all margins to 0?
        # "ValueError: left cannot be >= right"
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)

        xi, yi, zi = data[0][0], data[0][1], data[0][2]
        cs = axes.contourf(xi, yi, zi.reshape(xi.shape), levels=4)

        paths_col = cs.collections
        
        # only consider the two innermost contours
        #collections = paths_col[len(paths_col) - 2:len(paths_col)]
        collections = paths_col
        polygons_all = []

        for collection in collections:
            paths = collection.get_paths()
            for path in paths:
                polygons = path.to_polygons()
                polygons_all.append(polygons)

        polygons = [x for y in polygons_all for x in y]

        # scale polygons, as an example
        for i in range(len(polygons)):
            polygon = polygons[i]
            xl = [coord[0]/x_scale for coord in polygon]
            yl = [coord[1]/y_scale for coord in polygon]
            polygons[i] = [[xl[i], yl[i]] for i in range(len(xl))]

        # pick the last contour
        # last item in the list is the first contour
        first_contour = polygons[-1]
        second_contour = polygons[-2] # why is the second contour the same as the first one?
        third_contour = polygons[-3]
        #fourth_contour = polygons[-4]
        #contour = first_contour + third_contour
        contour = third_contour 
        #for i in polygons[2:]:
        #    contour += i 
        # resample n points in the contour

        contour = polygons[-1]
        contour = scipy.signal.resample(contour, 50)

        return contour

    # Automatic feature points
    def autofeaturepoints(self, img, featuregridsize=7, showfeatures=True):
        result = []

        try:

            if (showfeatures) : print(img.shape)
            
            # add the 4 corners to result
            result += ([ [0, 0], [(img.shape[1]-1), 0], [0, (img.shape[0]-1)], [(img.shape[1]-1), (img.shape[0]-1)] ])

            h = int(img.shape[0] / featuregridsize)-1
            w = int(img.shape[1] / featuregridsize)-1
            
            for i in range(0,featuregridsize) :
                for j in range(0,featuregridsize) :
                    
                    # crop to a small part of the image and find 1 feature pont or middle point
                    crop_img = img[ (j*h):(j*h)+h, (i*w):(i*w)+w ]
                    gray = cv2.cvtColor(crop_img,cv2.COLOR_BGR2GRAY)
                    featurepoints = cv2.goodFeaturesToTrack(gray,1,0.1,10) # TODO: parameters can be tuned
                    if featurepoints is None:
                        featurepoints = [[[ h/2, w/2 ]]]
                    featurepoints = np.intp(featurepoints)
                    
                    # add feature point to result, optionally draw
                    for featurepoint in featurepoints:
                        x,y = featurepoint.ravel()
                        y = y + (j*h)
                        x = x + (i*w)
                        if (showfeatures) : cv2.circle(img,(x,y),3,255,-1)
                        result.append([x,y])
            
            # optionally draw features
            if (showfeatures) : 
                cv2.imshow("",img)
                cv2.waitKey(0)

        except Exception as ex : 
            print('Exception: ', ex) 
            
            #print(idx) 
        
        print(result)
        return result
 

    def generate(self, time_step=10, num_steps=0, x_scale=1, y_scale=1, show=False):
        # Our elementary time unit is seconds.
        first_ts = self.move_events[0][2]
        last_ts = self.move_events[-1][2]
        print('First timestamp: ', first_ts)
        print('Last timestamp: ', last_ts)
        print('Time period: ', str(last_ts - first_ts))
        seconds_per_day = 86400

        if num_steps > 0:
            time_step = (last_ts - first_ts) // num_steps

        print('Step ', time_step, ' = ', time_step/seconds_per_day, 'days')

        for x in range(first_ts, last_ts, time_step):
            print('Round ', x)
            ini_ts = x
            end_ts = x + time_step

            moves_filtered = self.slice_moves(ini_ts, end_ts)
            # We may end up with an empty slice of the data.
            #print(f'Processing {len(moves_filtered)} points in slice {x} ...', file=sys.stderr)
            try:
                data = self.get_coordinates_kde(moves_filtered)
            except:
                #print('Error: Could not apply KDE. Most likely got a numpy.linalg.LinAlgError', file=sys.stderr)
                continue

            #fig, axes = plt.subplots(figsize=(1280/100, 768/100), dpi=100)
            fig, axes = plt.subplots(figsize=(self.map_width/self.dpi, self.map_height/self.dpi), dpi=self.dpi)
            axes.pcolormesh(data[0], data[1], data[2].reshape(data[0].shape), shading='auto', cmap=plt.cm.jet)
            axes.invert_yaxis()
            plt.gca().set_axis_off()
            # FIXME: Why we can't set all margins to 0?
            # "ValueError: left cannot be >= right"
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            fig.canvas.draw()
            #plt.show()
            os.makedirs('plots', exist_ok = True) 
            plt.savefig(f'plots/figure_{x}_scale_{x_scale}_{y_scale}.png')
            # EDIT:
            #plt.show()
            # make smaller resolution heatmaps
            # resolution - n_bins - why does changing the amount of bins not work?

            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            img = plt.imread(buf) * 255
            img[:, :, [0, 1, 2]] = img[:, :, [2, 1, 0]]
            img_array = img.astype(np.uint8)
            img = Image.fromarray(img_array)
            self.images.append(np.array(img))
            
            polygons = self.create_polygons([data], axes, x_scale, y_scale)

            # add the 4 corners to result
            polygons = np.vstack((polygons, np.array([[0, 0], [(img_array.shape[1]-1), 0], [0, (img_array.shape[0]-1)], [(img_array.shape[1]-1), (img_array.shape[0]-1)]])))
    
            autofeatures_grid = np.array(self.autofeaturepoints(img_array))
            
            print(polygons.shape, autofeatures_grid.shape)
            polygons = np.vstack((polygons, autofeatures_grid))
            self.polygons.append(polygons)
      
            x_values = [point[0] for point in polygons]
            y_values = [point[1] for point in polygons]
            plt.scatter(x_values, y_values, s = 1)    
            #if show: 
            print("Image is displayed - close to continue")
            plt.show()

            #plt.close()  
            
            os.makedirs('coordinates', exist_ok = True) 
            textfile = open(f'coordinates/figure_{x}_scale_{x_scale}_{y_scale}-png.txt', 'w')
            for i in range(0, len(x_values)):
                textfile.write("  ")
                textfile.write(str(x_values[i]))
                textfile.write("  ")
                textfile.write(str(y_values[i]))
                textfile.write("\n")
            textfile.close()

        return self.images, self.polygons

if __name__ == '__main__':
    # TODO: Use a config file instead of CLI args, if we plan to have lots of different options.

    parser = argparse.ArgumentParser(description='Create a heatmap animation out of multiple CSV files.',
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('filenames', nargs='+', help='names of the CSV files to be used')
    parser.add_argument('--image_id', default='-', help='image ID from UEyes dataset')
    parser.add_argument('--dataset_name', default='-', help='dataset name')
    parser.add_argument('--delimiter', default=' ', help='column delimiter')
    parser.add_argument('--outfile', help='output video filename')
    parser.add_argument('--time_step', type=int, default=100, help='time step duration, in seconds')
    parser.add_argument('--num_steps', type=int, default=0, help='number of steps; overrides the --time_step option')
    parser.add_argument('--duration', type=int, default=5, help='length of the animation, in seconds')
    parser.add_argument('--fps', type=int, default=24, help='frame rate, or animation speed')
    parser.add_argument('--map_width', type=int, default=1280, help='output animation width')
    parser.add_argument('--map_height', type=int, default=768, help='output animation height')
    parser.add_argument('--dpi', type=int, default=100, help='image resolution, in dots per inch')
    parser.add_argument('--kde_bins', type=int, default=200, help='number of bins for kernel density estimation')
    parser.add_argument('--verbose', default=False, action='store_true', help='print more progress messges')
    parser.add_argument('--time_overlay', default=False, action='store_true',
                        help='display text overlay indicating when the data was recorded')
    parser.add_argument('--show', default=False, action='store_true',
                        help='show intermediate heatmaps')
    parser.add_argument('--x_scale', type=float, default=1, help='the x coordinates are divided by this scale')
    parser.add_argument('--y_scale', type=float, default=1, help='the y coordinates are divided by this scale')
    parser.add_argument('--smoothing', type=int, default=100, help="median_filter smoothing/blur to remove image artifacts, for example -smoothing 2 will blur lightly. (default: %(default)s)")
    args = parser.parse_args()


    now = time.mktime(time.gmtime())
    # Animation filename
    if not args.outfile:
        args.outfile = f'{args.dataset_name}_imId-{args.image_id}_smoothing-{args.smoothing}_{str(now)}.mp4'

    # About the CSV format:
    # - At least three columns must be provided: `X`, `Y`, and `Unix timestamp` (either in seconds or milliseconds).
    # - The X and Y coordinates are relative to the user viewport.
    move_events = []
    now = time.mktime(time.gmtime())

    
    #for filename in args.filenames:
    # EDIT: args.filenames -> glob.glob(args.filenames[0])
    # list all files matching the .csv pattern
    
    for filename in glob.glob(args.filenames[0]):
        with open(filename) as f:
            reader = csv.reader(f) #, delimiter=args.delimiter)
            for row in reader:
                #print(row)
                # TODO: Consider a 4th column (movment duration) in the future.
                x, y, t = row[0:3]
                #print(x, y, t)
                #exit()
                
                # Ignore CSV header, if present.
                try:
                    t = float(t)
                    # transform CPU timetick to Unix timestamp:
                    # convert microseconds to seconds,
                    # subtract n_seconds between 0001-00-01 00:00:00 and 1970-01-01 01:00:00
                    #t = t/10000000 #- 62136892800
                except:
                    continue

                # Always work in seconds.
                if t > now:
                    t = t / 1000.
    
                move_events.append([float(x)*1000, float(y)*1000, int(t)])
    print(move_events)

    text_overlay = ''
    if args.time_overlay:
        first_ts = move_events[0][2]
        last_ts = move_events[-1][2]
        print(first_ts)
        print(last_ts)
        display_format = '%Y-%m-%d %H:%M:%S'
        first_date = datetime.fromtimestamp(first_ts).strftime(display_format)
        last_date = datetime.fromtimestamp(last_ts).strftime(display_format)
        text_overlay = f'{first_date} --> {last_date}'
        print(text_overlay)

    #

    hf = HeatmapFrames(move_events, args.map_width, args.map_height, args.kde_bins)
    images, polygons = hf.generate(time_step=args.time_step, num_steps=args.num_steps, x_scale=args.x_scale, y_scale=args.y_scale, show = args.show)

    # TODO: Export frames to PNG files so that we can try other blending techniques offline.

    ac = AnimationCreator(images, polygons, args.duration, args.fps, text_overlay, args.smoothing)
    ac.save(args.outfile)