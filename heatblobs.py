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
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.path import Path
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import median_filter
from scipy.spatial import Delaunay
from scipy.stats import gaussian_kde
import scipy
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
        self.minX = int(self.vertices[:, 0].min())
        self.maxX = int(self.vertices[:, 0].max())
        self.minY = int(self.vertices[:, 1].min())
        self.maxY = int(self.vertices[:, 1].max())


    def get_points(self):
        x_list = range(self.minX, self.maxX + 1)
        y_list = range(self.minY, self.maxY + 1)
        empty_list = list((x, y) for x in x_list for y in y_list)

        points = np.array(empty_list, np.float64)
        p = Path(self.vertices)
        grid = p.contains_points(points)
        mask = grid.reshape(self.maxX - self.minX + 1, self.maxY - self.minY + 1)

        true_array = np.where(np.array(mask) == True)
        coord_array = np.vstack((true_array[0] + self.minX, true_array[1] + self.minY, np.ones(true_array[0].shape[0])))

        return coord_array


class Morpher:
    def __init__(self, left_image, left_triangles, right_image, right_triangles):
        if type(left_image) != np.ndarray:
            raise TypeError('Input leftImage is not an np.ndarray')
        if left_image.dtype != np.uint8:
            raise TypeError('Input leftImage is not of type np.uint8')
        if type(right_image) != np.ndarray:
            raise TypeError('Input rightImage is not an np.ndarray')
        if right_image.dtype != np.uint8:
            raise TypeError('Input rightImage is not of type np.uint8')
        if type(left_triangles) != list:
            raise TypeError('Input leftTriangles is not of type List')
        for j in left_triangles:
            if isinstance(j, Triangle) == 0:
                raise TypeError('Element of input leftTriangles is not of Class Triangle')
        if type(right_triangles) != list:
            raise TypeError('Input leftTriangles is not of type List')
        for k in right_triangles:
            if isinstance(k, Triangle) == 0:
                raise TypeError('Element of input rightTriangles is not of Class Triangle')

        self.leftImage = np.ndarray.copy(left_image)
        self.newLeftImage = np.ndarray.copy(left_image)
        self.leftTriangles = left_triangles
        self.rightImage = np.ndarray.copy(right_image)
        self.newRightImage = np.ndarray.copy(right_image)
        self.rightTriangles = right_triangles
        self.leftInterpolation = RectBivariateSpline(np.arange(self.leftImage.shape[0]),
                                                     np.arange(self.leftImage.shape[1]),
                                                     self.leftImage)
        self.rightInterpolation = RectBivariateSpline(np.arange(self.rightImage.shape[0]),
                                                      np.arange(self.rightImage.shape[1]),
                                                      self.rightImage)

    def get_image_at_alpha(self, alpha):
        for leftTriangle, rightTriangle in zip(self.leftTriangles, self.rightTriangles):
            self.interpolate_points(leftTriangle, rightTriangle, alpha)

        blend_arr = ((1 - alpha) * self.newLeftImage + alpha * self.newRightImage)
        blend_arr = blend_arr.astype(np.uint8)
        return blend_arr


    def interpolate_points(self, left_triangle, right_triangle, alpha):
        target_triangle = Triangle(left_triangle.vertices + (right_triangle.vertices - left_triangle.vertices) * alpha)
        target_vertices = target_triangle.vertices.reshape(6, 1)
        temp_left_matrix = np.array([[left_triangle.vertices[0][0], left_triangle.vertices[0][1], 1, 0, 0, 0],
                                     [0, 0, 0, left_triangle.vertices[0][0], left_triangle.vertices[0][1], 1],
                                     [left_triangle.vertices[1][0], left_triangle.vertices[1][1], 1, 0, 0, 0],
                                     [0, 0, 0, left_triangle.vertices[1][0], left_triangle.vertices[1][1], 1],
                                     [left_triangle.vertices[2][0], left_triangle.vertices[2][1], 1, 0, 0, 0],
                                     [0, 0, 0, left_triangle.vertices[2][0], left_triangle.vertices[2][1], 1]])
        temp_right_matrix = np.array([[right_triangle.vertices[0][0], right_triangle.vertices[0][1], 1, 0, 0, 0],
                                      [0, 0, 0, right_triangle.vertices[0][0], right_triangle.vertices[0][1], 1],
                                      [right_triangle.vertices[1][0], right_triangle.vertices[1][1], 1, 0, 0, 0],
                                      [0, 0, 0, right_triangle.vertices[1][0], right_triangle.vertices[1][1], 1],
                                      [right_triangle.vertices[2][0], right_triangle.vertices[2][1], 1, 0, 0, 0],
                                      [0, 0, 0, right_triangle.vertices[2][0], right_triangle.vertices[2][1], 1]])

        lefth = np.linalg.solve(temp_left_matrix, target_vertices)
        righth = np.linalg.solve(temp_right_matrix, target_vertices)
        leftH = np.array([[lefth[0][0], lefth[1][0], lefth[2][0]], [lefth[3][0], lefth[4][0], lefth[5][0]], [0, 0, 1]])
        rightH = np.array([[righth[0][0], righth[1][0], righth[2][0]], [righth[3][0], righth[4][0], righth[5][0]], [0, 0, 1]])
        leftinvH = np.linalg.pinv(leftH)
        rightinvH = np.linalg.pinv(rightH)
        target_points = target_triangle.get_points()

        left_source_points = np.transpose(np.matmul(leftinvH, target_points))
        right_source_points = np.transpose(np.matmul(rightinvH, target_points))
        target_points = np.transpose(target_points)

        for x, y, z in zip(target_points, left_source_points, right_source_points):
            if int(x[1]) >=  self.newLeftImage.shape[0]:
                x[1] = x[1]-1
            self.newLeftImage[int(x[1])][int(x[0])] = self.leftInterpolation(y[1], y[0])
            self.newRightImage[int(x[1])][int(x[0])] = self.rightInterpolation(z[1], z[0])


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
        contour = scipy.signal.resample(contour, 200)

        return contour


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
            img = img.astype(np.uint8)
            img = Image.fromarray(img)
            self.images.append(np.array(img))
            
            polygons = self.create_polygons([data], axes, x_scale, y_scale)
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

    if not args.outfile:
        args.outfile = f'{args.dataset_name}_imId-{args.image_id}_smoothing-{args.smoothing}.mp4'

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