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
import datetime
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.path import Path
from scipy.interpolate import RectBivariateSpline
from scipy.spatial import Delaunay
from scipy.stats import gaussian_kde
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
        self.leftTriangles = left_triangles
        self.rightImage = np.ndarray.copy(right_image)
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

        blend_arr = ((1 - alpha) * self.leftImage + alpha * self.rightImage)
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
            self.leftImage[int(x[1])][int(x[0])] = self.leftInterpolation(y[1], y[0])
            self.rightImage[int(x[1])][int(x[0])] = self.rightInterpolation(z[1], z[0])


class AnimationCreator:
    def __init__(self, images, polygons, duration=3, fps=25, text=''):
        self.images = images
        self.polygons = polygons
        self.duration = duration
        self.fps = fps
        self.num_morph_frames = fps // duration # FIXME
        self.overlay_text = text


    def init_morph(self, start_image_path, end_image_path, polygons):
        left_image_arr = start_image_path

        right_image_raw = end_image_path
        right_image_raw = cv2.resize(right_image_raw, (left_image_arr.shape[1], left_image_arr.shape[0]), interpolation=cv2.INTER_CUBIC)
        right_image_arr = np.asarray(right_image_raw)

        triangle_tuple = polygons

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
            out_image = np.dstack([
                np.array(morphers[0].get_image_at_alpha(alpha)),
                np.array(morphers[1].get_image_at_alpha(alpha)),
                np.array(morphers[2].get_image_at_alpha(alpha)),
            ])

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
                self.init_morph(self.images[idx], self.images[idx + 1], (self.polygons[idx], self.polygons[idx + 1]))
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

        xi, yi = np.mgrid[0:self.map_width:self.kde_bins * 1j, 0:self.map_height:self.kde_bins * 1j]

        kernel = gaussian_kde(my_data)
        zi = kernel(np.vstack([xi.flatten(), yi.flatten()]))

        return [xi, yi, zi]


    def create_polygons(self, data, axes):
        plt.gca().set_axis_off()
        # FIXME: Why we can't set all margins to 0?
        # "ValueError: left cannot be >= right"
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)

        xi, yi, zi = data[0][0], data[0][1], data[0][2]
        cs = axes.contourf(xi, yi, zi.reshape(xi.shape), levels=4)

        paths_col = cs.collections
        collections = paths_col[len(paths_col) - 2:len(paths_col)]
        polygons_all = []

        for collection in collections:
            paths = collection.get_paths()
            for path in paths:
                polygons = path.to_polygons()
                for pol in polygons[0:1]:
                    for coord in pol:
                        x, y = coord[0:2]
                        plt.plot(x, y)
                polygons_all.append(polygons)

        # FIXME: Why x/10? What does it mean?
        polygons = [x/10 for y in polygons_all for x in y]
        simplices = polygons[0][Delaunay(polygons[0]).simplices]

        return [Triangle(p) for p in simplices]


    def generate(self, time_step=10, num_steps=0):
        # Our elementary time unit is seconds.
        first_ts = self.move_events[0][2]
        last_ts = self.move_events[-1][2]

        if num_steps > 0:
            time_step = (last_ts - first_ts) // num_steps

        for x in range(first_ts, last_ts, time_step):
            ini_ts = x
            end_ts = x + time_step

            moves_filtered = self.slice_moves(ini_ts, end_ts)
            # We may end up with an empty slice of the data.
            if not moves_filtered:
                continue

            print(f'Processing {len(moves_filtered)} points in slice {x} ...', file=sys.stderr)
            try:
                data = self.get_coordinates_kde(moves_filtered)
            except:
                print('Error: Could not apply KDE. Most likely got a numpy.linalg.LinAlgError', file=sys.stderr)
                continue

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

            self.polygons.append(self.create_polygons([data], axes))

            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            img = plt.imread(buf) * 255
            img[:, :, [0, 1, 2]] = img[:, :, [2, 1, 0]]
            img = img.astype(np.uint8)
            img = Image.fromarray(img)

            self.images.append(np.array(img))
            plt.close()

        return self.images, self.polygons


if __name__ == '__main__':
    # TODO: Use a config file instead of CLI args, if we plan to have lots of different options.

    parser = argparse.ArgumentParser(description='Create a heatmap animation out of multiple CSV files.',
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('filenames', nargs='+', help='names of the CSV files to be used')
    parser.add_argument('--delimiter', default=' ', help='column delimiter')
    parser.add_argument('--outfile', help='output video filename')
    parser.add_argument('--time_step', type=int, default=10, help='time step duration, in seconds')
    parser.add_argument('--num_steps', type=int, default=0, help='number of steps; overrides the --time_step option')
    parser.add_argument('--duration', type=int, default=3, help='length of the animation, in seconds')
    parser.add_argument('--fps', type=int, default=60, help='frame rate, or animation speed')
    parser.add_argument('--map_width', type=int, default=1280, help='output animation width')
    parser.add_argument('--map_height', type=int, default=768, help='output animation height')
    parser.add_argument('--dpi', type=int, default=100, help='image resolution, in dots per inch')
    parser.add_argument('--kde_bins', type=int, default=50, help='number of bins for kernel density estimation')
    parser.add_argument('--verbose', default=False, action='store_true', help='print more progress messges')
    parser.add_argument('--time_overlay', default=False, action='store_true',
                        help='display text overlay indicating when the data was recorded')
    args = parser.parse_args()

    if not args.outfile:
        args.outfile = f'hb_{int(time.time())}.mp4'

    # About the CSV format:
    # - At least three columns must be provided: `X`, `Y`, and `Unix timestamp` (either in seconds or milliseconds).
    # - The X and Y coordinates are relative to the user viewport.
    move_events = []
    now = time.mktime(time.gmtime())

#    for filename in args.filenames:
    for filename in glob.glob(args.filenames[0])[:5]:
        with open(filename) as f:
            reader = csv.reader(f, delimiter=args.delimiter)
            for row in reader:
                # TODO: Consider a 4th column (movment duration) in the future.
                x, y, t = row[0:3]

                # Ignore CSV header, if present.
                try:
                    t = float(t)
                except:
                    continue

                # Always work in seconds.
                if t > now:
                    t = t / 1000.

                move_events.append([float(x), float(y), int(t)])


    text_overlay = ''
    if args.time_overlay:
        first_ts = move_events[0][2]
        last_ts = move_events[-1][2]
        display_format = '%Y-%m-%d %H:%M:%S'
        first_date = datetime.fromtimestamp(first_ts).strftime(display_format)
        last_date = datetime.fromtimestamp(last_ts).strftime(display_format)
        text_overlay = f'{first_date} --> {last_date}'


    hf = HeatmapFrames(move_events, args.map_width, args.map_height, args.kde_bins)
    images, polygons = hf.generate(time_step=args.time_step, num_steps=args.num_steps)

    # TODO: Export frames to PNG files so that we can try other blending techniques offline.

    ac = AnimationCreator(images, polygons, args.duration, args.fps, text_overlay)
    ac.save(args.outfile)
