# HeatBlobs

Generate animated heatmaps.

## Installation

It is highly recommended to use a virtual environment (e.g. conda, pipenv, virtualenv, etc.)

```
~$ pip install -r requirements.txt
```

## Data format

It is assumed that the data to be plotted (e.g. eye or mouse movements) is in CSV format,
space-delimited, with the following columns: `x y time` or `x y time duration`.

**Note:** It is recommendend to set `--map_width` and `--map_height` to the size of the viewport where you ran the experiments. 

For online experiments, where each user may have a different browser resolution, it is recommended to normalize the X coordinates,
so that the resulting heatmaps are comparable across users. Note that the Y coordinates should *not* be normalized, since browsers add scrollbars in vertical.

## Usage

Defaults:
```
~$ python heatblobs.py /path/to/*.csv
```

show help:
```
~$ python heatblobs.py -h
usage: heatblobs.py [-h] [--delimiter DELIMITER] [--outfile OUTFILE] [--duration DURATION] [--fps FPS] [--map_width MAP_WIDTH] [--map_height MAP_HEIGHT] [--kde_bins KDE_BINS] [--time_overlay]
                    filenames [filenames ...]

Create a heatmap animation out of multiple CSV files.

positional arguments:
  filenames             names of the CSV files to be used

options:
  -h, --help            show this help message and exit
  --delimiter DELIMITER
                        column delimiter (default: )
  --outfile OUTFILE     output video filename (default: None)
  --duration DURATION   length of the animation, in seconds (default: 3)
  --fps FPS             frame rate, or animation speed (default: 200)
  --map_width MAP_WIDTH
                        output animation width (default: 1280)
  --map_height MAP_HEIGHT
                        output animation height (default: 768)
  --kde_bins KDE_BINS   number of bins for kernel density estimation (default: 200)
  --time_overlay        display text overlay indicating when the data was recorded (default: False)
```

