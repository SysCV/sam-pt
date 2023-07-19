# Demo Data

This directory contains demo data that users can use to understand the structure and format of input data. Below, we've detailed the sources of our demo data and provided an in-depth explanation of the query points format.

## Data Sources

The provided clips in this directory serve as sample data for the demo and were obtained from Pixabay:

1. [`street.mp4`](.street.mp4) - [Video source](https://pixabay.com/videos/street-bus-village-bus-stop-city-38590/).
2. [`bees.mp4`](bees.mp4) - [Video source](https://pixabay.com/videos/bees-honey-bees-insect-pollen-35093/).

## Query Points Format

Query points are crucial for our application as they define the target object (positive points) and the background/non-target objects (negative points). 

They can be provided interactively by the user or derived from a ground truth mask. The following section explains how they're structured when saved to a text file:

```bash
number_of_positive_points
mask_1_timestep ; pos_x_1,pos_y_1 ... pos_x_n,pos_y_n neg_x_1,neg_y_1 ... neg_x_m,neg_y_m
mask_2_timestep ; pos_x_1,pos_y_1 ... pos_x_n,pos_y_n neg_x_1,neg_y_1 ... neg_x_m,neg_y_m
...
```

- `number_of_positive_points` - Specifies the number of positive points
- `mask_x_timestep` - The timestamp for each mask
- `pos_x_i,pos_y_i` - x, y coordinates of the positive points
- `neg_x_i,neg_y_i` - x, y coordinates of the negative points

Note: The number of negative points is inferred from the total number of points minus the number of positive points.

Here is a simple example of a query point file with two masks:

```sh
1
0 ;      10,20       30,30   40,40
4 ; 123.123,456.456  72,72    5,6
```

In this example, each mask has one positive point and two negative points. The positive query point for the first mask, for instance, has (x,y) coordinates of (10,20). Here, the value '10' denotes a distance of 10 pixels from the left image border, and '20' indicates a distance of 20 pixels from the top image border (as the coordinate system begins at the top left corner of the image).
