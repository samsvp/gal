# GAL - Genetic Algorithm Painter

A genetic algorithm implemented using ArrayFire to paint images.

## Dependencies

- [ArrayFire](https://github.com/arrayfire/arrayfire) + [Forge](https://github.com/arrayfire/forge).
- OpenCL: OpenCL installation depends on OS and GPU/CPU manufactorer.

## Build

To build run

```
mkdir build && cd build
cmake ..
make -j8
```

## Usage
To use the painter run
```
./main <img_source> <brush_path>
```

Where image_source is the path of the image you want to paint and brush path
is the path of the brush (a png image) which the algorithm should use to paint with.
The folder `brushes` has a few preset brushes. 
