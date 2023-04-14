# Collapse

A pure Rust implementation of the wave-function collapse algorithm.
Inspired by [WaveFunctionCollapse](https://github.com/mxgmn/WaveFunctionCollapse) by [Maxim Gumin](https://github.com/mxgmn).

## Installation

You need a running Rust install and the rest is just done via cargo. Building the project is as easy as running `cargo build`.
If you want to run the project directly, you can do so via `cargo run`.
Note, since the program requires parameters as CLI-Arguments, you need to add an extra `--`, to tell cargo to pass them along to the program itself, so: `cargo run -- <Arguments>`

## Arguments

Program configuration is provided via arguments:

- **-s** / **--sample-path** path to the image to sample patterns from
- **--result-width** width of the output image(s) in px, defaults to 64
- **--result-height** height of the output image(s) in px, defaults to 64
- **-n** controls the pattern size, defaults to 3

Note: only pngs are supported as samples, for now.

For example, to generate a 32x32px image from the sample `image.png` with a pattern size of 3, you would use the following command: ```cargo run -- -s image.png --result-width 32 --result-height 32 -n 3```.

Note that running in debug mode (the default for cargo run) may be quite slow, so you can either build the program and run it, or go with release mode for cargo run: ```cargo run --release -- -s image.png --result-width 32 --result-height 32 -n 3```.

Intermediate steps are written to `${CWD}/steps/iteration_{idx}`. You can render them into a nice video with FFMPEG for example: ```ffmpeg -framerate 60 -pattern_type sequence -start_number 0 -i "<PATH_TO_STEPS_DIRECTORY>/iteration_%d.png" <VIDEO_FILENAME>```.

## Known Issues

- The algorithm is still very slow
