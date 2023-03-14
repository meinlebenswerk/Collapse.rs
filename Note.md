ffmpeg -framerate 20 -pattern_type sequence -start_number 0 -i "P:\Projects\Rust\collapse\steps\iteration_%d.png" -c:v libx264 -pix_fmt yuv420p collapse.mp4

#(Maybe) Better entropy metrics
# Fix the generation, what we're doing currently seems weird.


cargo flamegraph  -- -s .\samples\Flowers.png -n 3 --result-width 64 --result-height 64
cargo run --release -- -s .\samples\Flowers.png -n 3 --result-width 256 --result-height 256


Bing bong
It's working so now faster!
A lot faster!
Move from the wavefunction float thing to something like a compressed bit vec!

# With 4x4 upscaling and nearest neighbor upscaling
ffmpeg -framerate 60 -pattern_type sequence -start_number 0 -i "P:\Projects\Rust\collapse\steps\iteration_%d.png" -s 512x512 -sws_flags neighbor flowers.avi