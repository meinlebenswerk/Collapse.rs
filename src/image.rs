use std::{fs::File, path::Path, io::BufWriter};

use anyhow::Result;
use pix::{el::Pixel, Raster};
use png_pong::{Decoder, Encoder};

pub struct Image {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<[u16; 4]>
}


pub struct PaletteImage {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<usize>
}


// Load an image from a path, converts results to RGB by default.
pub fn load_image_from_path(path: &String) -> Result<Image> {
    let file = File::open(path)?;
    let decoder = Decoder::new(file)?.into_steps();
    let png_pong::Step { raster, .. } = decoder.last()
        .expect("No frames in PNG")?;

    
    // Convert all formats into the same RGBA16 internal format
    let image = match raster {
        png_pong::PngRaster::Gray8(raster) => {
            println!("G8");
            let width = raster.width() as usize;
            let height = raster.height() as usize;
            let pixels = raster.pixels().iter()
                .map(| pixel | [pixel.one().into(), pixel.one().into(), pixel.one().into(), 0xff])
                .map(| pixel| pixel.map(u16::from))
                .collect();
            Image {
                width,
                height,
                pixels
            }
        },
        png_pong::PngRaster::Gray16(_) => {
            println!("G16");
            todo!()
        },
        png_pong::PngRaster::Rgb8(raster) => {
            println!("RGB8");
            let width = raster.width() as usize;
            let height = raster.height() as usize;
            let pixels = raster.pixels().iter()
                .map(| pixel | [pixel.one().into(), pixel.two().into(), pixel.three().into(), 0xff])
                .map(| pixel| pixel.map(u16::from))
                .collect();
            Image {
                width,
                height,
                pixels
            }
        },
        png_pong::PngRaster::Rgb16(_) => {
            println!("RGB16");
            todo!()
        },
        png_pong::PngRaster::Palette(_, _, _) => {
            println!("Palette");
            todo!()
        },
        png_pong::PngRaster::Graya8(_) => {
            println!("Graya8");
            todo!()
        },
        png_pong::PngRaster::Graya16(_) => {
            println!("Graya16");
            todo!()
        },
        png_pong::PngRaster::Rgba8(raster) => {
            println!("RGBA8");
            let width = raster.width() as usize;
            let height = raster.height() as usize;
            let pixels = raster.pixels().iter()
                .map(| pixel | [pixel.one().into(), pixel.two().into(), pixel.three().into(), pixel.four().into()])
                .map(| pixel: [u8; 4]| pixel.map(u16::from))
            .collect();
            Image {
                width,
                height,
                pixels
            }
        },
        png_pong::PngRaster::Rgba16(raster) => {
            println!("RGBA16");
            let width = raster.width() as usize;
            let height = raster.height() as usize;
            let pixels = raster.pixels().iter()
                .map(| pixel | [pixel.one().into(), pixel.two().into(), pixel.three().into(), pixel.four().into()])
            .collect();
            Image {
                width,
                height,
                pixels
            }
        },
    };

    Ok(image)
}

pub fn save_image_to_path(image: &Image, path: &Path) -> Result<()> {
    let Image{height, width, .. } = image;
    let buffer: Vec<u16> = image.pixels.iter().flatten().copied().collect();
    let raster: Raster<pix::rgb::SRgba16> = Raster::with_u16_buffer(*width as u32, *height as u32, buffer);

    let writer = BufWriter::new(File::create(path)?);
    let mut encoder = Encoder::new(writer).into_step_enc();
    encoder.still(&raster)?;
    Ok(())
}
