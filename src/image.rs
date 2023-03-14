use std::{fs::File, path::Path, io::BufWriter};

use anyhow::Result;
use ndarray::{Array2, Array};
use pix::{el::Pixel, Raster};
use png_pong::{Decoder, Encoder};

pub type Image = Array2<[u16; 4]>;
// Load an image from a path, converts results to RGB by default.
pub fn load_image_from_path(path: &String) -> Result<Image> {
    let file = File::open(path)?;
    let decoder = Decoder::new(file)?.into_steps();
    let png_pong::Step { raster, .. } = decoder.last()
        .expect("No frames in PNG")?;

    
    // Convert all formats into the same RGBA16 internal format
    let image: Array2<[u16; 4]> = match raster {
        png_pong::PngRaster::Gray8(raster) => {
            println!("G8");
            let width = raster.width() as usize;
            let height = raster.height() as usize;
            Array::from_shape_vec((height, width),
                raster.pixels().iter()
                .map(| pixel | [pixel.one().into(), pixel.one().into(), pixel.one().into(), 0xff])
                .map(| pixel| pixel.map(u16::from))
                .collect()
            )?
        },
        png_pong::PngRaster::Gray16(_) => {
            println!("G16");
            todo!()
        },
        png_pong::PngRaster::Rgb8(raster) => {
            println!("RGB8");
            let width = raster.width() as usize;
            let height = raster.height() as usize;
            Array::from_shape_vec((height, width),
                raster.pixels().iter()
                .map(| pixel | [pixel.one().into(), pixel.two().into(), pixel.three().into(), 0xff])
                .map(| pixel| pixel.map(u16::from))
                .collect()
            )?
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
            Array::from_shape_vec((height, width),
                raster.pixels().iter()
                .map(| pixel | [pixel.one().into(), pixel.two().into(), pixel.three().into(), pixel.four().into()])
                .map(| pixel: [u8; 4]| pixel.map(u16::from))
                .collect()
            )?
        },
        png_pong::PngRaster::Rgba16(_) => {
            println!("RGBA16");
            todo!()
        },
    };

    Ok(image)
}

pub fn save_image_to_path(image: &Image, path: &Path) -> Result<()> {
    let (height, width) = image.dim();
    let buffer: Vec<u16> = image.iter().flat_map(| e | e).copied().collect();
    let raster: Raster<pix::rgb::SRgba16> = Raster::with_u16_buffer(width as u32, height as u32, buffer);

    let writer = BufWriter::new(File::create(path)?);
    let mut encoder = Encoder::new(writer).into_step_enc();
    encoder.still(&raster)?;
    Ok(())
}
