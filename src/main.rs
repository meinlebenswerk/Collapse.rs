use anyhow::Result;
use bitflags::bitflags;
use clap::Parser;
use image::PaletteImage;
use rand_wyrand::WyRand;
use std::{
    collections::HashMap,
    convert::TryInto,
    fs,
    path::Path,
};

use itertools::{iproduct};
use rand::{distributions::WeightedIndex, SeedableRng, Rng};
use rayon::prelude::{
    IntoParallelRefIterator, ParallelBridge,
    ParallelIterator,
};

use bitvec::prelude::*;

mod cli;
mod image;
use crate::image::Image;
use crate::{
    cli::CLIArguments,
    image::{load_image_from_path, save_image_to_path},
};

type BitvecUnderlyingType = u32;

bitflags! {
    struct Symmetry: u8 {
        const ROTATE_90         = 0b00000001;
        const ROTATE_180        = 0b00000010;
        const ROTATE_270        = 0b00000100;
        const FLIP_X            = 0b00001000;
        const FLIP_Y            = 0b00010000;
        const FLIP_X_ROTATE_90  = 0b00100000;
        const FLIP_Y_ROTATE_90  = 0b01000000;
    }
}

#[derive(Eq, PartialEq, Hash, Debug, Clone)]
struct Pattern {
    size: usize,
    pixels: Vec<usize>
}

impl Pattern {
    pub fn from_image_and_offset(image: &PaletteImage, (y, x): (usize, usize), n: usize) -> Self {
        let PaletteImage { width, .. } = image;
        let pixels = iproduct!(y..y+n, x..x+n)
            .map(|(y, x)| {
                let index = y*width + x;
                image.pixels[index]
            })
            .collect();

        Self { size: n, pixels }
    }

    fn _mask_with_offset(&self, (offset_y, offset_x): &(i32, i32)) -> Vec<usize> {
        let size = self.size.try_into().unwrap();
        if offset_x.abs() > size || offset_y.abs() > size {
            return Vec::new();
        }
        let range_x = (0.max(*offset_x) as usize)..((offset_x + size).min(size) as usize);
        let range_y = (0.max(*offset_y) as usize)..((offset_y + size).min(size) as usize);

        // Slice the thing
        iproduct!(range_y, range_x)
            .map(|(y, x)| self.pixels[y*self.size + x])
            .collect()
    }

    pub fn check_for_overlap(&self, other: &Pattern, offset: &(i32, i32)) -> bool {
        let masked_values_self = self._mask_with_offset(offset);
        let masked_values_other = other._mask_with_offset(&(-offset.0, -offset.1));
        masked_values_self == masked_values_other
    }

    // Symmetry helpers
    fn rotate_90deg(&self) -> Self {
        let pixels: Vec<_> = iproduct!(0..self.size, 0..self.size)
            .map(|(y, x)| {
                // Transposed indices
                let index = x * self.size + y;
                self.pixels[index]
            })
            .collect();

        Self {
            size: self.size,
            pixels
        }
    }
    
    fn rotate_180deg(&self) -> Self {
        self.rotate_90deg().rotate_90deg()
    }
    
    fn rotate_270deg(&self) -> Self {
        self.rotate_90deg().rotate_90deg().rotate_90deg()
    }

    fn flip_x(&self) -> Self {
        let pixels: Vec<_> = iproduct!(0..self.size, 0..self.size)
            .map(|(y, x)| {
                let index = y * self.size + (self.size - x - 1) ;
                self.pixels[index]
            })
            .collect();

        Self {
            size: self.size,
            pixels
        }
    }

    fn flip_y(&self) -> Self {
        let pixels: Vec<_> = iproduct!(0..self.size, 0..self.size)
            .map(|(y, x)| {
                let index = (self.size - y - 1) * self.size + x;
                self.pixels[index]
            })
            .collect();

        Self {
            size: self.size,
            pixels
        }
    }
}

#[derive(PartialEq)]
enum ObserveResult {
    Ok(Vec<(usize, usize)>, f64),
    Contradiction,
    Collapse,
}

fn generate_pattern_offsets(n: i32) -> Vec<(i32, i32)> {
    iproduct!(-n + 1..n, -n + 1..n)
        .par_bridge()
        .filter(|(y, x)| !(*y == 0 && *x == 0))
        .collect()
}

fn generate_patterns_frequencies_probabilities_from_image(
    sample: &PaletteImage,
    pattern_size: usize,
    symmetry: Symmetry
) -> (Vec<Pattern>, Vec<usize>, Vec<f64>) {
    let mut pattern_map: HashMap<Pattern, usize> = HashMap::new();

    let PaletteImage{ width, height, .. } = sample;

    for y in 0..(height - pattern_size) {
        for x in 0..(width - pattern_size) {
            // Base pattern
            let pattern = Pattern::from_image_and_offset(sample, (y, x), pattern_size);
            
            // Symmetries, if required
            if symmetry.contains(Symmetry::ROTATE_90) {
                let sym_pattern = pattern.rotate_90deg();
                *pattern_map.entry(sym_pattern).or_insert(0) += 1;
            }
            if symmetry.contains(Symmetry::ROTATE_180) {
                let sym_pattern = pattern.rotate_180deg();
                *pattern_map.entry(sym_pattern).or_insert(0) += 1;
            }
            if symmetry.contains(Symmetry::ROTATE_270) {
                let sym_pattern = pattern.rotate_270deg();
                *pattern_map.entry(sym_pattern).or_insert(0) += 1;
            }
            if symmetry.contains(Symmetry::FLIP_X) {
                let sym_pattern = pattern.flip_x();
                *pattern_map.entry(sym_pattern).or_insert(0) += 1;
            }
            if symmetry.contains(Symmetry::FLIP_Y) {
                let sym_pattern = pattern.flip_y();
                *pattern_map.entry(sym_pattern).or_insert(0) += 1;
            }
            if symmetry.contains(Symmetry::FLIP_X_ROTATE_90) {
                let sym_pattern = pattern.flip_x().rotate_90deg();
                *pattern_map.entry(sym_pattern).or_insert(0) += 1;
            }
            if symmetry.contains(Symmetry::FLIP_Y_ROTATE_90) {
                let sym_pattern = pattern.flip_y().rotate_90deg();
                *pattern_map.entry(sym_pattern).or_insert(0) += 1;
            }

            // Insert base pattern into pattern_map
            *pattern_map.entry(pattern).or_insert(0) += 1;
        }
    }
    let patterns = pattern_map.keys().into_iter().cloned().collect();
    let frequencies: Vec<_> = pattern_map.values().into_iter().cloned().collect();

    let frequency_sum: f64 = frequencies.iter().map(|e| *e as f64).sum();
    let probabilities = frequencies.iter().map(|e| (*e as f64) / frequency_sum).collect();

    (patterns, frequencies, probabilities)
}

/// Generate the rules for a given set of patterns, output is a list for ea. pattern, which contains a list of Rules
fn generate_rules(patterns: &Vec<Pattern>, offsets: &[(i32, i32)]) -> Vec<Vec<Vec<usize>>> {
    patterns
        .par_iter()
        .map(|pattern| {
            // Now, for each offset and other pattern, we need to check for overlaps
            offsets
                .iter()
                .map(|offset| {
                    let compatible_indices = patterns
                        .iter()
                        // Add an index
                        .enumerate()
                        // Filter out non-overlapping patterns
                        .filter(|(_, other_pattern)| {
                            pattern.check_for_overlap(other_pattern, offset)
                        })
                        // Sort out and collect only the compatible indices
                        .map(|(pattern_index, _)| pattern_index)
                        .collect();
                    compatible_indices
                })
                .collect()
        })
        .collect()
}

fn render_image_from_coefficients(
    coefficient_matrix: &BitVec<u32, Msb0>,
    patterns: &[Pattern],
    palette: &[[u16; 4]],
    output_shape: &(usize, usize, usize),
    pattern_size: usize
) -> Image {
    let height = output_shape.0;
    let width = output_shape.1;

    // Iterate over the whole image, the patterns are overlayed!
    // This means the result is 1px larger in each dim than the coeff_matrix, I think.
    // Essentially, the results of the patterns are shingled over the image
    // Is that true tho, In the other calculations, we think about the pattern's center
    // Reconstructing that should be possible as well
    // For a 3x3, 2x2, 4x4 we need 1px extra on each side, I think this generalized to any other size as well
    // Only for a 1x1 pattern, this doesnt not hold, but the whole algorithm breaks down.

    let pattern_index_matrix: Vec<Vec<&Pattern>> = iproduct!(0..output_shape.0, 0..output_shape.1)
        .map(| (y, x)| {
            let start_index = ((y * output_shape.1) + x) * output_shape.2;
            coefficient_matrix[start_index..start_index + output_shape.2]
                .iter()
                .enumerate()
                // Filter out zero elements
                .filter(|(_, mask)| **mask)
                // Map indices to patterns
                .map(|(index, _)| &patterns[index])
                .collect::<Vec<&Pattern>>()
        }).collect();
    
    let preimage_width = width + pattern_size - 1;
    let preimage_height =  height + pattern_size - 1;

    let mut preimage: Vec<Vec<usize>> = vec![Vec::new(); preimage_width*preimage_height];

    // Shingle the patterns out
    for y in 0..height {
        for x in 0..width {
            // Get the pattern
            let patterns = &pattern_index_matrix[(y*width) + x];
            // let patterns = pattern_index_matrix.get((y, x)).unwrap();
            for yy in 0..pattern_size {
                for xx in 0..pattern_size {
                    let mut pixel_indices: Vec<usize> = patterns
                        .iter()
                        .map(|pattern| pattern.pixels[(yy*pattern.size + xx)])
                        .collect();
                    let preimage_index = (y+yy)*preimage_width + (x+xx);
                    preimage[preimage_index].append(&mut pixel_indices);
                }
            }
        }
    }

    // Collapse the preimage!
    let collapsed_pixels = preimage
        .iter()
        .map(| indices | {
            let n = indices.len();
            indices
                .iter()
                // Map to pixels
                .map(|&index| palette[index].map(f64::from))
                // Accumulate (This version requires unstable code :))
                // .fold([0.0; 4], | acc, e | acc.zip(e).map(| a, b | a + b))
                .fold([0.0; 4], |acc, e| {
                    [acc[0] + e[0], acc[1] + e[1], acc[2] + e[2], acc[3] + e[3]]
                })
                // Normalize! and cast back
                .map(|e| (e / n as f64) as u16)
        })
        .collect();
    
    Image {
        width: preimage_width,
        height: preimage_height,
        pixels: collapsed_pixels
    }
}

// Helper methods
fn in_bounds(position: &(usize, usize), bounds: &(usize, usize, usize)) -> bool {
    position.0 > 0 && position.0 < bounds.0 && position.1 > 0 && position.1 < bounds.1
}

fn is_cell_collapsed(
    (y, x): &(usize, usize),
    coefficient_matrix: &mut BitVec<u32, Msb0>,
    output_shape: &(usize, usize, usize),
) -> bool {
    let start_index = ((y * output_shape.1) + x) * output_shape.2;
    coefficient_matrix[start_index..start_index + output_shape.2].count_ones() == 1
}

// Main loop
fn collapse_single_cell(
    (y, x): &(usize, usize),
    coefficient_matrix: &mut BitVec<u32, Msb0>,
    probabilities: &[f64],
    rng: &mut WyRand,
    output_shape: &(usize, usize, usize),
) {
    // Get the possible choices at the given position
    let start_index = ((y * output_shape.1) + x) * output_shape.2;
    let possible_choices = &coefficient_matrix[start_index..start_index + output_shape.2];

    // Use them to weight to probabilities and renormalize the resulting prob-vector (so it sums up to 1)
    let weighted_probabilities: Vec<f64> = possible_choices
        .iter()
        .zip(probabilities.iter())
        // This also doesn't work
        // .par_bridge()
        .map(|(mask, &probability)| if *mask { probability } else { 0.0 })
        .collect();

    let weighted_probabilities_sum: f64 = weighted_probabilities.par_iter().sum();
    let weighted_probabilities: Vec<f64> = weighted_probabilities
        .par_iter()
        .map(|e| e / weighted_probabilities_sum)
        .collect();

    // Construct a distribution from the probabilties
    let distribution = WeightedIndex::new(&weighted_probabilities).unwrap();
    let picked_collapse_index = rng.sample(distribution);

    // Generate the collapsed vector
    let mut updated_wavefunction = bitvec![BitvecUnderlyingType, Msb0; 0; output_shape.2];
    updated_wavefunction.set(picked_collapse_index, true);

    // Write back to coefficient_matrix
    coefficient_matrix[start_index..start_index + output_shape.2]
        .copy_from_bitslice(&updated_wavefunction);
}

fn observe(
    coefficient_matrix: &mut BitVec<u32, Msb0>,
    probabilities: &[f64],
    output_shape: &(usize, usize, usize),
) -> ObserveResult {
    // Check for contradications, aka if any of the wave-masks are completely false
    let has_contradiction = iproduct!(0..output_shape.0, 0..output_shape.1)
        .par_bridge()
        .any(|(y, x)| {
            let start_index = ((y * output_shape.1) + x) * output_shape.2;
            coefficient_matrix[start_index..start_index + output_shape.2].not_any()
        });

    if has_contradiction {
        return ObserveResult::Contradiction;
    }

    // Calculate the entropy matrix (Do not par_iter - this garbles the indices)
    let entropy_matrix: Vec<f64> = iproduct!(0..output_shape.0, 0..output_shape.1)
        .map(|(y, x)| {
            let start_index = ((y * output_shape.1) + x) * output_shape.2;
            let wavefunction_mask = &coefficient_matrix[start_index..start_index + output_shape.2];
            if wavefunction_mask.count_ones() == 1 {
                return 0.0;
            };

            wavefunction_mask
                .iter()
                .zip(probabilities.iter())
                .map(|(mask, &probability)| if *mask { probability } else { 0.0 })
                .sum()
        })
        .collect();

    // Calculate total entropy
    let total_entropy: f64 = entropy_matrix.par_iter().sum();

    // Check if the wavefunction has collapsed
    if total_entropy == 0.0 {
        return ObserveResult::Collapse;
    }

    // Get minimum entropy from non-zero element(s)
    let min_entropy = entropy_matrix
        .par_iter()
        .filter(|e| **e > 0.0)
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    // Get the corresponding indices
    let min_entropy_indices: Vec<(usize, usize)> = iproduct!(0..output_shape.0, 0..output_shape.1)
        .par_bridge()
        .map(|(y, x)| ((y, x), entropy_matrix[(y * output_shape.1) + x]))
        .filter(|(_, value)| value == min_entropy)
        .map(|(index, _)| index)
        .collect();

    ObserveResult::Ok(min_entropy_indices, total_entropy)
}

fn propagate_cell(
    position: &(usize, usize),
    offset: &(i32, i32),
    offset_index: usize,
    coefficient_matrix: &mut BitVec<u32, Msb0>,
    rules: &[Vec<Vec<usize>>],
    output_shape: &(usize, usize, usize),
) -> bool {
    let adjacent_cell_position = (
        (position.0 as i32 + offset.0) as usize,
        (position.1 as i32 + offset.1) as usize,
    );

    // Get a list of valid patterns for the cells
    let start_index_adjacent =
        ((adjacent_cell_position.0 * output_shape.1) + adjacent_cell_position.1) * output_shape.2;
    let valid_pattern_mask_adjacent =
        &coefficient_matrix[start_index_adjacent..start_index_adjacent + output_shape.2];
    let start_index_origin = ((position.0 * output_shape.1) + position.1) * output_shape.2;
    let valid_pattern_mask_original =
        &coefficient_matrix[start_index_origin..start_index_origin + output_shape.2];

    // Get a list of possible indices for the origin cell
    let possible_pattern_indices_in_origin_cell: Vec<usize> = valid_pattern_mask_original
        .iter()
        .enumerate()
        .par_bridge()
        // Filter for only possible elements
        .filter(|(_, mask)| **mask)
        // For the possible elements, get their indices
        .map(|(index, _)| index)
        .collect();

    // Based on the possible state(s) of the origin cell, compute a list of allowed index-states for the new offset cell
    let mut allowed_states_adjacent = bitvec![BitvecUnderlyingType, Msb0; 0; output_shape.2];
    possible_pattern_indices_in_origin_cell
        .iter()
        // For each state of the origin cell, retreive the allowed states of the adj. cell
        .flat_map(|&pattern_index| rules[pattern_index][offset_index].clone())
        .for_each(|index| allowed_states_adjacent.set(index, true));

    // Mask the possible against the allowed states
    let weighted_wavemask_adjacent = allowed_states_adjacent & valid_pattern_mask_adjacent;

    // Check if we changed anything (xor and count zeros would also be possible)
    // if (weighted_wavemask_adjacent.clone() ^ valid_pattern_mask_adjacent).count_ones() == 0 { return false }
    if weighted_wavemask_adjacent.contains(valid_pattern_mask_adjacent) {
        return false;
    }

    // Otherwise write back
    coefficient_matrix[start_index_adjacent..start_index_adjacent + output_shape.2]
        .copy_from_bitslice(&weighted_wavemask_adjacent);

    true
}

fn propagate(
    position: (usize, usize),
    coefficient_matrix: &mut BitVec<u32, Msb0>,
    rules: &[Vec<Vec<usize>>],
    offsets: &[(i32, i32)],
    output_shape: &(usize, usize, usize),
) {
    let mut propagation_queue = vec![position];

    while let Some(position) = propagation_queue.pop() {
        for (offset_index, offset) in offsets.iter().enumerate() {
            let adjacent_cell_position = (
                (position.0 as i32 + offset.0) as usize,
                (position.1 as i32 + offset.1) as usize,
            );
            // Check if the cell is collapsed
            if in_bounds(&adjacent_cell_position, output_shape)
                && !is_cell_collapsed(&adjacent_cell_position, coefficient_matrix, output_shape)
            {
                // Propagate in the chosen direction
                if propagate_cell(
                    &position,
                    offset,
                    offset_index,
                    coefficient_matrix,
                    rules,
                    output_shape,
                ) {
                    // We've updated the cell's probabilities, so we need to process it as well.
                    propagation_queue.push(adjacent_cell_position);
                }
            }
        }
    }
}

fn main() -> Result<()> {
    let args = CLIArguments::parse();

    let offsets = generate_pattern_offsets(args.n as i32);
    println!("Generated {} offsets.", offsets.len());

    // Load Image, generate palette and encode
    let sample = load_image_from_path(&args.sample_path)?;
    let (palette, reverse_palette) = sample.generate_palette();
    let sample = sample.encode_to_palette(&reverse_palette);

    // Generate Patterns
    let pattern_size = args.n as usize;
    let (patterns, _, pattern_probabilities) =
        generate_patterns_frequencies_probabilities_from_image(
            &sample,
            pattern_size,
            if args.symmetry { Symmetry::all() } else { Symmetry::empty() }
        );
    println!("Generated {} unique patterns", patterns.len());

    // Generate Rules
    let rules = generate_rules(&patterns, &offsets);
    println!(
        "Generated {} rules",
        rules.iter().fold(0, |acc, rl| acc + rl.len())
    );
    let empty_rules = rules.iter().fold(0, |acc, rl| {
        acc + rl
            .iter()
            .fold(0, |acc, e| if e.is_empty() { acc + 1 } else { acc })
    });
    println!("There's {} empty_rules", empty_rules);

    // Make sure the steps dir exists and is empty
    // TODO! configurable via args
    let step_dir_path = "steps";
    if Path::new(step_dir_path).is_dir() {
        fs::remove_dir_all(step_dir_path)?;
    }
    fs::create_dir(step_dir_path)?;

    // Initialize coefficient matrix
    let output_shape = (args.result_height, args.result_width, patterns.len());
    let mut coefficient_matrix =
        bitvec!(BitvecUnderlyingType, Msb0; 1; output_shape.0 * output_shape.1 * output_shape.2);

    let mut rng = WyRand::seed_from_u64(5);

    let maximum_entropy = (output_shape.0 * output_shape.1) as f64;
    let mut progress_bar = progress::Bar::new();
    progress_bar.set_job_title("Starting Wavefunction collapse...");

    let mut last_selected_indices = Vec::new();
    let mut last_coefficient_matrix = coefficient_matrix.clone();

    // Loop while we get an okay result
    let mut iteration_index = 0;
    let result = loop {
        let result = observe(
            &mut coefficient_matrix,
            &pattern_probabilities,
            &output_shape,
        );
        // Check if the wave-function has collapsed, or if we've come accross a Contradiction
        match result {
            ObserveResult::Ok(min_entropy_indices, entropy) => {
                // Select one of the min-entropy-indices to collapse
                // TODO! Technically we should check if this has any elements left.
                let selected_index = min_entropy_indices[rng.gen_range(0..min_entropy_indices.len())];

                // Save the indices and matrix
                last_selected_indices = min_entropy_indices.iter().filter(| &&e | e != selected_index).copied().collect();
                last_coefficient_matrix = coefficient_matrix.clone();

                // Collapse the selected cell
                collapse_single_cell(
                    &selected_index,
                    &mut coefficient_matrix,
                    &pattern_probabilities,
                    &mut rng,
                    &output_shape,
                );

                progress_bar.set_job_title(&format!("Collapsing Wavefunction, current entropy {:.02}", entropy));
                progress_bar.reach_percent(((1.0 - entropy/maximum_entropy) * 100.0) as i32);
                propagate(
                    selected_index,
                    &mut coefficient_matrix,
                    &rules,
                    &offsets,
                    &output_shape,
                )
            },
            ObserveResult::Contradiction => {
                if !last_selected_indices.is_empty() {
                    // Backtrack matrix
                    coefficient_matrix.copy_from_bitslice(&last_coefficient_matrix);

                    let selected_index = last_selected_indices[rng.gen_range(0..last_selected_indices.len())];
                    
                    // Update the indices
                    last_selected_indices = last_selected_indices.iter().filter(| &&e | e != selected_index).copied().collect();

                    // Collapse the selected cell
                    collapse_single_cell(
                        &selected_index,
                        &mut coefficient_matrix,
                        &pattern_probabilities,
                        &mut rng,
                        &output_shape,
                    );

                    progress_bar.set_job_title(&format!("Encountered Contradiction, backtracking..."));
                    propagate(
                        selected_index,
                        &mut coefficient_matrix,
                        &rules,
                        &offsets,
                        &output_shape,
                    )
                } else {
                    break result;
                }
            }
            _ => {
                break result;
            }
        };
        // println!("Rendering intermediate image...");
        let step_image = render_image_from_coefficients(&coefficient_matrix, &patterns, &palette, &output_shape, pattern_size);
        save_image_to_path(&step_image, Path::new(&format!("steps/iteration_{}.png", iteration_index)))?;
        iteration_index += 1;
    };
    println!();

    match result {
        ObserveResult::Collapse => {
            println!("Collapse!");
            // let final_image = render_image_from_coefficients(&coefficient_matrix, &patterns, &palette);
            // save_image_to_path(&final_image, Path::new(&format!("steps/iteration_{}.png", iteration_index)))?;
        }
        ObserveResult::Contradiction => println!("Contradiction!"),
        _ => println!("Uh, oh, we shouldn't be here..."),
    }

    Ok(())
}
