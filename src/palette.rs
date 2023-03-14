pub fn generate_palette_from_image(sample: &Image) -> (Vec<[u16; 4]>, HashMap<[u16; 4], usize>) {
  let palette_set: HashSet<[u16; 4]> = sample.iter().copied().collect();
  let reverse_palette: HashMap<[u16; 4], usize> = palette_set
      .borrow()
      .into_iter()
      .copied()
      .enumerate()
      .map(|(idx, pixel)| (pixel, idx))
      .collect();

  let mut palette: Vec<(usize, [u16; 4])> =
      reverse_palette.iter().map(|(&p, &idx)| (idx, p)).collect();
  palette.sort_by_key(|e| e.0);
  let palette = palette.into_iter().map(|(_, p)| p).collect();

  (palette, reverse_palette)
}