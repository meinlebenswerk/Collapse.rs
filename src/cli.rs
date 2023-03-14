use clap::{arg, command, Parser};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct CLIArguments {
   /// Path to the sample file
   #[arg(short, long)]
   pub sample_path: String,

   /// Generated Image width
   #[arg(long, default_value_t = 64)]
   pub result_width: usize,

   /// Generated Image height
   #[arg(long, default_value_t = 64)]
   pub result_height: usize,

   /// Pattern size parameter, generated patters are NxN pixels
   #[arg(short, long, default_value_t = 3)]
   pub n: u32,

   /// Consider edges
   #[arg(short, long, default_value_t = false)]
   pub consider_edges: bool,
}
