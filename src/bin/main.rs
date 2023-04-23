/// onnxruntime-rs example
use std::{path::PathBuf, sync::Arc};

use image::{GrayImage, Pixel};
use ndarray::{Array2, Array4};
use ndarray_stats::QuantileExt;
use ort::{
    tensor::{DynOrtTensor, FromArray, InputTensor, OrtOwnedTensor},
    Environment, ExecutionProvider, GraphOptimizationLevel, OrtResult, Session, SessionBuilder,
};
use structopt::StructOpt;

#[derive(Debug, structopt::StructOpt)]
struct Opt {
    #[structopt(long = "input", short, default_value = "input")]
    input_dir: PathBuf,
    #[structopt(long = "output", short, default_value = "output")]
    output_dir: PathBuf,
    #[structopt(long = "model", short, default_value = "models/model-small.onnx")]
    model_file: PathBuf,
}

struct Shape {
    w: usize,
    h: usize,
}

/// ndarray to image for depth map visualization
fn array_to_image(arr: Array2<u8>) -> GrayImage {
    assert!(arr.is_standard_layout());

    let (height, width) = arr.dim();
    let raw = arr.into_raw_vec();

    GrayImage::from_raw(width as u32, height as u32, raw)
        .expect("container should have the right size for the image dimensions")
}

/// Input RGBImage
fn predict(
    session: &Session,
    array: Array4<f32>,
    shape: &Shape,
) -> OrtResult<image::ImageBuffer<image::Luma<u8>, Vec<u8>>> {
    let input_tensor_values = [InputTensor::from_array(array.into_dyn())];

    // run the model on the input -> extract output tensor -> convert to Vec<f32>
    let outputs: Vec<DynOrtTensor<ndarray::Dim<ndarray::IxDynImpl>>> =
        session.run(input_tensor_values)?;
    let output: OrtOwnedTensor<_, _> = outputs[0].try_extract::<f32>()?;
    let depth_map = output
        .view()
        .clone()
        .into_shape((shape.w, shape.h))
        .unwrap();
    // let v: Vec<f32>  = output.view().iter().copied().collect::<Vec<_>>();
    let max = depth_map.max().unwrap();
    let grayscale = { depth_map.mapv(|x| (x / max * 255.0) as u8) };
    let grayscale_image = array_to_image(grayscale);
    Ok(grayscale_image)
}

fn main() -> anyhow::Result<()> {
    let opt = Opt::from_args();

    if !opt.input_dir.is_dir() {
        return Err(anyhow::format_err!(
            "input directory {:?} is not a directory",
            opt.input_dir
        ));
    }
    if !opt.output_dir.is_dir() {
        std::fs::create_dir(&opt.output_dir)?;
        log::info!("created output directory {:?}", opt.output_dir);
    }

    let shape = Shape { w: 256, h: 256 };
    tracing_subscriber::fmt::init();

    let environment = Arc::new({
        let mut builder = Environment::builder().with_name("MiDaS");
        if ExecutionProvider::cuda().is_available() {
            builder = builder.with_execution_providers([ExecutionProvider::cuda()]);
        }
        builder.build()?
    });

    let session = SessionBuilder::new(&environment)?
        .with_optimization_level(GraphOptimizationLevel::Level1)?
        .with_intra_threads(1)?
        .with_model_from_file(&opt.model_file)?;

    let entries = std::fs::read_dir(&opt.input_dir)
        .expect(&format!("failed to read directory {:?}", opt.input_dir));
    for entry in entries.filter_map(|x| x.ok()) {
        let path: PathBuf = entry.path();
        if !path.is_file() {
            continue;
        }
        // readimage -> resize -> normalize -> make input tensor
        let image = image::open(&path).unwrap().to_rgb8();
        let resized = image::imageops::resize(
            &image,
            shape.w as u32,
            shape.h as u32,
            ::image::imageops::FilterType::Triangle,
        );
        let array = ndarray::Array::from_shape_fn((1, 3, shape.w, shape.h), |(_, c, j, i)| {
            let pixel = resized.get_pixel(i as u32, j as u32);
            let channels = pixel.channels();

            // range [0, 255] -> range [0, 1]
            (channels[c] as f32) / 255.0
        });
        let grayscale_image = predict(&session, array, &shape)?;
        let outpath = opt.output_dir.join(path.file_name().unwrap());
        grayscale_image.save(outpath).unwrap();
    }

    // println!("result: {grayscale_image:?}");
    Ok(())
}
