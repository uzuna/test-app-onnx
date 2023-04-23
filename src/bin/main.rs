/// onnxruntime-rs example
use std::sync::Arc;

use image::{Pixel, GrayImage};
use ndarray::Array2;
use ndarray_stats::QuantileExt;
use ort::{OrtResult, ExecutionProvider, Environment, SessionBuilder, GraphOptimizationLevel, tensor::{InputTensor, DynOrtTensor, FromArray, OrtOwnedTensor}};

/// ndarray to image for depth map visualization
fn array_to_image(arr: Array2<u8>) -> GrayImage {
    assert!(arr.is_standard_layout());

    let (height, width) = arr.dim();
    let raw = arr.into_raw_vec();

    GrayImage::from_raw(width as u32, height as u32, raw)
        .expect("container should have the right size for the image dimensions")
}

fn main() -> OrtResult<()> {
    let width = 256;
    tracing_subscriber::fmt::init();

	let environment = Arc::new(
		{
            let mut builder = Environment::builder().with_name("MiDaS");
            if ExecutionProvider::cuda().is_available() {
                builder = builder.with_execution_providers([ExecutionProvider::cuda()]);
            }
			builder.build()?
        }
	);


	let session = SessionBuilder::new(&environment)?
		.with_optimization_level(GraphOptimizationLevel::Level1)?
		.with_intra_threads(1)?
		.with_model_from_file("model-small.onnx")?;

    // readimage -> resize -> normalize -> make input tensor
    let image = image::open("qb35_20180914_mlbt1410.jpg").unwrap().to_rgb8();
    let resized =
        image::imageops::resize(&image, width, width, ::image::imageops::FilterType::Triangle);
    let array = ndarray::Array::from_shape_fn((1, 3, width as usize, width as usize), |(_, c, j, i)| {
        let pixel = resized.get_pixel(i as u32, j as u32);
        let channels = pixel.channels();

        // range [0, 255] -> range [0, 1]
        (channels[c] as f32) / 255.0
    });
    let input_tensor_values = [InputTensor::from_array(array.into_dyn())];

    // run the model on the input -> extract output tensor -> convert to Vec<f32>
    let outputs: Vec<DynOrtTensor<ndarray::Dim<ndarray::IxDynImpl>>> = session.run(input_tensor_values)?;
    let output: OrtOwnedTensor<_, _> = outputs[0].try_extract::<f32>()?;
    let depth_map = output.view().clone().into_shape((width as usize, width as usize)).unwrap();
    // let v: Vec<f32>  = output.view().iter().copied().collect::<Vec<_>>();
    let max = depth_map.max().unwrap();
    let grayscale = {
        depth_map.mapv(|x| (x / max * 255.0) as u8)
    };
    let grayscale_image = array_to_image(grayscale);
    grayscale_image.save("grayscale.png").unwrap();
    // println!("result: {grayscale_image:?}");
    Ok(())
}