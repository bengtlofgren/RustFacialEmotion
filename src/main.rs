use tch::Tensor;
#[allow(unused_imports)]
use tch::vision::{image, imagenet};
use anyhow::{bail, Result};
use std::ops::{Sub, Div}; // Add this at the top with other imports

fn main() {
    enet_b2().unwrap();
}

pub fn image_pytorch() -> anyhow::Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let (model_file, image_file) = match args.as_slice() {
        [_, m, i] => (m.to_owned(), i.to_owned()),
        _ => bail!("usage: main model.pt image.jpg"),
    };
    let image = imagenet::load_image_and_resize(image_file, 128, 128)?;
    let model = tch::CModule::load(&model_file).map_err(|e| anyhow::anyhow!("Failed to load model: {}", e))?;
    let output = model.forward_ts(&[image.unsqueeze(0)])?.softmax(-1, tch::Kind::Float);
    for (probability, class) in imagenet::top(&output, 5).iter() {
        println!("{:50} {:5.2}%", class, 100.0 * probability)
    }
    Ok(())
}

#[derive(Debug)]
pub enum Emotion {
    Anger,
    Contempt,
    Disgust,
    Fear,
    Happiness,
    Neutral,
    Sadness,
    Surprise,
}

impl Emotion {
    pub fn from_index(index: i64) -> Self {
        match index {
            0 => Emotion::Anger,
            1 => Emotion::Contempt,
            2 => Emotion::Disgust,
            3 => Emotion::Fear,
            4 => Emotion::Happiness,
            5 => Emotion::Neutral,
            6 => Emotion::Sadness,
            7 => Emotion::Surprise,
            _ => panic!("Invalid emotion index"),
        }
    }
}

// Implement Display trait manually
impl std::fmt::Display for Emotion {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        // Use match to convert enum variant to string
        write!(f, "{}", match self {
            Emotion::Anger => "Anger",
            Emotion::Contempt => "Contempt",
            Emotion::Disgust => "Disgust",
            Emotion::Fear => "Fear",
            Emotion::Happiness => "Happiness",
            Emotion::Neutral => "Neutral",
            Emotion::Sadness => "Sadness",
            Emotion::Surprise => "Surprise",
        })
    }
}

pub fn enet_b2() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let (model_file, image_file) = match args.as_slice() {
        [_, m, i] => (m.to_owned(), i.to_owned()),
        _ => bail!("usage: main model.pt image.jpg"),
    };
    let normalised_image = preprocess_image(&image_file)?;
    let model = tch::CModule::load(&model_file).map_err(|e| anyhow::anyhow!("Failed to load model: {}", e))?;
    let input = normalised_image.unsqueeze(0);
    let output = model.forward_ts(&[input])?.softmax(-1, tch::Kind::Float);
    // The output tensor has shape [1, 8] (1 batch, 8 classes)
    // We can just flatten it to get all probabilities
    let values = output.view([-1]);  // Flattens to [8]

    // values contains the actual probabilities
    // view([-1]) flattens the tensor into a 1D array
    let values = values.view([-1]);

    // Create indices array from 0 to 7
    for i in 0..8 {
        let probability = values.double_value(&[i]);
        println!("{}: {:5.2}%", Emotion::from_index(i), 100.0 * probability);
    }
    Ok(())
}


pub fn preprocess_image(path: &str) -> Result<Tensor> {
    // Load and resize image
    let image = imagenet::load_image_and_resize(path, 260, 260)?;
    let image = image.view([3, 260, 260]);
    
    // Normalize using ImageNet mean and std
    let mean = Tensor::from_slice(&[0.485f32, 0.456, 0.406])
        .view([3, 1, 1]);
    let std = Tensor::from_slice(&[0.229f32, 0.224, 0.225])
        .view([3, 1, 1]);
    
    let normalised = image.sub(&mean).div(&std);
    
    Ok(normalised)
}
