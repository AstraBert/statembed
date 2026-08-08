use statemebed::load::load_safetensors_file;

fn main() {
    let path = "testfiles/model.safetensors";
    let (header, tensor) = load_safetensors_file(path).expect("Should load file");
    println!("{:#?}\n{:?}", header, tensor.len());
}
