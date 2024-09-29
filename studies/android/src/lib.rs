// pub mod model;

// use crate::model;
// use crate::{ data::DataLoader, model::vgg };

// #![cfg(target_os = "android")]
// #![allow(non_snake_case)]

use std::os::raw::{ c_char };
use std::ffi::{ CString, CStr };

// fn main(image_path: &str) -> i64 {
//     let device = Device::cuda_if_available();
//     let mut vs = nn::VarStore::new(device);
//     let vgg_net = vgg(&vs.root(), 2, false);

//     let net = nn
//         ::seq_t()
//         .add(vgg_net)
//         .add(nn::linear(vs.root() / "fc", 4096, 2, Default::default()));
//     vs.load("weights/best_model.ot").unwrap();

//     let image = vision::imagenet::load_image_and_resize224(image_path).unwrap().unsqueeze(0);
//     let out = net.forward_t(&image.to_device(device), false);
//     let prediction = out.argmax(1, false);

//     i64::try_from(prediction).unwrap()
// }

// #[cfg(target_os = "android")]
// #[allow(non_snake_case)]
// pub mod android {
//     extern crate jni;

//     use super::*;
//     use self::jni::JNIEnv;
//     use self::jni::objects::{ JClass, JString };
//     use self::jni::sys::{ jstring };

//     #[no_mangle]
//     pub unsafe extern fn Java_com_solana_mobilewalletadapter_fakedapp_RustGreetings_greeting(
//         mut env: JNIEnv,
//         _: JClass,
//         java_pattern: JString
//     ) -> jstring {
//         // Our Java companion code might pass-in "world" as a string, hence the name.
//         let world = rust_greeting(
//             env.get_string(java_pattern).expect("invalid pattern string").as_ptr()
//         );
//         // Retake pointer so that we can use it below and allow memory to be freed when it goes out of scope.
//         let world_ptr = CString::from_raw(world);
//         let output = env
//             .new_string(world_ptr.to_str().unwrap())
//             .expect("Couldn't create java string!");

//         output.into_raw()
//     }
// }

// #[no_mangle]
// pub extern fn rust_greeting(to: *const c_char) -> *mut c_char {
//     let c_str = unsafe { CStr::from_ptr(to) };
//     let recipient = c_str.to_str().unwrap_or_else(|_| "there");

//     CString::new("From rust: Hello ".to_owned() + recipient)
//         .unwrap()
//         .into_raw()
// }

#[no_mangle]
pub extern "C" fn Java_com_example_androidrust_NativeLib_example() {
    println!("Hello from Rust's example function!");
}
