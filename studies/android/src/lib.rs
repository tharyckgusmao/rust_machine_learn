// pub mod model;

// #![cfg(target_os = "android")]
// #![allow(non_snake_case)]

use std::os::raw::{ c_char };
use std::ffi::{ CString, CStr };
use burn_wgpu::{ Wgpu, WgpuDevice };

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

// #[no_mangle]
// pub extern fn rust_greeting(to: *const c_char) -> *mut c_char {
//     let c_str = unsafe { CStr::from_ptr(to) };
//     let recipient = c_str.to_str().unwrap_or_else(|_| "there");

//     CString::new("From rust: Hello ".to_owned() + recipient)
//         .unwrap()
//         .into_raw()
// }

#[no_mangle]
pub extern "C" fn example() {
    println!("Hello from Rust's example function!");
}

#[cfg(target_os = "android")]
#[allow(non_snake_case)]
pub mod android {
    extern crate jni;

    use jni::objects::{ JObject, JValue };

    use super::*;
    use self::jni::JNIEnv;
    use self::jni::objects::{ JClass, JString };

    #[no_mangle]
    pub unsafe extern fn Java_com_example_androidrust_NativeLib_example(
        mut env: JNIEnv,
        _: JClass
    ) {
        println!("Hello from Rust's stdout. This message is sent to /dev/null by Android.");
        let device = WgpuDevice::default();
        let message = format!("device available{:?}", device);
        let tag = env.new_string("AndroidRust").unwrap();
        let message = env.new_string(message).unwrap();
        let class = env.find_class("android/util/Log").unwrap();
        env.call_static_method(
            class,
            "d",
            "(Ljava/lang/String;Ljava/lang/String;)I",
            &[JValue::Object(&tag), JValue::Object(&message)]
        ).unwrap();
    }
}
