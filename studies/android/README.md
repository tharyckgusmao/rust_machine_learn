export ANDROID_HOME=/home/tharyckgusmaometzker/Android/Sdk
export PATH=${PATH}:${ANDROID_HOME}/cmdline-tools/latest/bin:${ANDROID_HOME}/platform-tools
export NDK_HOME=$ANDROID_HOME/ndk

//test
export ANDROID_NDK_HOME=$HOME/Android/Sdk/ndk/27.1.1229700 
export PATH=$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin:$PATH

sudo apt-get install build-essential
sudo apt-get install gcc-arm-linux-gnueabi

${NDK_HOME}/27.1.12297006/build/tools/make_standalone_toolchain.py --arch arm64 --install-dir NDK/arm64
${NDK_HOME}/27.1.12297006/build/tools/make_standalone_toolchain.py --arch arm --install-dir NDK/arm
${NDK_HOME}/27.1.12297006/build/tools/make_standalone_toolchain.py --arch x86 --install-dir NDK/x86

rustup target add armv7-linux-androideabi
rustup target add --toolchain nightly x86_64-linux-android

cargo build --target armv7-linux-androideabi --release

export PATH=$PATH:/home/tharyckgusmaometzker/Android/Sdk/ndk/27.1.12297006/toolchains/llvm/prebuilt/linux-x86_64/bin


rustup target add \
    aarch64-linux-android \
    armv7-linux-androideabi \
    x86_64-linux-android \
    i686-linux-android


cargo install cargo-ndk

cargo ndk -t armeabi-v7a -t arm64-v8a -o ./jniLibs build --release
cargo ndk -t armeabi-v7a build

//Header file example to C
javac -h . NativeLib.java